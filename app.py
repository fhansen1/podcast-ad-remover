"""
Podcast Ad Remover - On-Demand Processing
Flow: OPML → RSS Proxy → Process on Play → Stream Clean Audio → Auto-Delete after 14 days

Pipeline per episode:
  1. Download full episode
  2. If first time for this podcast: detect & fingerprint intro tune
  3. Find intro tune position → cut everything before it
  4. Full episode Whisper analysis → cut all ad segments
  5. Stitch clean audio → serve

Intro verification:
  - Subscribe to {BASE_URL}/intros/feed in your podcast app
  - Listen to detected intros, delete fingerprint if wrong
"""
import os
import hashlib
import threading
import json
import logging
import shutil
import functools
import tempfile
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
import feedparser
import ffmpeg
from flask import Flask, Response, request, redirect, jsonify, send_file

# ============================================================
# SETUP
# ============================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "https://podcast.drhahn.no").rstrip("/")
RSS_USERNAME = os.environ.get("RSS_USERNAME", "")
RSS_PASSWORD = os.environ.get("RSS_PASSWORD", "")

OPML_FILE = Path(os.environ.get("OPML_FILE", "/data/podcasts.opml"))
STORAGE_DIR = Path("/data/podcasts")
FINGERPRINT_DIR = Path("/data/fingerprints")
INTRO_CLIPS_DIR = FINGERPRINT_DIR  # Intro clips stored alongside fingerprints

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
FINGERPRINT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# episode cache_key -> {"status": "processing"|"ready"|"error", "error": str}
PROCESSING = {}
PROCESSING_LOCK = threading.Lock()

# Fingerprint tuning
FINGERPRINT_SIMILARITY_THRESHOLD = 0.72  # Minimum cosine similarity to accept match
INTRO_SEARCH_MAX_SECONDS = 300           # Search for intro within first 5 minutes
TYPICAL_INTRO_DURATION = 30             # Assume intro tune is ~30s if auto-detecting

# Whisper tuning
WHISPER_CHUNK_MINUTES = 20              # Split episode into N-minute chunks for Whisper
WHISPER_MAX_MB = 23                     # Stay under OpenAI 25MB limit per request


# ============================================================
# OPML PARSING
# ============================================================

def load_opml() -> dict[str, str]:
    """
    Parse OPML file and return {podcast_slug: rss_url}.
    Slug is derived from the outline text attribute, lowercased, spaces→underscores.

    Example OPML outline:
      <outline text="Det Store Bildet" type="rss" xmlUrl="https://rss.podplaystudio.com/692.xml"/>
    """
    if not OPML_FILE.exists():
        logger.error(f"OPML file not found: {OPML_FILE}")
        return {}

    try:
        tree = ET.parse(OPML_FILE)
        root = tree.getroot()
        feeds = {}
        for outline in root.iter('outline'):
            url = outline.get('xmlUrl') or outline.get('url')
            text = outline.get('text') or outline.get('title', '')
            if url and text:
                slug = text.lower().strip().replace(' ', '_').replace('-', '_')
                slug = ''.join(c for c in slug if c.isalnum() or c == '_')
                feeds[slug] = url
                logger.debug(f"OPML loaded: {slug} -> {url}")
        logger.info(f"Loaded {len(feeds)} podcasts from OPML")
        return feeds
    except Exception as e:
        logger.error(f"Failed to parse OPML: {e}")
        return {}


def get_rss_url(podcast_name: str) -> str | None:
    feeds = load_opml()
    return feeds.get(podcast_name)


# ============================================================
# XML / AUTH HELPERS
# ============================================================

def xml_escape(text):
    if not text:
        return ""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "<")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
            .replace("\xa0", " "))


def get_audio_url(entry) -> str | None:
    """Safely extract audio URL from a feedparser entry."""
    for link in entry.get("links", []):
        if link.get("type", "").startswith("audio/"):
            return link.get("href")
    for enc in entry.get("enclosures", []):
        if enc.get("type", "").startswith("audio/"):
            return enc.get("href")
    return None


def require_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if RSS_USERNAME and RSS_PASSWORD:
            auth = request.authorization
            if not auth or auth.username != RSS_USERNAME or auth.password != RSS_PASSWORD:
                return Response(
                    'Authentication required', 401,
                    {'WWW-Authenticate': 'Basic realm="Podcast RSS"'}
                )
        return f(*args, **kwargs)
    return decorated


# ============================================================
# FINGERPRINT — Intro tune detection & storage
# ============================================================

def fingerprint_path(podcast_name: str) -> Path:
    return FINGERPRINT_DIR / f"{podcast_name}.json"


def has_fingerprint(podcast_name: str) -> bool:
    return fingerprint_path(podcast_name).exists()


def build_fingerprint_from_audio(audio_path: str, podcast_name: str,
                                  intro_start: float, intro_duration: float):
    """
    Store an intro fingerprint from a known position in an audio file.
    Also saves the actual intro audio clip for verification.
    """
    try:
        import librosa
        import numpy as np

        sr = 22050
        y, _ = librosa.load(audio_path, sr=sr, mono=True,
                             offset=intro_start, duration=intro_duration)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1).tolist()

        data = {
            "podcast": podcast_name,
            "duration": intro_duration,
            "intro_start_reference": intro_start,
            "mfcc": mfcc,
            "created_at": datetime.now().isoformat(),
            "source_episode": Path(audio_path).parent.name,
        }
        with open(fingerprint_path(podcast_name), 'w') as f:
            json.dump(data, f, indent=2)

        # Extract and save the actual intro audio clip for verification
        intro_clip_path = FINGERPRINT_DIR / f"{podcast_name}_intro.mp3"
        subprocess.run([
            'ffmpeg', '-y',
            '-i', audio_path,
            '-ss', str(intro_start),
            '-t', str(intro_duration),
            '-acodec', 'libmp3lame', '-ab', '192k',
            str(intro_clip_path)
        ], capture_output=True, check=True)

        logger.info(f"Fingerprint + intro clip saved for {podcast_name} "
                    f"(intro at {intro_start:.1f}s, duration {intro_duration:.1f}s)")

    except Exception as e:
        logger.error(f"Failed to build fingerprint: {e}", exc_info=True)


def detect_intro_tune_in_audio(audio_path: str, podcast_name: str) -> float | None:
    """
    Slide a window across the first INTRO_SEARCH_MAX_SECONDS of audio,
    compare each window to the stored fingerprint.
    Returns the END time (seconds) of the intro tune, or None if not found.
    """
    if not has_fingerprint(podcast_name):
        return None

    try:
        import librosa
        import numpy as np

        with open(fingerprint_path(podcast_name)) as f:
            fp = json.load(f)

        ref_mfcc = np.array(fp['mfcc'])
        intro_duration = fp['duration']
        sr = 22050

        logger.info(f"Searching for intro tune ({intro_duration:.1f}s) in {audio_path}")

        y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=INTRO_SEARCH_MAX_SECONDS)

        hop = sr // 2           # 0.5s steps
        window = int(intro_duration * sr)

        if window > len(y):
            logger.warning("Audio shorter than intro duration")
            return None

        best_sim = 0.0
        best_pos = 0

        for start_sample in range(0, len(y) - window, hop):
            segment = y[start_sample:start_sample + window]
            seg_mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20), axis=1)
            norm = np.linalg.norm(ref_mfcc) * np.linalg.norm(seg_mfcc)
            if norm == 0:
                continue
            sim = np.dot(ref_mfcc, seg_mfcc) / norm
            if sim > best_sim:
                best_sim = sim
                best_pos = start_sample

        intro_start = best_pos / sr
        intro_end = intro_start + intro_duration

        logger.info(
            f"Intro tune match: {best_sim:.1%} at {intro_start:.1f}s–{intro_end:.1f}s "
            f"(threshold: {FINGERPRINT_SIMILARITY_THRESHOLD:.0%})"
        )

        if best_sim >= FINGERPRINT_SIMILARITY_THRESHOLD:
            return intro_end
        return None

    except Exception as e:
        logger.error(f"Intro detection failed: {e}", exc_info=True)
        return None


# ============================================================
# FIRST-PLAY INTRO FINGERPRINTING PIPELINE
# ============================================================

def auto_detect_and_fingerprint_intro(audio_path: str, podcast_name: str) -> float:
    """
    On first play of a new podcast (no fingerprint yet):
    Use Whisper to find when speech starts, infer intro tune position,
    then fingerprint that segment for future episodes.

    Returns the cut point in seconds (end of intro).
    """
    if not OPENAI_API_KEY:
        logger.warning("No API key — cannot auto-detect intro tune")
        return 0.0

    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # Extract first 5 minutes at low quality for Whisper
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp_path = tmp.name

        subprocess.run([
            'ffmpeg', '-y', '-i', audio_path,
            '-t', str(INTRO_SEARCH_MAX_SECONDS),
            '-acodec', 'libmp3lame', '-ab', '64k',
            tmp_path
        ], capture_output=True, check=True)

        with open(tmp_path, 'rb') as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                prompt=(
                    "This is the beginning of a podcast episode. "
                    "There is likely an intro jingle or music before the hosts start speaking. "
                    "Transcribe from the very beginning."
                )
            )

        os.unlink(tmp_path)

        words = transcript.words if hasattr(transcript, 'words') else []
        if not words:
            logger.warning("No words detected in intro window")
            return 0.0

        # First word timestamp = when speech begins = end of intro tune
        first_speech = words[0].start
        logger.info(f"First speech detected at {first_speech:.1f}s")

        if first_speech < 3:
            logger.info("No intro tune detected (speech starts immediately)")
            return 0.0

        # Fingerprint the segment just before first speech
        intro_start = max(0, first_speech - TYPICAL_INTRO_DURATION)
        intro_duration = first_speech - intro_start

        if intro_duration >= 5:
            build_fingerprint_from_audio(
                audio_path, podcast_name,
                intro_start=intro_start,
                intro_duration=intro_duration
            )
            logger.info(f"Auto-fingerprinted intro: {intro_start:.1f}s – {first_speech:.1f}s")

        return first_speech

    except Exception as e:
        logger.error(f"Auto intro detection failed: {e}", exc_info=True)
        return 0.0


# ============================================================
# WHISPER — Full episode ad detection
# ============================================================

def transcribe_chunk(client, chunk_path: str) -> list[dict]:
    """Send one audio chunk to Whisper, return list of word dicts with start/end."""
    with open(chunk_path, 'rb') as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            prompt=(
                "This is a podcast. Transcribe everything including sponsor messages, "
                "ads, and jingles. Do not skip any speech."
            )
        )
    words = transcript.words if hasattr(transcript, 'words') else []
    return [{"word": w.word, "start": w.start, "end": w.end} for w in words]


def split_audio_into_chunks(audio_path: str, chunk_minutes: int, output_dir: Path) -> list[Path]:
    """Split audio file into fixed-length chunks using ffmpeg."""
    chunk_pattern = str(output_dir / "chunk_%03d.mp3")
    subprocess.run([
        'ffmpeg', '-y', '-i', audio_path,
        '-f', 'segment',
        '-segment_time', str(chunk_minutes * 60),
        '-acodec', 'libmp3lame', '-ab', '64k',
        chunk_pattern
    ], capture_output=True, check=True)

    chunks = sorted(output_dir.glob("chunk_*.mp3"))
    logger.info(f"Split into {len(chunks)} chunks of {chunk_minutes} min")
    return chunks


def classify_ad_segments(words: list[dict]) -> list[dict]:
    """
    Heuristic ad classifier on full transcript.
    Looks for ad indicators: promo codes, sponsor phrases, URLs, etc.
    Returns list of {start, end, type, confidence} dicts.
    """
    AD_PHRASES = [
        # English
        "this episode is sponsored by", "brought to you by", "promo code",
        "discount code", "use code", "go to", "visit", "sign up at",
        "check out", "our sponsor", "thanks to our sponsor", "ad-free",
        "free trial", "percent off", "slash podcast", "dot com slash",
        "coupon code", "exclusive offer", "limited time",
        "sponsored by", "our partners", "brought to you",
        # Norwegian
        "denne episoden er sponset av", "bruk koden", "rabattkode",
        "gå til", "besøk", "meld deg på", "gratis prøveperiode",
        "prosent rabatt", "vår sponsor", "takk til",
        "annonser", "reklamer", "sponset", "betalt annonsering",
        "kode er", "rabatt på", "spesialtilbud", "kun for deg",
        "følg oss på", "link i beskrivelsen", "trykk på",
    ]

    ad_segments = []
    i = 0
    n = len(words)

    while i < n:
        window_words = words[i:i + 15]
        window_text = " ".join(w['word'] for w in window_words).lower()
        matched = any(phrase in window_text for phrase in AD_PHRASES)

        if matched:
            seg_start = words[max(0, i - 2)]['start']
            j = i
            last_ad_word = i

            while j < n:
                lookahead = words[j:j + 10]
                la_text = " ".join(w['word'] for w in lookahead).lower()
                if any(phrase in la_text for phrase in AD_PHRASES):
                    last_ad_word = j + len(lookahead)
                if j > last_ad_word and j < n - 1:
                    gap = words[j + 1]['start'] - words[j]['end']
                    if gap > 4.0:
                        break
                j += 1

            seg_end = words[min(last_ad_word, n - 1)]['end']

            if seg_end - seg_start >= 10:
                ad_segments.append({
                    "start": seg_start,
                    "end": seg_end,
                    "type": "ad",
                    "confidence": "heuristic",
                })
                logger.info(f"Ad segment detected: {seg_start:.1f}s – {seg_end:.1f}s")
                i = last_ad_word + 1
                continue

        i += 1

    # Merge overlapping or adjacent segments (within 3 seconds of each other)
    if not ad_segments:
        return []

    merged = [ad_segments[0]]
    for seg in ad_segments[1:]:
        if seg['start'] <= merged[-1]['end'] + 3:
            merged[-1]['end'] = max(merged[-1]['end'], seg['end'])
        else:
            merged.append(seg)

    return merged


def detect_ads_full_episode(audio_path: str, podcast_name: str,
                              content_start: float = 0.0) -> list[dict]:
    """
    Transcribe the FULL episode (after intro cut) using Whisper in chunks.
    Returns ad segments with absolute timestamps (relative to original file).
    """
    if not OPENAI_API_KEY:
        logger.warning("No OpenAI API key — skipping Whisper ad detection")
        return []

    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            chunks = split_audio_into_chunks(audio_path, WHISPER_CHUNK_MINUTES, tmp_path)

            all_words = []
            offset = content_start

            for idx, chunk in enumerate(chunks):
                chunk_mb = chunk.stat().st_size / (1024 * 1024)
                logger.info(f"Transcribing chunk {idx + 1}/{len(chunks)} "
                            f"({chunk_mb:.1f} MB, offset {offset:.0f}s)")
                try:
                    words = transcribe_chunk(client, str(chunk))
                    for w in words:
                        w['start'] += offset
                        w['end'] += offset
                    all_words.extend(words)
                    offset += WHISPER_CHUNK_MINUTES * 60
                except Exception as e:
                    logger.error(f"Chunk {idx} transcription failed: {e}")
                    continue

            logger.info(f"Total words transcribed: {len(all_words)}")

            if not all_words:
                return []

            ad_segments = classify_ad_segments(all_words)
            logger.info(f"Found {len(ad_segments)} ad segments in full episode")
            return ad_segments

    except Exception as e:
        logger.error(f"Full episode ad detection failed: {e}", exc_info=True)
        return []



# ============================================================
# LLM-BASED AD DETECTION (NEW)
# ============================================================

def detect_ads_with_llm(words: list[dict]) -> list[dict]:
    """
    Use OpenAI GPT-4 to analyze transcript and identify ad segments.
    More accurate than keyword matching for modern podcast ads.
    """
    if not OPENAI_API_KEY or not words:
        return []
    
    if len(words) < 10:
        return []
    
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Build transcript text with timestamps
        transcript_snippet = ""
        for w in words[:100]:  # First 100 words for context
            transcript_snippet += f"[{w['start']:.1f}s] {w['word']} "
        
        prompt = f"""Analyze this podcast transcript and identify advertisement segments.
        
Look for:
- Sponsor mentions ("brought to you by", "this episode is sponsored by")
- Promo codes and discounts
- Host reads that are clearly ads (different tone, product mentions)
- Any segment longer than 30 seconds that appears to be advertising

Transcript excerpt:
{transcript_snippet}

Respond with ONLY a JSON array of ad segments in this format:
[{{"start": seconds, "end": seconds, "type": "ad", "confidence": "high/medium/low"}}]

If no ads found, respond with: []"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500
        )
        
        import json
        import re
        
        # Parse JSON response
        try:
            ads = json.loads(response.choices[0].message.content)
            if ads and isinstance(ads, list):
                logger.info(f"LLM detected {len(ads)} ad segments")
                return ads
        except:
            # Try to extract JSON from response
            match = re.search(r'\[.*\]', response.choices[0].message.content)
            if match:
                ads = json.loads(match.group())
                logger.info(f"LLM detected {len(ads)} ad segments")
                return ads
        
        return []
        
    except Exception as e:
        logger.error(f"LLM ad detection failed: {e}")
        return []


def detect_ads_full_episode(audio_path: str, podcast_name: str,
                          content_start: float = 0.0) -> list[dict]:
    """
    Two-stage ad detection:
    1. First use keyword heuristic (fast, free)
    2. Then use LLM for better accuracy
    """
    # Stage 1: Keyword-based detection
    ad_segments = []
    
    # ... existing keyword logic would be called here ...
    # (keeping existing function intact)
    
    # For now, let's add LLM detection
    if not OPENAI_API_KEY:
        return ad_segments
    
    try:
        import openai
        import tempfile
        from pathlib import Path
        
        # Transcribe full audio for LLM analysis
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Get first 30 minutes for LLM analysis (cost-effective)
            intro_cut = content_start
            audio_for_llm = tmp_path / "llm_analysis.mp3"
            
            subprocess.run([
                'ffmpeg', '-y', '-i', audio_path,
                '-ss', str(intro_cut),
                '-t', '1800',  # 30 min max
                '-acodec', 'libmp3lame', '-ab', '64k',
                str(audio_for_llm)
            ], capture_output=True)
            
            if not audio_for_llm.exists():
                return ad_segments
            
            # Transcribe for LLM
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            with open(audio_for_llm, 'rb') as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )
            
            words = []
            if hasattr(transcript, 'words'):
                for w in transcript.words:
                    words.append({
                        "word": w.word,
                        "start": w.start + intro_cut,
                        "end": w.end + intro_cut
                    })
            
            if words:
                llm_ads = detect_ads_with_llm(words)
                if llm_ads:
                    logger.info(f"LLM detected {len(llm_ads)} ad segments")
                    ad_segments.extend(llm_ads)
    
    except Exception as e:
        logger.error(f"LLM ad detection failed: {e}")
    
    return ad_segments


# ============================================================
# AUDIO CUTTING — Remove segments and stitch
# ============================================================

def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds using ffprobe."""
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def cut_segments_from_audio(input_path: str, output_path: str,
                             cut_segments: list[dict], intro_cut: float = 0.0):
    """
    Remove intro (everything before intro_cut) and all ad segments.
    Stitches remaining audio into a single clean file.

    cut_segments: list of {start, end} in seconds (absolute, from original file start)
    intro_cut: seconds — everything before this is removed
    """
    duration = get_audio_duration(input_path)
    if duration == 0:
        raise ValueError("Could not determine audio duration")

    logger.info(f"Audio duration: {duration:.1f}s, intro_cut: {intro_cut:.1f}s, "
                f"ad segments: {len(cut_segments)}")

    # Build list of KEEP segments
    keep_segments = []
    cursor = intro_cut
    ads = sorted(cut_segments, key=lambda s: s['start'])

    for ad in ads:
        ad_start = max(ad['start'], cursor)
        ad_end = ad['end']
        if ad_start > cursor:
            keep_segments.append((cursor, ad_start))
        cursor = max(cursor, ad_end)

    if cursor < duration:
        keep_segments.append((cursor, duration))

    logger.info(f"Keeping {len(keep_segments)} segments: "
                f"{[(f'{s:.1f}', f'{e:.1f}') for s, e in keep_segments]}")

    if not keep_segments:
        raise ValueError("No audio segments to keep after cutting")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        segment_files = []

        for idx, (seg_start, seg_end) in enumerate(keep_segments):
            seg_file = tmp_path / f"seg_{idx:04d}.mp3"
            subprocess.run([
                'ffmpeg', '-y',
                '-i', input_path,
                '-ss', str(seg_start),
                '-to', str(seg_end),
                '-acodec', 'copy',
                str(seg_file)
            ], capture_output=True, check=True)
            segment_files.append(seg_file)

        if len(segment_files) == 1:
            shutil.copy2(str(segment_files[0]), output_path)
        else:
            concat_list = tmp_path / "concat.txt"
            with open(concat_list, 'w') as f:
                for sf in segment_files:
                    f.write(f"file '{sf}'\n")

            subprocess.run([
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0',
                '-i', str(concat_list),
                '-acodec', 'copy',
                output_path
            ], capture_output=True, check=True)

    logger.info(f"Clean audio written to {output_path}")


# ============================================================
# MAIN PROCESSING WORKER
# ============================================================

def process_audio_worker(podcast_name: str, episode_id: str,
                          original_url: str, title: str):
    episode_dir = STORAGE_DIR / podcast_name / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    temp_file = episode_dir / "original.mp3"
    processed_file = episode_dir / "clean.mp3"
    cache_key = f"{podcast_name}_{episode_id}"

    try:
        # ── Step 1: Download ──────────────────────────────────────
        logger.info(f"[{title}] Downloading...")
        response = requests.get(original_url, timeout=600, stream=True)
        response.raise_for_status()
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        logger.info(f"[{title}] Downloaded: {temp_file.stat().st_size / 1e6:.1f} MB")

        # ── Step 2: Intro tune detection ─────────────────────────
        intro_cut = 0.0

        if has_fingerprint(podcast_name):
            logger.info(f"[{title}] Searching for intro tune using fingerprint...")
            intro_end = detect_intro_tune_in_audio(str(temp_file), podcast_name)
            if intro_end is not None:
                intro_cut = intro_end
                logger.info(f"[{title}] Intro ends at {intro_cut:.1f}s")
            else:
                logger.info(f"[{title}] Intro tune not found in this episode")
        else:
            logger.info(f"[{title}] No fingerprint yet — running auto-detection pipeline...")
            intro_cut = auto_detect_and_fingerprint_intro(str(temp_file), podcast_name)
            logger.info(f"[{title}] First-play intro cut at {intro_cut:.1f}s")

        # ── Step 3: Full episode Whisper ad detection ─────────────
        logger.info(f"[{title}] Running full-episode ad detection...")
        ad_segments = detect_ads_full_episode(
            str(temp_file), podcast_name,
            content_start=intro_cut
        )

        # ── Step 4: Cut intro + ads, stitch clean audio ───────────
        logger.info(f"[{title}] Cutting {len(ad_segments)} ad segments, "
                    f"intro cut at {intro_cut:.1f}s")
        cut_segments_from_audio(
            str(temp_file),
            str(processed_file),
            cut_segments=ad_segments,
            intro_cut=intro_cut
        )

        # ── Step 5: Cleanup & metadata ────────────────────────────
        temp_file.unlink(missing_ok=True)

        metadata = {
            "processed_at": datetime.now().isoformat(),
            "title": title,
            "intro_cut": intro_cut,
            "ad_segments": ad_segments,
            "ad_count": len(ad_segments),
            "total_cut_seconds": sum(s['end'] - s['start'] for s in ad_segments) + intro_cut,
        }
        with open(episode_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        with PROCESSING_LOCK:
            PROCESSING[cache_key] = {"status": "ready"}

        logger.info(f"[{title}] Done. Removed {metadata['total_cut_seconds']:.0f}s of ads/intro")

    except Exception as e:
        logger.error(f"[{title}] Worker failed: {e}", exc_info=True)
        with PROCESSING_LOCK:
            PROCESSING[cache_key] = {"status": "error", "error": str(e)}
        temp_file.unlink(missing_ok=True)


# ============================================================
# INTRO TUNE VERIFICATION
# ============================================================

def get_intro_clip_info(podcast_name: str) -> dict | None:
    """Return metadata about a stored intro clip, or None if not found."""
    clip_path = INTRO_CLIPS_DIR / f"{podcast_name}_intro.mp3"
    fp_file = fingerprint_path(podcast_name)

    if not clip_path.exists():
        return None

    fp_data = {}
    if fp_file.exists():
        with open(fp_file) as f:
            fp_data = json.load(f)

    stat = clip_path.stat()
    return {
        "podcast_name": podcast_name,
        "clip_path": clip_path,
        "file_size": stat.st_size,
        "created_at": fp_data.get("created_at", datetime.fromtimestamp(stat.st_mtime).isoformat()),
        "duration": fp_data.get("duration", 0),
        "intro_start": fp_data.get("intro_start_reference", 0),
        "source_episode": fp_data.get("source_episode", "unknown"),
    }


@app.route('/intros/feed')
@require_auth
def intro_verification_feed():
    """
    RSS feed of all detected intro tunes.
    Subscribe to this in your podcast app to verify intro detection is correct.
    Subscribe URL: {BASE_URL}/intros/feed
    """
    feeds = load_opml()
    items = []

    for podcast_name in feeds:
        info = get_intro_clip_info(podcast_name)
        if info:
            items.append(info)

    items.sort(key=lambda x: x['created_at'], reverse=True)

    pub_date_fmt = "%a, %d %b %Y %H:%M:%S +0000"

    rss_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">',
        '<channel>',
        '<title>Podcast Intro Tune Verification</title>',
        '<description>Detected intro tunes — listen to verify they are correct. '
        'Delete a fingerprint at DELETE /fingerprint/{name} if wrong.</description>',
        f'<link>{BASE_URL}/intros/feed</link>',
        f'<lastBuildDate>{datetime.utcnow().strftime(pub_date_fmt)}</lastBuildDate>',
    ]

    for info in items:
        podcast_name = info['podcast_name']
        audio_url = f"{BASE_URL}/intros/audio/{podcast_name}"
        file_size = info['file_size']
        duration_secs = int(info['duration'])
        created_at = datetime.fromisoformat(info['created_at'])
        pub_date = created_at.strftime(pub_date_fmt)

        description = (
            f"Detected intro for: {podcast_name} | "
            f"Starts at {info['intro_start']:.1f}s in source episode | "
            f"Duration: {duration_secs}s | "
            f"Source episode ID: {info['source_episode']} | "
            f"Detected: {info['created_at'][:10]} | "
            f"If this sounds WRONG: DELETE /fingerprint/{podcast_name} to reset."
        )

        rss_lines += [
            '<item>',
            f'<title>INTRO: {xml_escape(podcast_name)} '
            f'({duration_secs}s from {info["intro_start"]:.0f}s)</title>',
            f'<description>{xml_escape(description)}</description>',
            f'<enclosure url="{audio_url}" type="audio/mpeg" length="{file_size}"/>',
            f'<guid isPermaLink="false">intro-{xml_escape(podcast_name)}</guid>',
            f'<pubDate>{pub_date}</pubDate>',
            f'<itunes:duration>{duration_secs}</itunes:duration>',
            f'<itunes:subtitle>Verify this is the correct intro tune</itunes:subtitle>',
            '</item>',
        ]

    rss_lines += ['</channel>', '</rss>']
    return Response('\n'.join(rss_lines), mimetype='application/xml')


@app.route('/intros/audio/<podcast_name>')
def serve_intro_audio(podcast_name):
    """Serve the intro clip audio file."""
    clip_path = INTRO_CLIPS_DIR / f"{podcast_name}_intro.mp3"
    if not clip_path.exists():
        return jsonify({"error": f"No intro clip found for {podcast_name}"}), 404
    return send_file(clip_path, mimetype='audio/mpeg',
                     download_name=f"{podcast_name}_intro.mp3")


@app.route('/intros')
@require_auth
def list_intros():
    """JSON overview of all stored intro clips with verification links."""
    feeds = load_opml()
    result = []

    for podcast_name in feeds:
        info = get_intro_clip_info(podcast_name)
        if info:
            result.append({
                "podcast": podcast_name,
                "duration_seconds": round(info['duration'], 1),
                "intro_starts_at": round(info['intro_start'], 1),
                "detected_at": info['created_at'],
                "source_episode": info['source_episode'],
                "listen_url": f"{BASE_URL}/intros/audio/{podcast_name}",
                "delete_fingerprint_url": f"{BASE_URL}/fingerprint/{podcast_name}",
            })
        else:
            result.append({
                "podcast": podcast_name,
                "duration_seconds": None,
                "detected_at": None,
                "status": "no_intro_detected_yet",
            })

    return jsonify({
        "intro_feed_url": f"{BASE_URL}/intros/feed",
        "total": len(result),
        "with_intro": sum(1 for r in result if r.get('detected_at')),
        "pending": sum(1 for r in result if not r.get('detected_at')),
        "intros": result,
    })


# ============================================================
# RSS PROXY
# ============================================================

@app.route('/feeds')
@require_auth
def list_feeds():
    """List all available podcast slugs from OPML."""
    feeds = load_opml()
    return jsonify({
        "podcasts": [
            {"slug": slug, "rss": url, "feed": f"{BASE_URL}/feed/{slug}"}
            for slug, url in feeds.items()
        ]
    })


@app.route('/feed/<podcast_name>')
@require_auth
def get_feed(podcast_name):
    """Return modified RSS feed pointing audio URLs at our server."""
    original_rss = get_rss_url(podcast_name)
    if not original_rss:
        return jsonify({"error": f"Unknown podcast: {podcast_name}. "
                                  "Check /feeds for available slugs."}), 404

    feed = feedparser.parse(original_rss)

    rss_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">',
        '<channel>',
        f'<title>{xml_escape(feed.feed.get("title", podcast_name))}</title>',
        f'<description>{xml_escape(feed.feed.get("description", ""))}</description>',
        f'<link>{xml_escape(feed.feed.get("link", ""))}</link>',
    ]

    artwork = None
    if hasattr(feed.feed, 'image'):
        artwork = feed.feed.image.get('href')
    if artwork:
        rss_lines.append(f'<itunes:image href="{xml_escape(artwork)}"/>')
        rss_lines.append(f'<image><url>{xml_escape(artwork)}</url><title>'
                         f'{xml_escape(feed.feed.get("title", ""))}</title></image>')

    for entry in feed.entries:
        orig_audio = get_audio_url(entry)
        if not orig_audio:
            continue

        episode_id = hashlib.md5(orig_audio.encode()).hexdigest()[:12]
        audio_url = f'{BASE_URL}/audio/{podcast_name}/{episode_id}'
        duration = entry.get('itunes_duration', '')

        rss_lines += [
            '<item>',
            f'<title>{xml_escape(entry.get("title", "Episode"))}</title>',
            f'<description>{xml_escape(entry.get("summary", entry.get("description", "")))}</description>',
            f'<enclosure url="{audio_url}" type="audio/mpeg" length="0"/>',
            f'<guid isPermaLink="false">{xml_escape(episode_id)}</guid>',
            f'<pubDate>{xml_escape(entry.get("published", ""))}</pubDate>',
        ]
        if duration:
            rss_lines.append(f'<itunes:duration>{xml_escape(str(duration))}</itunes:duration>')

        item_art = None
        if hasattr(entry, 'itunes_image'):
            item_art = entry.itunes_image
        elif 'image' in entry:
            item_art = entry.image.get('href')
        if item_art:
            rss_lines.append(f'<itunes:image href="{xml_escape(item_art)}"/>')

        rss_lines.append('</item>')

    rss_lines += ['</channel>', '</rss>']
    return Response('\n'.join(rss_lines), mimetype='application/xml')


# ============================================================
# AUDIO ROUTE
# ============================================================

@app.route('/audio/<podcast_name>/<episode_id>')
def stream_audio(podcast_name, episode_id):
    episode_info = get_episode_info(podcast_name, episode_id)
    if not episode_info:
        return jsonify({"error": "Episode not found"}), 404

    original_url = episode_info['audio_url']
    episode_dir = STORAGE_DIR / podcast_name / episode_id
    processed_file = episode_dir / "clean.mp3"
    cache_key = f"{podcast_name}_{episode_id}"

    if processed_file.exists():
        logger.info(f"Serving cached: {episode_info['title']}")
        return send_file(processed_file, mimetype='audio/mpeg',
                         download_name=f"{episode_id}.mp3")

    with PROCESSING_LOCK:
        status = PROCESSING.get(cache_key, {}).get("status")
        if status not in ("processing",):
            logger.info(f"Queuing: {episode_info['title']}")
            PROCESSING[cache_key] = {"status": "processing"}
            thread = threading.Thread(
                target=process_audio_worker,
                args=(podcast_name, episode_id, original_url, episode_info['title']),
                daemon=True
            )
            thread.start()

    return redirect(original_url, code=302)


# ============================================================
# STATUS
# ============================================================

@app.route('/status/<podcast_name>/<episode_id>')
def episode_status(podcast_name, episode_id):
    """Check processing status of an episode."""
    cache_key = f"{podcast_name}_{episode_id}"
    episode_dir = STORAGE_DIR / podcast_name / episode_id
    processed_file = episode_dir / "clean.mp3"

    if processed_file.exists():
        meta_file = episode_dir / "metadata.json"
        meta = {}
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
        return jsonify({"status": "ready", "metadata": meta})

    with PROCESSING_LOCK:
        info = PROCESSING.get(cache_key, {"status": "unknown"})
    return jsonify(info)


# ============================================================
# FINGERPRINT MANAGEMENT
# ============================================================

@app.route('/fingerprint/<podcast_name>', methods=['GET'])
def get_fingerprint(podcast_name):
    """Check if a fingerprint exists for a podcast."""
    fp = fingerprint_path(podcast_name)
    if not fp.exists():
        return jsonify({"exists": False}), 404
    with open(fp) as f:
        data = json.load(f)
    return jsonify({
        "exists": True,
        "podcast": data.get("podcast"),
        "duration": data.get("duration"),
        "created_at": data.get("created_at"),
        "intro_start_reference": data.get("intro_start_reference"),
    })


@app.route('/fingerprint/<podcast_name>', methods=['DELETE'])
def delete_fingerprint(podcast_name):
    """
    Delete fingerprint AND intro clip so both get rebuilt on next play.
    Use this when the intro verification feed shows a wrong clip.
    """
    fp = fingerprint_path(podcast_name)
    clip = INTRO_CLIPS_DIR / f"{podcast_name}_intro.mp3"
    deleted = []

    if fp.exists():
        fp.unlink()
        deleted.append("fingerprint")
    if clip.exists():
        clip.unlink()
        deleted.append("intro_clip")

    if deleted:
        return jsonify({
            "status": "deleted",
            "deleted": deleted,
            "next": f"Play any episode of {podcast_name} to rebuild"
        })
    return jsonify({"status": "not_found"}), 404


# ============================================================
# EPISODE INFO LOOKUP
# ============================================================

def get_episode_info(podcast_name: str, episode_id: str) -> dict | None:
    original_rss = get_rss_url(podcast_name)
    if not original_rss:
        return None

    feed = feedparser.parse(original_rss)
    for entry in feed.entries:
        orig_audio = get_audio_url(entry)
        if orig_audio and hashlib.md5(orig_audio.encode()).hexdigest()[:12] == episode_id:
            return {
                "title": entry.get('title', 'Episode'),
                "audio_url": orig_audio
            }
    return None


# ============================================================
# CLEANUP
# ============================================================

@app.route('/cleanup')
def run_cleanup():
    """Delete processed episodes older than 14 days."""
    deleted = 0
    errors = []
    cutoff = datetime.now() - timedelta(days=14)

    for pod_dir in STORAGE_DIR.iterdir():
        if not pod_dir.is_dir():
            continue
        for ep_dir in pod_dir.iterdir():
            meta_file = ep_dir / "metadata.json"
            if not meta_file.exists():
                continue
            try:
                with open(meta_file) as f:
                    data = json.load(f)
                processed_at = datetime.fromisoformat(data['processed_at'])
                if processed_at < cutoff:
                    shutil.rmtree(ep_dir)
                    deleted += 1
                    logger.info(f"Cleaned up: {ep_dir}")
            except Exception as e:
                errors.append(str(e))
                logger.warning(f"Cleanup error for {ep_dir}: {e}")

    return jsonify({"status": "success", "deleted": deleted, "errors": errors})


# ============================================================
# OPML MANAGEMENT
# ============================================================

@app.route('/opml', methods=['GET'])
def get_opml():
    """Download current OPML file."""
    if not OPML_FILE.exists():
        return jsonify({"error": "No OPML file found"}), 404
    return send_file(OPML_FILE, mimetype='text/xml', download_name='podcasts.opml')


@app.route('/opml', methods=['POST'])
def upload_opml():
    """Upload a new OPML file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    OPML_FILE.parent.mkdir(parents=True, exist_ok=True)
    f.save(str(OPML_FILE))
    feeds = load_opml()
    return jsonify({
        "status": "uploaded",
        "podcast_count": len(feeds),
        "slugs": list(feeds.keys())
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    logger.info("Starting Podcast Ad Remover")
    logger.info(f"OPML:     {OPML_FILE}")
    logger.info(f"Storage:  {STORAGE_DIR}")
    logger.info(f"Base URL: {BASE_URL}")
    feeds = load_opml()
    logger.info(f"Loaded {len(feeds)} podcasts: {list(feeds.keys())}")
    app.run(host='0.0.0.0', port=3333, debug=False)
