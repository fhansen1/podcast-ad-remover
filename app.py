"""
Podcast Ad Remover - On-Demand Processing
Flow: RSS Proxy → Process on Play → Stream Clean Audio → Auto-Delete after 14 days

Ad detection via Whisper API - automatically finds ad segments!
"""
import os
import re
import time
import uuid
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
import feedparser
from flask import Flask, Response, request, redirect, jsonify, send_file
import html

# Whisper API for ad detection
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Helper to escape XML special characters
def xml_escape(text):
    if not text:
        return ""
    return (str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
        .replace("\xa0", " "))  # Non-breaking space


# ============================================================
# WHISPER API AD DETECTION
# ============================================================

def detect_ads_with_whisper(audio_path, podcast_name="default"):
    """
    Use Whisper API to detect ad segments.
    If file > 25MB, compress it first for Whisper.
    Returns timestamps for cutting the HIGH QUALITY original.
    """
    if not OPENAI_API_KEY:
        logger.warning("No OpenAI API key configured - using default timestamps")
        return None
    
    try:
        import openai
        import tempfile
        import subprocess
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Check file size
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Audio file size: {file_size_mb:.1f} MB")
        
        # Create temp file for Whisper
        temp_clip = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        temp_clip.close()
        
        if file_size_mb > 23:  # Leave margin under 25MB limit
            # Compress: reduce to 64kbps to fit in 25MB
            logger.info("File too large, compressing for Whisper...")
            subprocess.run([
                'ffmpeg', '-y', '-i', audio_path,
                '-ss', '0', '-t', '120',  # First 2 min is enough for ad detection
                '-acodec', 'libmp3lame', '-ab', '64k',
                temp_clip.name
            ], capture_output=True)
        else:
            # Small enough, just extract first 2 minutes
            subprocess.run([
                'ffmpeg', '-y', '-i', audio_path,
                '-ss', '0', '-t', '120',
                '-acodec', 'copy', temp_clip.name
            ], capture_output=True)
        
        # Send to Whisper with custom prompts per podcast
        prompts = {
            "det_store_bilded": "Norwegian political podcast. Identify when the MAIN CONTENT starts - this is when the hosts Øystein Hansen or Eirik Bergesen begin discussing the topic. Ignore intro music, sponsor ads, and jingles. Return the timestamp when the actual podcast discussion begins.",
            "pop_og_politikk": "Norwegian podcast about politics and pop culture. Identify when the MAIN CONTENT starts - this is when the hosts Asbjørn or Marte begin discussing the topic. Ignore intro music, sponsor ads, and jingles. Return the timestamp when the actual podcast discussion begins."
        }
        
        prompt = prompts.get(podcast_name, "Identify when the main podcast content starts. Ignore intro music and ads. Return the timestamp when the hosts begin speaking.")
        
        with open(temp_clip.name, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                prompt=prompt,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
        
        # Clean up temp file
        os.unlink(temp_clip.name)
        
        # Get words with timestamps
        words = transcript.words if hasattr(transcript, 'words') else []
        
        if not words:
            return None
        
        # Detect ad segments - based on Whisper detecting content start
        ad_segments = []
        first_words = [w for w in words if w.start < 90]
        
        if first_words:
            # The prompt asks Whisper to identify when content starts
            # So the first word should be the start of actual content
            content_start = first_words[0].start
            
            # If content starts after 10 seconds, there's likely an ad/jingle before
            if content_start > 10:
                ad_segments.append({
                    "start": 0,
                    "end": content_start,
                    "type": "detected"
                })
                logger.info(f"Detected ad/jingle: first content at {content_start}s")
        
        if ad_segments:
            logger.info(f"Detected ad segments via Whisper: {ad_segments}")
        
        return ad_segments if ad_segments else None
        
    except Exception as e:
        logger.error(f"Whisper detection failed: {e}")
        return None


# ============================================================
# FINGERPRINT DETECTION - Find intro using stored fingerprint
# ============================================================

def detect_intro_with_fingerprint(audio_path, podcast_name):
    """Use fingerprint to find where intro ends"""
    fingerprint_file = Path(__file__).parent / f"intro_fingerprint_{podcast_name}.json"
    
    if not fingerprint_file.exists():
        logger.info(f"No fingerprint found for {podcast_name}")
        return None
    
    try:
        import numpy as np
        import librosa
        
        with open(fingerprint_file) as f:
            fp_data = json.load(f)
        
        ref_mfcc = np.array(fp_data['fingerprint']['mfcc'])
        intro_duration = fp_data['fingerprint']['duration']
        
        # Expected intro ranges per podcast
        intro_ranges = {
            "det_store_bilded": {"min": 45, "max": 80},  # Intro typically 50-75s
            "pop_og_politikk": {"min": 0, "max": 30}
        }
        expected_range = intro_ranges.get(podcast_name, {"min": 0, "max": 90})
        
        # Extract MFCC from episode (first 90 seconds to cover intro range)
        y, sr = librosa.load(audio_path, sr=44100, mono=True, duration=90)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Sliding window - check each second
        best_match = 0
        best_position = 0
        
        window_size = int(intro_duration * sr)
        hop_length = sr  # 1 second hop
        
        for pos in range(int(expected_range["min"] * sr), min(int(expected_range["max"] * sr), len(y) - window_size), hop_length):
            window = y[pos:pos + window_size]
            window_mfcc = np.mean(librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13), axis=1)
            
            similarity = np.dot(ref_mfcc, window_mfcc) / (np.linalg.norm(ref_mfcc) * np.linalg.norm(window_mfcc))
            
            if similarity > best_match:
                best_match = similarity
                best_position = pos / sr
        
        logger.info(f"Fingerprint match: {best_match:.1%} at {best_position:.1f}s")
        
        # Return intro end position
        if best_match >= 0.60:
            return {"intro_end": best_position, "confidence": best_match}
        
        return None
        
    except Exception as e:
        logger.error(f"Fingerprint detection failed: {e}")
        return None


from werkzeug.routing import Map, Rule
import ffmpeg

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config - Set your external URL here
BASE_URL = os.environ.get("BASE_URL", "https://podcast.drhahn.no/")

# Optional: Basic auth (set via environment)
RSS_USERNAME = os.environ.get("RSS_USERNAME", "")
RSS_PASSWORD = os.environ.get("RSS_PASSWORD", "")

app = Flask(__name__)
STORAGE_DIR = Path("/data/podcasts")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Known ad timestamps (podcast_name -> {start_cut: seconds, end_cut: seconds})

# Track processing status
PROCESSING = {}  # episode_id -> {"status": "processing"|"ready"|"error", "progress": 0-100}

# ============================================================
# AUTH DECORATOR
# ============================================================

def require_auth(f):
    """Require basic auth if configured"""
    def decorated(*args, **kwargs):
        if RSS_USERNAME and RSS_PASSWORD:
            from flask import request
            auth = request.authorization
            if not auth or auth.username != RSS_USERNAME or auth.password != RSS_PASSWORD:
                return Response('Authentication required', 401, {'WWW-Authenticate': 'Basic realm="Podcast RSS"'})
        return f(*args, **kwargs)
    decorated.__name__ = f.__name__
    return decorated

# ============================================================
# RSS PROXY - Modifies original RSS to point to our server
# ============================================================

@app.route('/feed/<podcast_name>')
#@require_auth
def get_feed(podcast_name):
    """Get modified RSS with our audio URLs"""
    # Map podcast name to original RSS
    rss_map = {
    "det_store_bilded": "https://rss.podplaystudio.com/692.xml",
    "pop_og_politikk": "https://rss.podplaystudio.com/4039.xml",
    "frelst": "https://www.omnycontent.com/d/playlist/b879ead7-5fbb-4c64-bede-b13801013ca4/5ffb3cc5-5bac-4124-9a56-b164009e28a9/7c34fe34-5d23-4ccb-ae13-b164009e28c5/podcast.rss",
    "trygdekontoret": "https://podkast.nrk.no/program/trygdekontoret.rss",
    "dwarkesh": "https://apple.dwarkesh-podcast.workers.dev/feed.rss",
    "den_politiske_situasjonen": "https://feeds.acast.com/public/shows/f6006cc3-2278-4017-b9ca-35d6e229e7fd",
    "fotballfortellinger": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/2b102ed8-af4a-4d51-ad41-b27b0106e973/46b58f60-9633-43f6-aa67-b27b0106e97a/podcast.rss",
    "the_rest_is_politics_leading": "https://feeds.megaphone.fm/GLT9029505120",
    "otto_og_anne": "https://rss.podplaystudio.com/745.xml",
    "heia_fotball": "https://podkast.nrk.no/program/heia_fotball.rss",
    "tid_er_penger": "https://feeds.acast.com/public/shows/659c418069d2da0016ac759b",
    "empire_world_history": "https://feeds.megaphone.fm/empirepodcast",
    "fladseth": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/afaede69-c88e-4075-aa19-b15400c5650b/7960a0b2-0102-4b8c-a665-b15400c56519/podcast.rss",
    "historier_som_endret_verden": "https://feeds.acast.com/public/shows/63889d03eaf58500117bbbca",
    "rett_og_slett": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/abdff6d2-d127-4cdf-8754-b15e00efcf82/c7367e4f-9138-4cf7-8c55-b15e00efcf95/podcast.rss",
    "alt_fortalt": "https://podcast.stream.schibsted.media/vgtv/100447?podcast",
    "peter_zeihan": "https://media.rss.com/zeihan/feed.xml",
    "pedagogisk_idrettspreik": "https://rss.libsyn.com/shows/112281/destinations/626995.xml",
    "psykodrama": "https://feeds.acast.com/public/shows/401782a2-4af0-4760-8505-6a2075d6887a",
    "chivadze": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/f0177c62-0df5-47e1-afbc-b15e00efd15b/0f467901-7814-49b9-8700-b15e00efd169/podcast.rss",
    "dritte_halbzeit": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/f4d4250b-5585-46f1-8f08-b149016d7f2b/d46c4794-37fc-460b-bbb5-b149016d7f55/podcast.rss",
    "when_we_were_kings": "https://feed.pod.space/whenwewerekings",
    "skisporet": "https://skisporet.libsyn.com/rss",
    "misjonen": "https://smartpod.no/feed/misjonen",
    "real_dictators": "https://feeds.megaphone.fm/NSR6187882468",
    "aftenpodden_usa": "https://podcast.stream.schibsted.media/ap/100196?podcast",
    "krigsrevyen": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/3a78fd64-b2e2-4c2c-a854-b27c00ed7886/b086f1a2-fbd3-44cf-9e05-b27c00ed789b/podcast.rss",
    "foreldrepodden": "https://rss.buzzsprout.com/1909514.rss",
    "teknisk_sett": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/5d96f449-e2b9-434a-8880-b14c007a47b3/6a1f66ca-8da3-4bf7-9935-b14c007a47ca/podcast.rss",
    "the_rest_is_classified": "https://feeds.megaphone.fm/therestisclassified",
    "lawfare": "https://feeds.acast.com/public/shows/60518a52f69aa815d2dba41c",
    "the_rest_is_history": "https://feeds.megaphone.fm/GLT4787413333",
    "historier_som_endret_norge": "https://feeds.acast.com/public/shows/63889ebd60bbdd001068859c",
    "krimpodden": "https://podcast.stream.schibsted.media/vgtv/100417?podcast",
    "fotballklubben": "https://anchor.fm/s/10854bc18/podcast/rss",
    "rekommandert": "https://feeds.acast.com/public/shows/f2b8c8dd-60de-4e34-b49d-a689f67ff31f",
    "hacks_on_tap": "https://feeds.megaphone.fm/VMP7545057845",
    "stormkast": "https://rss.podplaystudio.com/602.xml",
    "breaking_marathon_limits": "https://rss.buzzsprout.com/299975.rss",
    "political_gabfest": "https://my.slate.com/podcasts/feeds/political-gabfest/",
    "the_rest_is_politics_us": "https://feeds.megaphone.fm/GLT5336643697",
    "laligaloca": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/ea6e7251-b0a6-4148-8690-b15400c4d0b9/81f76f17-14b9-4845-912f-b15400c4d0c7/podcast.rss",
    "avhort": "https://feeds.acast.com/public/shows/685bb5014a2d8b8b51e33360",
    "altinget_partianalysen": "https://feeds.acast.com/public/shows/altinget-partianalysen",
    "olympiapodden": "https://feeds.soundcloud.com/users/soundcloud:users:420086565/sounds.rss",
    "war_on_the_rocks": "https://rss.libsyn.com/shows/70702/destinations/298196.xml",
    "the_drop": "https://anchor.fm/s/347e3928/podcast/rss",
    "oppdatert": "https://podkast.nrk.no/program/oppdatert.rss",
    "valgkampsirkuset": "https://feeds.acast.com/public/shows/655761451a7d7e0012c36c95",
    "hans_holund": "https://feeds.acast.com/public/shows/653f7814bfecf80012750eb9",
    "short_history_of": "https://feeds.megaphone.fm/NSR8236478683",
    "the_rest_is_politics": "https://feeds.megaphone.fm/GLT9190936013",
    "foreldreradet": "https://feeds.acast.com/public/shows/de647d54-d4a9-4ed5-b404-d5e0e1679630",
    "lopeskopodden": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/3446a04c-4e9e-484d-864d-b27c00ed7950/25a6340a-216f-4e44-849a-b27c00ed795b/podcast.rss",
    "loperadet": "https://feeds.acast.com/public/shows/67c84bbd48f26a4bcaecfa09",
    "everything_everywhere_daily": "https://feeds.megaphone.fm/ADV3162807280",
    "dan_snow_history_hit": "https://access.acast.com/rss/c939f8d1-c4bc-478e-8bb9-e5343f9a7ab5",
    "ukraine_the_latest": "https://feeds.acast.com/public/shows/6583012e715d5300169dcdcb",
}
    
    original_rss = rss_map.get(podcast_name)
    if not original_rss:
        return jsonify({"error": "Unknown podcast"}), 404
    
    # Parse original feed
    feed = feedparser.parse(original_rss)
    
    # Build modified RSS
    rss_lines = []
    rss_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    rss_lines.append('<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">')
    rss_lines.append('<channel>')
    rss_lines.append(f'<title>{xml_escape(feed.feed.get("title", podcast_name))}</title>')
    rss_lines.append(f'<description>{xml_escape(feed.feed.get("description", ""))}</description>')
    
    # Add artwork - try to get from feed
    artwork = None
    if hasattr(feed.feed, 'image'):
        artwork = feed.feed.image.get('href')
    elif hasattr(feed.feed, 'itunes') and hasattr(feed.feed.itunes, 'image'):
        artwork = feed.feed.itunes.image
    if artwork:
        rss_lines.append(f'<itunes:image href="{xml_escape(artwork)}"/>')
        rss_lines.append(f'<image><url>{xml_escape(artwork)}</url></image>')
    
    for idx, entry in enumerate(feed.entries):
        # Use sequential ID - stable across feed updates
        episode_id = idx + 1
        
        rss_lines.append('<item>')
        rss_lines.append(f'<title>{xml_escape(entry.get("title", "Episode"))}</title>')
        rss_lines.append(f'<description>{xml_escape(entry.get("description", ""))}</description>')
        
        # Our custom audio URL - points to our processor
        audio_url = f'{BASE_URL}audio/{podcast_name}/{episode_id}'
        
        # Get duration if available
        duration = entry.get('itunes_duration', 0)
        
        rss_lines.append(f'<enclosure url="{audio_url}" type="audio/mpeg" length="0"/>')
        rss_lines.append(f'<guid>{audio_url}</guid>')
        rss_lines.append(f'<pubDate>{entry.get("published", "")}</pubDate>')
        if duration:
            rss_lines.append(f'<itunes:duration>{duration}</itunes:duration>')
        
        # Add item artwork if available
        if hasattr(entry, 'itunes') and hasattr(entry.itunes, 'image'):
            rss_lines.append(f'<itunes:image href="{xml_escape(entry.itunes.image)}"/>')
        
        rss_lines.append('</item>')
    
    rss_lines.append('</channel>')
    rss_lines.append('</rss>')
    
    return Response('\n'.join(rss_lines), mimetype='application/xml')

# ============================================================
# AUDIO PROCESSOR - Download, cut ads, serve
# ============================================================

@app.route('/audio/<podcast_name>/<episode_id>')
def stream_audio(podcast_name, episode_id):
    """Process and stream clean audio"""
    
    # Get episode info from RSS
    episode_info = get_episode_info(podcast_name, episode_id)
    if not episode_info:
        return jsonify({"error": "Episode not found"}), 404
    
    original_url = episode_info['audio_url']
    title = episode_info['title']
    
    # Create unique file path
    episode_dir = STORAGE_DIR / podcast_name / str(episode_id)
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    processed_file = episode_dir / "clean.mp3"
    
    # Check if already processed
    if processed_file.exists():
        logger.info(f"Serving cached: {title}")
        return send_file(processed_file, mimetype='audio/mpeg')
    
    # Check if currently processing
    cache_key = f"{podcast_name}_{episode_id}"
    if cache_key in PROCESSING and PROCESSING[cache_key].get("status") == "processing":
        return jsonify({"error": "Still processing, try again in a moment"}), 202
    
    # Start processing
    PROCESSING[cache_key] = {"status": "processing", "title": title}
    
    try:
        # Download original
        logger.info(f"Downloading: {title}")
        temp_file = episode_dir / "original.mp3"
        
        response = requests.get(original_url, timeout=300, stream=True)
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # STEP 1: Use fingerprint to find intro end
        intro_info = detect_intro_with_fingerprint(str(temp_file), podcast_name)
        
        intro_cut = 0
        if intro_info:
            intro_cut = intro_info.get("intro_end", 0)
            logger.info(f"Fingerprint: intro ends at {intro_cut:.1f}s (confidence: {intro_info.get('confidence', 0):.1%})")
        
        # STEP 2: Use Whisper to detect ads in the remaining audio
        # Extract audio after intro cut for Whisper analysis
        if intro_cut > 0:
            temp_after_intro = episode_dir / "after_intro.mp3"
            ffmpeg.input(str(temp_file), ss=intro_cut).output(
                str(temp_after_intro),
                acodec='copy'
            ).run(overwrite_output=True, quiet=True)
            whisper_segments = detect_ads_with_whisper(str(temp_after_intro), podcast_name)
            temp_after_intro.unlink()  # Clean up
        else:
            whisper_segments = detect_ads_with_whisper(str(temp_file), podcast_name)
        
        # STEP 3: Calculate final cuts
        ad_cuts = {"start": 0, "end": 0}
        
        if whisper_segments:
            for seg in whisper_segments:
                if seg.get("type") == "detected":
                    ad_start = intro_cut + seg.get("start", 0)
                    ad_end = intro_cut + seg.get("end", 0)
                    logger.info(f"Whisper: ad from {ad_start:.1f}s to {ad_end:.1f}s")
                    # Use first detected ad segment
                    ad_cuts = {"start": ad_start, "end": ad_end}
                    break
        
        # STEP 4: Process with FFmpeg - cut intro and ads
        logger.info(f"Processing: {title} (intro cut: {intro_cut}s, ad cut: {ad_cuts})")
        
        # Calculate final cut points
        cut_start = intro_cut
        duration = get_duration(str(temp_file))
        
        if ad_cuts.get("end", 0) > 0:
            # Cut from end after ad ends
            final_duration = ad_cuts["end"] - cut_start
        else:
            final_duration = duration - cut_start
        
        # Apply cuts using stream copy (no re-encoding)
        if cut_start > 0 or ad_cuts.get("end", 0) > 0:
            if ad_cuts.get("end", 0) > 0:
                # Middle ad: concat Part A (start to intro_cut) + Part B (ad_end to end)
                # Use file-based concat demuxer with stream copy
                part_a = episode_dir / "part_a.mp3"
                part_b = episode_dir / "part_b.mp3"
                
                # Part A: start to intro_cut
                ffmpeg.input(str(temp_file), ss=0, t=cut_start).output(
                    str(part_a), acodec='copy'
                ).run(overwrite_output=True, quiet=True)
                
                # Part B: ad_end to end
                ffmpeg.input(str(temp_file), ss=ad_cuts["end"]).output(
                    str(part_b), acodec='copy'
                ).run(overwrite_output=True, quiet=True)
                
                # Concat Part A + Part B
                concat_list = episode_dir / "concat.txt"
                with open(concat_list, 'w') as f:
                    f.write(f"file '{part_a}'\n")
                    f.write(f"file '{part_b}'\n")
                
                ffmpeg.input(str(concat_list), format='concat', safe=0).output(
                    str(processed_file), acodec='copy'
                ).run(overwrite_output=True, quiet=True)
                
                # Cleanup temp parts
                part_a.unlink(missing_ok=True)
                part_b.unlink(missing_ok=True)
                concat_list.unlink(missing_ok=True)
            else:
                # No middle ad, just cut from intro to end
                ffmpeg.input(str(temp_file), ss=cut_start).output(
                    str(processed_file), acodec='copy'
                ).run(overwrite_output=True, quiet=True)
        else:
            # No processing needed, just copy
            processed_file.write_bytes(temp_file.read_bytes())
        
        # Cleanup temp
        temp_file.unlink()
        
        # Record processing time for 14-day cleanup
        metadata = {
            "processed_at": datetime.now().isoformat(),
            "title": title,
            "duration": duration
        }
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        PROCESSING[cache_key] = {"status": "ready", "title": title}
        logger.info(f"Ready: {title}")
        
        return send_file(processed_file, mimetype='audio/mpeg')
    
    except Exception as e:
        logger.error(f"Error processing {title}: {e}")
        PROCESSING[episode_id] = {"status": "error", "error": str(e)}
        return jsonify({"error": str(e)}), 500


def get_episode_info(podcast_name, episode_id):
    """Get episode audio URL from RSS - supports both sequential and stable hash IDs"""
    import hashlib
    rss_map = {
    "det_store_bilded": "https://rss.podplaystudio.com/692.xml",
    "pop_og_politikk": "https://rss.podplaystudio.com/4039.xml",
    "frelst": "https://www.omnycontent.com/d/playlist/b879ead7-5fbb-4c64-bede-b13801013ca4/5ffb3cc5-5bac-4124-9a56-b164009e28a9/7c34fe34-5d23-4ccb-ae13-b164009e28c5/podcast.rss",
    "trygdekontoret": "https://podkast.nrk.no/program/trygdekontoret.rss",
    "dwarkesh": "https://apple.dwarkesh-podcast.workers.dev/feed.rss",
    "den_politiske_situasjonen": "https://feeds.acast.com/public/shows/f6006cc3-2278-4017-b9ca-35d6e229e7fd",
    "fotballfortellinger": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/2b102ed8-af4a-4d51-ad41-b27b0106e973/46b58f60-9633-43f6-aa67-b27b0106e97a/podcast.rss",
    "the_rest_is_politics_leading": "https://feeds.megaphone.fm/GLT9029505120",
    "otto_og_anne": "https://rss.podplaystudio.com/745.xml",
    "heia_fotball": "https://podkast.nrk.no/program/heia_fotball.rss",
    "tid_er_penger": "https://feeds.acast.com/public/shows/659c418069d2da0016ac759b",
    "empire_world_history": "https://feeds.megaphone.fm/empirepodcast",
    "fladseth": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/afaede69-c88e-4075-aa19-b15400c5650b/7960a0b2-0102-4b8c-a665-b15400c56519/podcast.rss",
    "historier_som_endret_verden": "https://feeds.acast.com/public/shows/63889d03eaf58500117bbbca",
    "rett_og_slett": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/abdff6d2-d127-4cdf-8754-b15e00efcf82/c7367e4f-9138-4cf7-8c55-b15e00efcf95/podcast.rss",
    "alt_fortalt": "https://podcast.stream.schibsted.media/vgtv/100447?podcast",
    "peter_zeihan": "https://media.rss.com/zeihan/feed.xml",
    "pedagogisk_idrettspreik": "https://rss.libsyn.com/shows/112281/destinations/626995.xml",
    "psykodrama": "https://feeds.acast.com/public/shows/401782a2-4af0-4760-8505-6a2075d6887a",
    "chivadze": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/f0177c62-0df5-47e1-afbc-b15e00efd15b/0f467901-7814-49b9-8700-b15e00efd169/podcast.rss",
    "dritte_halbzeit": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/f4d4250b-5585-46f1-8f08-b149016d7f2b/d46c4794-37fc-460b-bbb5-b149016d7f55/podcast.rss",
    "when_we_were_kings": "https://feed.pod.space/whenwewerekings",
    "skisporet": "https://skisporet.libsyn.com/rss",
    "misjonen": "https://smartpod.no/feed/misjonen",
    "real_dictators": "https://feeds.megaphone.fm/NSR6187882468",
    "aftenpodden_usa": "https://podcast.stream.schibsted.media/ap/100196?podcast",
    "krigsrevyen": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/3a78fd64-b2e2-4c2c-a854-b27c00ed7886/b086f1a2-fbd3-44cf-9e05-b27c00ed789b/podcast.rss",
    "foreldrepodden": "https://rss.buzzsprout.com/1909514.rss",
    "teknisk_sett": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/5d96f449-e2b9-434a-8880-b14c007a47b3/6a1f66ca-8da3-4bf7-9935-b14c007a47ca/podcast.rss",
    "the_rest_is_classified": "https://feeds.megaphone.fm/therestisclassified",
    "lawfare": "https://feeds.acast.com/public/shows/60518a52f69aa815d2dba41c",
    "the_rest_is_history": "https://feeds.megaphone.fm/GLT4787413333",
    "historier_som_endret_norge": "https://feeds.acast.com/public/shows/63889ebd60bbdd001068859c",
    "krimpodden": "https://podcast.stream.schibsted.media/vgtv/100417?podcast",
    "fotballklubben": "https://anchor.fm/s/10854bc18/podcast/rss",
    "rekommandert": "https://feeds.acast.com/public/shows/f2b8c8dd-60de-4e34-b49d-a689f67ff31f",
    "hacks_on_tap": "https://feeds.megaphone.fm/VMP7545057845",
    "stormkast": "https://rss.podplaystudio.com/602.xml",
    "breaking_marathon_limits": "https://rss.buzzsprout.com/299975.rss",
    "political_gabfest": "https://my.slate.com/podcasts/feeds/political-gabfest/",
    "the_rest_is_politics_us": "https://feeds.megaphone.fm/GLT5336643697",
    "laligaloca": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/ea6e7251-b0a6-4148-8690-b15400c4d0b9/81f76f17-14b9-4845-912f-b15400c4d0c7/podcast.rss",
    "avhort": "https://feeds.acast.com/public/shows/685bb5014a2d8b8b51e33360",
    "altinget_partianalysen": "https://feeds.acast.com/public/shows/altinget-partianalysen",
    "olympiapodden": "https://feeds.soundcloud.com/users/soundcloud:users:420086565/sounds.rss",
    "war_on_the_rocks": "https://rss.libsyn.com/shows/70702/destinations/298196.xml",
    "the_drop": "https://anchor.fm/s/347e3928/podcast/rss",
    "oppdatert": "https://podkast.nrk.no/program/oppdatert.rss",
    "valgkampsirkuset": "https://feeds.acast.com/public/shows/655761451a7d7e0012c36c95",
    "hans_holund": "https://feeds.acast.com/public/shows/653f7814bfecf80012750eb9",
    "short_history_of": "https://feeds.megaphone.fm/NSR8236478683",
    "the_rest_is_politics": "https://feeds.megaphone.fm/GLT9190936013",
    "foreldreradet": "https://feeds.acast.com/public/shows/de647d54-d4a9-4ed5-b404-d5e0e1679630",
    "lopeskopodden": "https://www.omnycontent.com/d/playlist/d79bdbb3-8eae-4ad2-b1f6-b0bc00cd1540/3446a04c-4e9e-484d-864d-b27c00ed7950/25a6340a-216f-4e44-849a-b27c00ed795b/podcast.rss",
    "loperadet": "https://feeds.acast.com/public/shows/67c84bbd48f26a4bcaecfa09",
    "everything_everywhere_daily": "https://feeds.megaphone.fm/ADV3162807280",
    "dan_snow_history_hit": "https://access.acast.com/rss/c939f8d1-c4bc-478e-8bb9-e5343f9a7ab5",
    "ukraine_the_latest": "https://feeds.acast.com/public/shows/6583012e715d5300169dcdcb",
}
    
    original_rss = rss_map.get(podcast_name)
    if not original_rss:
        return None
    
    feed = feedparser.parse(original_rss)
    
    # Try both sequential ID and stable hash
    try:
        seq_id = int(episode_id) if episode_id.isdigit() else None
    except:
        seq_id = None
    
    for idx, entry in enumerate(feed.entries):
        # Get audio URL
        audio_url = None
        for link in entry.get('links', []):
            if link.get('type', '').startswith('audio/'):
                audio_url = link.get('href')
                break
        
        if not audio_url:
            continue
        
        # Check sequential ID match
        if seq_id is not None and (idx + 1) == seq_id:
            return {
                "title": entry.get('title', 'Episode'),
                "audio_url": audio_url
            }
        
        # Check stable hash match  
        guid = entry.get('id', audio_url)
        stable_id = hashlib.md5(guid.encode()).hexdigest()[:12]
        
        if stable_id == episode_id:
            return {
                "title": entry.get('title', 'Episode'),
                "audio_url": audio_url,
                "stable_id": stable_id
            }
    
    return None


def get_duration(audio_file):
    """Get audio duration in seconds"""
    try:
        probe = ffmpeg.probe(str(audio_file))
        return float(probe['format']['duration'])
    except:
        return 0

# ============================================================
# CLEANUP JOB - Delete episodes older than 14 days
# ============================================================

@app.route('/cleanup')
def run_cleanup():
    """Manually trigger cleanup (or call from cron)"""
    deleted = 0
    cutoff = datetime.now() - timedelta(days=14)
    
    for podcast_dir in STORAGE_DIR.iterdir():
        if not podcast_dir.is_dir():
            continue
        
        for episode_dir in podcast_dir.iterdir():
            if not episode_dir.is_dir():
                continue
            
            metadata_file = episode_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                processed_at = datetime.fromisoformat(metadata['processed_at'])
                if processed_at < cutoff:
                    # Delete episode files
                    for f in episode_dir.iterdir():
                        f.unlink()
                    episode_dir.rmdir()
                    deleted += 1
                    logger.info(f"Deleted old episode: {metadata['title']}")
    
    return jsonify({"deleted": deleted, "message": "Cleanup complete"})

# ============================================================
# STATUS & HEALTH
# ============================================================

@app.route('/status')
def status():
    """Get processing status"""
    return jsonify({
        "processing": PROCESSING,
        "storage_used": get_storage_size(),
        "episodes": count_episodes()
    })


def get_storage_size():
    """Get total storage used"""
    total = 0
    for f in STORAGE_DIR.rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)  # MB


def count_episodes():
    """Count cached episodes"""
    count = 0
    for podcast_dir in STORAGE_DIR.iterdir():
        if podcast_dir.is_dir():
            count += len([d for d in podcast_dir.iterdir() if d.is_dir()])
    return count


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3333, debug=False)
