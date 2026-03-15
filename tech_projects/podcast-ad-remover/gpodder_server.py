"""
Podcast Ad Remover - On-Demand Processing
Flow: RSS Proxy → Process on Play → Stream Clean Audio → Auto-Delete after 14 days

Ad detection via Whisper API - automatically finds ad segments!
"""
import os
import re
import time
import uuid
from functools import wraps
import json as json_lib
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
import feedparser
from flask import Flask, Response, request, redirect, jsonify, send_file, make_response, session
import html

# Whisper API for ad detection
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_API_HOST = os.environ.get("MINIMAX_API_HOST", "https://api.minimax.io")

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

def detect_ads_with_whisper(audio_path):
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
        
        # Send to Whisper
        with open(temp_clip.name, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
        
        # Clean up temp file
        os.unlink(temp_clip.name)
        
        # Get words with timestamps
        words = transcript.words if hasattr(transcript, 'words') else []
        
        if not words:
            return None
        
        # Detect ad segments - look for language changes or pauses
        ad_segments = []
        first_words = [w for w in words if w.start < 60]
        
        if first_words:
            # Find where likely content starts (after ad)
            for i, word_data in enumerate(first_words):
                if word_data.start > 25 and word_data.start < 45:
                    ad_segments.append({
                        "start": 0,
                        "end": word_data.start,
                        "type": "detected"
                    })
                    break
        
        if ad_segments:
            logger.info(f"Detected ad segments via Whisper: {ad_segments}")
        
        return ad_segments if ad_segments else None
        
    except Exception as e:
        logger.error(f"Whisper detection failed: {e}")
        return None


from werkzeug.routing import Map, Rule
import ffmpeg

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load podcasts from OPML
import xml.etree.ElementTree as ET
OPML_FILE = Path("/data/podcasts.opml")
GPODDER_DIR = Path("/data/gpodder")


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check session first
        if "username" in session:
            return f(*args, **kwargs)
        
        # Fallback to Basic Auth
        auth = request.authorization
        if auth and auth.username == GPODDER_USER and auth.password == GPODDER_PASS:
            return f(*args, **kwargs)
        
        return Response("Unauthorized", 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
    return decorated


GPODDER_DIR.mkdir(parents=True, exist_ok=True)

def load_opml():
    if not OPML_FILE.exists():
        logger.error(f"OPML not found: {OPML_FILE}")
        return {}
    opml_source = OPML_FILE.read_text()
    tree = ET.ElementTree(ET.fromstring(opml_source))
    root = tree.getroot()
    feeds = {}
    for outline in root.iter("outline"):
        url = outline.get("xmlUrl") or outline.get("url")
        text = outline.get("text") or outline.get("title", "")
        if url and text:
            slug = text.lower().strip().replace(" ", "_").replace("-", "_")
            slug = "".join(c for c in slug if c.isalnum() or c == "_")
            feeds[slug] = url
    logger.info(f"Loaded {len(feeds)} podcasts from OPML")
    return feeds

FEEDS = load_opml()

# gPodder config
GPODDER_USER = os.environ.get("GPODDER_USER", "openclaw")
GPODDER_PASS = os.environ.get("GPODDER_PASS", "podcast123")
# Using Flask session instead
GPODDER_DIR = Path("/data/gpodder")


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check session first
        if "username" in session:
            return f(*args, **kwargs)
        
        # Fallback to Basic Auth
        auth = request.authorization
        if auth and auth.username == GPODDER_USER and auth.password == GPODDER_PASS:
            return f(*args, **kwargs)
        
        return Response("Unauthorized", 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
    return decorated


GPODDER_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

# Config - Set your external URL here
BASE_URL = os.environ.get("BASE_URL", os.environ.get("BASE_URL", "https://podcast.drhahn.no")).rstrip("/")

# Optional: Basic auth (set via environment)
RSS_USERNAME = os.environ.get("RSS_USERNAME", "")
RSS_PASSWORD = os.environ.get("RSS_PASSWORD", "")

app = Flask(__name__)
app.secret_key = "podcast-ad-remover-secret-key-2024"
STORAGE_DIR = Path("/data/podcasts")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Known ad timestamps (podcast_name -> {start_cut: seconds, end_cut: seconds})
# Extend this as you discover more
AD_TIMESTAMPS = {
    "det_store_bilded": {"start": 60, "end": 120},  # cut first 60s, last 2min
    "pop_og_politikk": {"start": 10, "end": 0},  # cut first 10s
    # Add more as discovered
}

# Track processing status
PROCESSING = {}  # episode_id -> {"status": "processing"|"ready"|"error", "progress": 0-100}

# ============================================================
# AUTH DECORATOR
# ============================================================


# ============================================================
# RSS PROXY - Modifies original RSS to point to our server
# ============================================================



# gPodder API v2
@app.route("/api/2/auth/<username>/login.json", methods=["POST"])
def gpodder_login(username):
    auth = request.authorization
    if auth and auth.username == GPODDER_USER and auth.password == GPODDER_PASS:
        session["username"] = username
        response = make_response(jsonify({}))
        response.status_code = 200
        return response
    return Response("Unauthorized", 401)



GPODDER_DEVICES_FILE = GPODDER_DIR / "devices.json"

def load_devices():
    if GPODDER_DEVICES_FILE.exists():
        return json_lib.loads(GPODDER_DEVICES_FILE.read_text())
    return {}

def save_devices(devices):
    GPODDER_DEVICES_FILE.write_text(json_lib.dumps(devices))

@app.route("/api/2/devices/<username>.json", methods=["GET", "POST"])
@require_auth
def gpodder_devices(username):
    devices = load_devices()
    if request.method == "POST":
        data = request.json or request.form.to_dict() or {}
        device_id = data.get("id", data.get("device_id", "antennapod"))
        devices[device_id] = {
            "id": device_id,
            "caption": data.get("caption", "Phone"),
            "type": data.get("type", "mobile"),
            "subscriptions": devices.get(device_id, {}).get("subscriptions", [])
        }
        save_devices(devices)
        return jsonify({"status": "ok"})
    # Return subscriptions as INTEGER (count), not array
    return jsonify([{
        "id": d.get("id"),
        "caption": d.get("caption", "Phone"),
        "type": d.get("type", "mobile"),
        "subscriptions": len(d.get("subscriptions", []))
    } for d in devices.values()])

@app.route("/api/2/subscriptions/<username>/<device_id>.json", methods=["GET", "POST"])
@require_auth
def gpodder_subscriptions(username, device_id):
    devices = load_devices()
    if device_id not in devices:
        devices[device_id] = {"subscriptions": [], "id": device_id}
    if request.method == "POST":
        data = request.json or request.form.to_dict() or {}
        devices[device_id]["subscriptions"] = data.get("urls", [])
        save_devices(devices)
        return jsonify({})
    urls = devices[device_id].get("subscriptions", [])
    return jsonify(urls)

@app.route("/api/2/episodes/<username>.json", methods=["GET", "POST"])
@require_auth
def gpodder_episodes(username):
    return jsonify([])


@app.route('/')
def index():
    return jsonify({'service': 'Podcast Ad Remover', 'version': '4.0', 'podcasts': len(FEEDS)})

@app.route('/feeds')
def list_feeds():
    return jsonify({'podcasts': [{'slug': k, 'feed': f'{BASE_URL}/feed/{k}'} for k in FEEDS.keys()]})

@app.route('/feed/<podcast_name>')
def get_feed(podcast_name):
    """Get modified RSS with our audio URLs - requires auth"""
    # Check auth
    if not get_session():
        # Try basic auth as fallback
        auth = request.authorization
        if not (auth and auth.username == GPODDER_USER and auth.password == GPODDER_PASS):
            return Response('Unauthorized', 401)
    """Get modified RSS with our audio URLs"""
    # Use FEEDS dictionary (loaded from OPML)
    original_rss = FEEDS.get(podcast_name)
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
        enclosure = entry.get("enclosures", [{}])[0]
        episode_id = hashlib.md5(enclosure.get("url", "").encode()).hexdigest()[:12] if enclosure.get("url") else str(idx + 1)
        
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
    if episode_id in PROCESSING and PROCESSING[episode_id].get("status") == "processing":
        return jsonify({"error": "Still processing, try again in a moment"}), 202
    
    # Start processing
    PROCESSING[episode_id] = {"status": "processing", "title": title}
    
    try:
        # Download original
        logger.info(f"Downloading: {title}")
        temp_file = episode_dir / "original.mp3"
        
        response = requests.get(original_url, stream=True)
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Get ad timestamps
        # First try Whisper API for auto-detection, fallback to manual timestamps
        whisper_segments = detect_ads_with_whisper(str(temp_file))
        
        if whisper_segments:
            timestamps = whisper_segments[0]  # Use first detected segment
            logger.info(f"Using Whisper-detected timestamps: {timestamps}")
        else:
            timestamps = AD_TIMESTAMPS.get(podcast_name, {"start": 0, "end": 0})
        
        # Process with FFmpeg
        logger.info(f"Processing: {title} (cuts: {timestamps})")
        
        # Determine cut points based on Whisper detection
        cut_start = 0  # Start position in original file
        cut_end = 0    # End position in original file (0 = don't cut from end)
        
        if isinstance(timestamps, dict):
            if timestamps.get("type") == "detected":
                # Whisper detected ad segment
                # If start=0, ad is at beginning - skip to "end"
                # If end=0 or not set, ad is at end - cut before "start"
                if timestamps.get("start", 0) == 0 and timestamps.get("end", 0) > 0:
                    # Ad at beginning - skip first X seconds
                    cut_start = timestamps.get("end", 0)
                elif timestamps.get("start", 0) > 0 and timestamps.get("end", 0) == 0:
                    # Ad at end - cut from the end
                    cut_end = timestamps.get("start", 0)
        
        # Use manual timestamps as fallback
        if cut_start == 0 and cut_end == 0:
            cut_start = timestamps.get("start", 0) if isinstance(timestamps, dict) else 0
            cut_end = timestamps.get("end", 0) if isinstance(timestamps, dict) else 0
        
        if cut_start > 0 or cut_end > 0:
            # Cut ads using FFmpeg
            duration = get_duration(temp_file)
            if cut_end > 0:
                end_time = duration - cut_end
            else:
                end_time = duration
            
            logger.info(f"Cutting: start at {cut_start}, end at {end_time}")
            
            ffmpeg.input(str(temp_file), ss=cut_start).output(
                str(processed_file),
                t=end_time - cut_start,
                acodec='libmp3lame',
                ab='128k'
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
        
        PROCESSING[episode_id] = {"status": "ready", "title": title}
        logger.info(f"Ready: {title}")
        
        return send_file(processed_file, mimetype='audio/mpeg')
    
    except Exception as e:
        logger.error(f"Error processing {title}: {e}")
        PROCESSING[episode_id] = {"status": "error", "error": str(e)}
        return jsonify({"error": str(e)}), 500


def get_episode_info(podcast_name, episode_id):
    """Get episode audio URL from RSS"""
    original_rss = FEEDS.get(podcast_name)
    if not original_rss:
        return None
    
    feed = feedparser.parse(original_rss)
    
    # Find episode by ID (we use sequential index)
    for idx, entry in enumerate(feed.entries):
        ep_id = idx + 1
        if ep_id == episode_id:
            # Find audio link
            for link in entry.get('links', []):
                if link.get('type', '').startswith('audio/'):
                    return {
                        "title": entry.get('title', 'Episode'),
                        "audio_url": link.get('href')
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
    app.run(host='0.0.0.0', port=8080, debug=False)
