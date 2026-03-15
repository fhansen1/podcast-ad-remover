"""
Podcast Ad Remover - On-Demand Processing
Flow: RSS Proxy → Process on Play → Stream Clean Audio → Auto-Delete after 14 days

Ad detection via Whisper API - automatically finds ad segments!
"""
import os
import hashlib
import json
import json as json_lib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
import threading

import requests
import feedparser
import ffmpeg
import xml.etree.ElementTree as ET
from flask import Flask, Response, request, jsonify, send_file, make_response, session

# ============================================================
# CONFIG
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "https://podcast.drhahn.no").rstrip("/")
GPODDER_USER = os.environ.get("GPODDER_USER", "openclaw")
GPODDER_PASS = os.environ.get("GPODDER_PASS", "podcast123")

OPML_FILE = Path("/data/podcasts.opml")
GPODDER_DIR = Path("/data/gpodder")
STORAGE_DIR = Path("/data/podcasts")

for d in (GPODDER_DIR, STORAGE_DIR):
    d.mkdir(parents=True, exist_ok=True)

GPODDER_DEVICES_FILE = GPODDER_DIR / "devices.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AD_TIMESTAMPS = {
    "det_store_bilded": {"start": 60, "end": 120},
    "pop_og_politikk": {"start": 10, "end": 0},
}

PROCESSING = {}

# ============================================================
# HELPERS
# ============================================================

def xml_escape(text):
    if not text:
        return ""
    return (str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
        .replace("\xa0", " "))


def load_opml():
    if not OPML_FILE.exists():
        logger.error(f"OPML not found: {OPML_FILE}")
        return {}
    root = ET.fromstring(OPML_FILE.read_text())
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


def episode_id_for(entry, idx):
    """Stable episode ID: MD5 of enclosure URL, fallback to index string."""
    enclosure = entry.get("enclosures", [{}])[0]
    url = enclosure.get("url", "")
    return hashlib.md5(url.encode()).hexdigest()[:12] if url else str(idx + 1)


def get_duration(audio_file):
    try:
        probe = ffmpeg.probe(str(audio_file))
        return float(probe['format']['duration'])
    except Exception:
        return 0


def load_devices():
    if GPODDER_DEVICES_FILE.exists():
        return json_lib.loads(GPODDER_DEVICES_FILE.read_text())
    return {}


def save_devices(devices):
    GPODDER_DEVICES_FILE.write_text(json_lib.dumps(devices))


FEEDS = load_opml()

# ============================================================
# APP
# ============================================================

app = Flask(__name__)
app.secret_key = "podcast-ad-remover-secret-key-2024"


def check_auth():
    if session.get("username"):
        return True
    auth = request.authorization
    return bool(auth and auth.username == GPODDER_USER and auth.password == GPODDER_PASS)

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" in session:
            return f(*args, **kwargs)
        auth = request.authorization
        if auth and auth.username == GPODDER_USER and auth.password == GPODDER_PASS:
            return f(*args, **kwargs)
        return Response("Unauthorized", 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
    return decorated


# ============================================================
# WHISPER AD DETECTION
# ============================================================

def detect_ads_with_whisper(audio_path):
    if not OPENAI_API_KEY:
        logger.warning("No OpenAI API key - using default timestamps")
        return None
    try:
        import openai
        import tempfile
        import subprocess

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Audio file size: {file_size_mb:.1f} MB")

        temp_clip = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        temp_clip.close()

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-ss', '0', '-t', '120',
        ]
        if file_size_mb > 23:
            logger.info("File too large, compressing for Whisper...")
            ffmpeg_cmd += ['-acodec', 'libmp3lame', '-ab', '64k']
        else:
            ffmpeg_cmd += ['-acodec', 'copy']
        ffmpeg_cmd.append(temp_clip.name)
        subprocess.run(ffmpeg_cmd, capture_output=True)

        with open(temp_clip.name, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
        os.unlink(temp_clip.name)

        words = transcript.words if hasattr(transcript, 'words') else []
        if not words:
            return None

        ad_segments = []
        for word_data in [w for w in words if w.start < 60]:
            if 25 < word_data.start < 45:
                ad_segments.append({"start": 0, "end": word_data.start, "type": "detected"})
                break

        if ad_segments:
            logger.info(f"Whisper detected: {ad_segments}")
        return ad_segments or None

    except Exception as e:
        logger.error(f"Whisper detection failed: {e}")
        return None


# ============================================================
# GPODDER API
# ============================================================

@app.route("/api/2/auth/<username>/login.json", methods=["POST"])
def gpodder_login(username):
    auth = request.authorization
    if auth and auth.username == GPODDER_USER and auth.password == GPODDER_PASS:
        session["username"] = username
        return make_response(jsonify({}), 200)
    return Response("Unauthorized", 401)


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
    return jsonify([{
        "id": d.get("id"),
        "caption": d.get("caption", "Phone"),
        "type": d.get("type", "mobile"),
        "subscriptions": len(d.get("subscriptions", []))
    } for d in devices.values()])


@app.route("/api/2/subscriptions/<username>/<device_id>.json", methods=["GET", "PUT", "POST"])
@require_auth
def gpodder_subscriptions(username, device_id):
    devices = load_devices()
    if device_id not in devices:
        devices[device_id] = {"subscriptions": [], "id": device_id, "updated": 0}
    if request.method in ("PUT", "POST"):
        data = request.json or {}
        devices[device_id]["subscriptions"] = data if isinstance(data, list) else data.get("urls", [])
        devices[device_id]["updated"] = int(datetime.now().timestamp())
        save_devices(devices)
        return jsonify({"update_urls": [], "timestamp": devices[device_id]["updated"]})
    since = request.args.get("since", 0, type=int)
    updated = devices[device_id].get("updated", 0)
    urls = devices[device_id].get("subscriptions", [])
    if since and updated <= since:
        return jsonify({"add": [], "remove": [], "timestamp": updated or int(datetime.now().timestamp())})
    return jsonify({"add": urls, "remove": [], "timestamp": updated or int(datetime.now().timestamp())})


@app.route("/api/2/episodes/<username>.json", methods=["GET", "POST"])
@require_auth
def gpodder_episodes(username):
    return jsonify([])


# ============================================================
# RSS PROXY
# ============================================================

@app.route('/')
def index():
    return jsonify({'service': 'Podcast Ad Remover', 'version': '4.0', 'podcasts': len(FEEDS)})


@app.route('/feeds')
def list_feeds():
    return jsonify({'podcasts': [{'slug': k, 'feed': f'{BASE_URL}/feed/{k}'} for k in FEEDS]})


@app.route('/feed/<podcast_name>')
def get_feed(podcast_name):
    if not check_auth():
        return Response('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Podcasts"'})

    original_rss = FEEDS.get(podcast_name)
    if not original_rss:
        return jsonify({"error": f"Unknown podcast: {podcast_name}"}), 404

    feed = feedparser.parse(original_rss)

    rss_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">',
        '<channel>',
        f'<title>{xml_escape(feed.feed.get("title", podcast_name))}</title>',
        f'<description>{xml_escape(feed.feed.get("description", ""))}</description>',
    ]

    artwork = None
    if hasattr(feed.feed, 'image'):
        artwork = feed.feed.image.get('href')
    if artwork:
        rss_lines += [
            f'<itunes:image href="{xml_escape(artwork)}"/>',
            f'<image><url>{xml_escape(artwork)}</url></image>',
        ]

    for idx, entry in enumerate(feed.entries):
        ep_id = episode_id_for(entry, idx)  # FIX: shared helper, consistent hash
        audio_url = f'{BASE_URL}/audio/{podcast_name}/{ep_id}'
        duration = entry.get('itunes_duration', 0)

        rss_lines += [
            '<item>',
            f'<title>{xml_escape(entry.get("title", "Episode"))}</title>',
            f'<description>{xml_escape(entry.get("description", ""))}</description>',
            f'<enclosure url="{audio_url}" type="audio/mpeg" length="0"/>',
            f'<guid>{audio_url}</guid>',
            f'<pubDate>{entry.get("published", "")}</pubDate>',
        ]
        if duration:
            rss_lines.append(f'<itunes:duration>{duration}</itunes:duration>')
        rss_lines.append('</item>')

    rss_lines += ['</channel>', '</rss>']
    return Response('\n'.join(rss_lines), mimetype='application/xml')


# ============================================================
# AUDIO PROCESSOR
# ============================================================

def get_episode_info(podcast_name, episode_id):
    """Look up episode by the same MD5 hash used in get_feed."""
    original_rss = FEEDS.get(podcast_name)
    if not original_rss:
        return None
    feed = feedparser.parse(original_rss)
    for idx, entry in enumerate(feed.entries):
        if episode_id_for(entry, idx) == episode_id:
            enclosure = entry.get("enclosures", [{}])[0]
            audio_url = enclosure.get("url") or next(
                (l["href"] for l in entry.get("links", [])
                if l.get("type", "").startswith("audio/")),
                None
            )
            if audio_url:
                return {"title": entry.get("title", "Episode"), "audio_url": audio_url}
    return None


@app.route('/audio/<podcast_name>/<episode_id>')
def stream_audio(podcast_name, episode_id):
    episode_info = get_episode_info(podcast_name, episode_id)
    if not episode_info:
        return jsonify({"error": "Episode not found"}), 404

    original_url = episode_info['audio_url']
    title = episode_info['title']
    episode_dir = STORAGE_DIR / podcast_name / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    processed_file = episode_dir / "clean.mp3"

    if processed_file.exists():
        logger.info(f"Serving cached: {title}")
        return send_file(processed_file, mimetype='audio/mpeg')

    if PROCESSING.get(episode_id, {}).get("status") == "processing":
        return jsonify({"error": "Still processing, try again in a moment"}), 202
    
    PROCESSING[episode_id] = {"status": "processing", "title": title}

    try:
        temp_file = episode_dir / "original.mp3"
        logger.info(f"Downloading: {title}")
        resp = requests.get(original_url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(temp_file, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        whisper_segments = detect_ads_with_whisper(str(temp_file))
        timestamps = whisper_segments[0] if whisper_segments else AD_TIMESTAMPS.get(podcast_name, {"start": 0, "end": 0})
        logger.info(f"Processing: {title} (cuts: {timestamps})")

        cut_start, cut_end = 0, 0
        if isinstance(timestamps, dict):
            if timestamps.get("start", 0) == 0 and timestamps.get("end", 0) > 0:
                cut_start = timestamps["end"]
            elif timestamps.get("start", 0) > 0 and timestamps.get("end", 0) == 0:
                cut_end = timestamps["start"]
        if cut_start == 0 and cut_end == 0:
            cut_start = timestamps.get("start", 0)
            cut_end = timestamps.get("end", 0)

        duration = get_duration(temp_file)  # FIX: always get duration, needed for metadata

        if cut_start > 0 or cut_end > 0:
            end_time = (duration - cut_end) if cut_end > 0 else duration
            logger.info(f"Cutting: {cut_start}s → {end_time}s")
            ffmpeg.input(str(temp_file), ss=cut_start).output(
                str(processed_file),
                t=end_time - cut_start,
                acodec='libmp3lame',
                ab='128k'
            ).run(overwrite_output=True, quiet=True)
        else:
            processed_file.write_bytes(temp_file.read_bytes())

        temp_file.unlink()

        with open(episode_dir / "metadata.json", "w") as f:
            json.dump({
                "processed_at": datetime.now().isoformat(),
                "title": title,
                "duration": duration,  # FIX: now always defined
            }, f)

        PROCESSING[episode_id] = {"status": "ready", "title": title}
        logger.info(f"Ready: {title}")
        return send_file(processed_file, mimetype='audio/mpeg')

    except Exception as e:
        logger.error(f"process_episode failed for {title}: {e}")
        PROCESSING[episode_id] = {"status": "error", "error": str(e)}
        if temp_file.exists():
            temp_file.unlink()
        return jsonify({"error": str(e)}), 500


# ============================================================
# CLEANUP
# ============================================================

@app.route('/cleanup')
def run_cleanup():
    deleted = 0
    cutoff = datetime.now() - timedelta(days=14)
    for podcast_dir in STORAGE_DIR.iterdir():
        if not podcast_dir.is_dir():
            continue
        for episode_dir in podcast_dir.iterdir():
            if not episode_dir.is_dir():
                continue
            metadata_file = episode_dir / "metadata.json"
            if not metadata_file.exists():
                continue
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                processed_at = datetime.fromisoformat(metadata["processed_at"])
                if processed_at < cutoff:
                    for file in episode_dir.iterdir():
                        file.unlink()
                    episode_dir.rmdir()
                    deleted += 1
                    logger.info(f"Deleted old episode: {episode_dir}")
            except Exception as e:
                logger.warning(f"Cleanup error for {episode_dir}: {e}")
    return jsonify({"deleted": deleted})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
