"""
Podcast Ad Remover - On-Demand Processing (FIXED)
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
import sys
import hashlib
import threading
import json
import logging
import shutil
import functools
import tempfile
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import Optional, Dict, List, Any
from collections import OrderedDict
from functools import lru_cache

import requests
import feedparser
import ffmpeg
from flask import Flask, Response, request, redirect, jsonify, send_file

# ============================================================
# SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(threadName)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/data/podcast_ad_remover.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "https://podcast.drhahn.no").rstrip("/")
RSS_USERNAME = os.environ.get("RSS_USERNAME", "")
RSS_PASSWORD = os.environ.get("RSS_PASSWORD", "")

OPML_FILE = Path(os.environ.get("OPML_FILE", "/data/podcasts.opml"))
STORAGE_DIR = Path("/data/podcasts")
FINGERPRINT_DIR = Path("/data/fingerprints")
INTRO_CLIPS_DIR = FINGERPRINT_DIR # Intro clips stored alongside fingerprints

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
FINGERPRINT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# episode cache_key -> {"status": "processing"|"ready"|"error", "error": str, "progress": float}
PROCESSING = {}
PROCESSING_LOCK = threading.Lock()

# RSS cache: {rss_url: {"feed": parsed_feed, "cached_at": timestamp}}
RSS_CACHE: Dict[str, Dict[str, Any]] = {}
RSS_CACHE_LOCK = threading.Lock()
RSS_CACHE_TTL = 300 # seconds

# Fingerprint tuning
FINGERPRINT_SIMILARITY_THRESHOLD = 0.72 # Minimum cosine similarity to accept match
INTRO_SEARCH_MAX_SECONDS = 300 # Search for intro within first 5 minutes
TYPICAL_INTRO_DURATION = 30 # Assume intro tune is ~30s if auto-detecting

# Whisper tuning
WHISPER_CHUNK_MINUTES = 20 # Split episode into N-minute chunks for Whisper
WHISPER_MAX_MB = 23 # Stay under OpenAI 25MB limit per request

# Processing timeouts
PROCESSING_TIMEOUT_SECONDS = 300 # Max time to wait for processing before fallback
POLLING_INTERVAL_SECONDS = 2 # Client polling interval suggestion


# ============================================================
# DEPENDENCY VALIDATION
# ============================================================

def validate_dependencies() -> List[str]:
    """Check for required system and Python dependencies. Returns list of missing items."""
    missing = []
    
    # System binaries
    for cmd in ['ffmpeg', 'ffprobe']:
        if shutil.which(cmd) is None:
            missing.append(f"system command: {cmd}")
    
    # Python packages
    required_packages = ['librosa', 'numpy', 'openai', 'feedparser', 'requests', 'flask']
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(f"python package: {pkg}")
    
    # API key check (warning, not error - can run without Whisper)
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set - Whisper features disabled")
    
    return missing


def startup_checks():
    """Run validation checks at application startup."""
    missing = validate_dependencies()
    if missing:
        logger.error(f"Missing dependencies: {missing}")
        logger.error("Please install missing dependencies before starting the server.")
        # Don't crash - allow read-only operations to still work
    else:
        logger.info("All dependencies validated successfully")
    
    # Check storage permissions
    for d in [STORAGE_DIR, FINGERPRINT_DIR]:
        if not os.access(d, os.W_OK):
            logger.error(f"Cannot write to directory: {d}")


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
            .replace("<", "&lt;")
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
# RSS CACHING
# ============================================================

def get_cached_feed(rss_url: str) -> feedparser.FeedParserDict | None:
    """
    Get parsed RSS feed from cache or fetch and cache it.
    Cache TTL: RSS_CACHE_TTL seconds.
    """
    with RSS_CACHE_LOCK:
        cached = RSS_CACHE.get(rss_url)
        if cached and (datetime.now().timestamp() - cached['cached_at']) < RSS_CACHE_TTL:
            return cached['feed']
    
    try:
        logger.debug(f"Fetching RSS feed: {rss_url}")
        feed = feedparser.parse(rss_url)
        if feed.bozo:
            logger.warning(f"RSS parse warnings for {rss_url}: {feed.bozo_exception}")
        
        with RSS_CACHE_LOCK:
            RSS_CACHE[rss_url] = {
                'feed': feed,
                'cached_at': datetime.now().timestamp()
            }
        return feed
