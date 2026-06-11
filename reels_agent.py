"""
SciTech Reels Agent
Generates short vertical videos (Facebook Reels) from top news articles.
Picks 2-3 best articles per day, renders animated frames with article images,
adds ambient background audio, stitches into MP4, posts to Facebook as Reels
and drops a copy in Telegram promo group.
"""

import os
import re
import sys
import json
import time
import random
import base64
import logging
import requests
import psycopg2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image as PILImage
from playwright.sync_api import sync_playwright
from moviepy import (
    ImageClip,
    CompositeVideoClip,
    AudioClip,
    concatenate_videoclips,
    AudioFileClip,
    ColorClip,
    vfx,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Env ──

def _load_dotenv(path=".env"):
    try:
        p = Path(path)
        if not p.exists():
            return
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception as e:
        log.warning(f".env load failed: {e}")

_load_dotenv()

GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_PROMO_CHAT  = os.getenv("TELEGRAM_PROMO_CHAT")
FB_PAGE_ID           = os.getenv("FB_PAGE_ID")
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")
DATABASE_URL         = os.getenv("DATABASE_URL")
SITE_DOMAIN          = "scitech.top"

REQUIRED = ["GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_PROMO_CHAT",
            "FB_PAGE_ID", "FB_PAGE_ACCESS_TOKEN", "DATABASE_URL"]
missing = [k for k in REQUIRED if not os.getenv(k)]
if missing:
    log.error(f"Missing required env vars: {missing}")
    sys.exit(2)

DATA_DIR   = Path(os.getenv("DATA_DIR", ".")).resolve()
FRAMES_DIR = DATA_DIR / "frames";  FRAMES_DIR.mkdir(parents=True, exist_ok=True)
REELS_DIR  = DATA_DIR / "reels";   REELS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = DATA_DIR / "images";  IMAGES_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_MODEL    = "gemini-2.5-flash"
GEMINI_TEXT_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

W, H = 1080, 1920  # vertical Reel

CATEGORY_COLORS = {
    "AI": "#c084fc", "Cyber": "#f87171", "Robotics": "#38bdf8",
    "Technology": "#00e5c0", "Space": "#818cf8", "Energy": "#fbbf24",
    "Health": "#34d399", "Science": "#60a5fa", "General": "#94a3b8",
}

CATEGORY_ICONS = {
    "AI": "🤖", "Cyber": "🔒", "Robotics": "🦾", "Space": "🚀",
    "Technology": "💡", "Energy": "⚡", "Health": "🧬", "Science": "🔬",
    "General": "📰",
}


# ── Gemini ──

def gemini_text(prompt, label="gemini", max_output_tokens=2048, retries=2, timeout=120):
    gen_cfg = {
        "responseMimeType": "application/json",
        "maxOutputTokens": max_output_tokens,
        "temperature": 0.7,
        "thinkingConfig": {"thinkingBudget": 0},
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    }
    rate_limit_hits = 0
    attempt = 0
    while attempt <= retries + rate_limit_hits:
        try:
            resp = requests.post(
                GEMINI_TEXT_URL, params={"key": GEMINI_API_KEY},
                json=payload, timeout=timeout,
            )
            if resp.status_code == 429:
                rate_limit_hits += 1
                if rate_limit_hits > 5:
                    return None
                time.sleep(min(30 * (2 ** (rate_limit_hits - 1)), 300))
                continue
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                attempt += 1; continue
            parts = candidates[0].get("content", {}).get("parts", [])
            text = parts[0].get("text", "") if parts else ""
            return _parse_json(text)
        except Exception as e:
            log.error(f"[{label}] attempt {attempt}: {e}")
            attempt += 1
            time.sleep(2 ** attempt)
    return None


def _parse_json(raw):
    if not raw:
        return None
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if clean.lower().startswith("json"):
            clean = clean[4:].strip()
    try:
        return json.loads(clean)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return None


# ── Article Image Fetcher ──

def fetch_article_image(website_url):
    if not website_url:
        return None
    try:
        resp = requests.get(website_url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; SciTechReelsBot/1.0)"
        })
        resp.raise_for_status()
        html = resp.text
        m = re.search(r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if not m:
            m = re.search(r'<meta\s+content=["\']([^"\']+)["\']\s+property=["\']og:image["\']', html, re.IGNORECASE)
        if not m:
            log.info(f"[image] no og:image found on {website_url}")
            return None
        img_url = m.group(1)
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        elif img_url.startswith("/"):
            from urllib.parse import urlparse
            parsed = urlparse(website_url)
            img_url = f"{parsed.scheme}://{parsed.netloc}{img_url}"

        img_resp = requests.get(img_url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (compatible; SciTechReelsBot/1.0)"
        })
        img_resp.raise_for_status()

        out = IMAGES_DIR / f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        # Save raw bytes first, then resize to keep memory low
        tmp = IMAGES_DIR / f"_tmp_{datetime.now().strftime('%H%M%S')}"
        tmp.write_bytes(img_resp.content)
        try:
            img = PILImage.open(tmp)
            img = img.convert("RGB")
            # Resize to max 1080px wide, preserving aspect ratio
            if img.width > 1080:
                ratio = 1080 / img.width
                img = img.resize((1080, int(img.height * ratio)), PILImage.LANCZOS)
            img.save(str(out), "JPEG", quality=85)
            tmp.unlink(missing_ok=True)
        except Exception as e:
            log.warning(f"[image] resize failed, using raw: {e}")
            tmp.rename(out)
        log.info(f"[image] article image ready: {out.name} ({out.stat().st_size / 1024:.0f} KB)")
        return out
    except Exception as e:
        log.warning(f"[image] failed to fetch article image: {e}")
        return None


def _img_to_data_url(path):
    suffix = Path(path).suffix.lower().lstrip(".") or "jpeg"
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}
    mime = mime_map.get(suffix, "jpeg")
    with open(path, "rb") as f:
        return f"data:image/{mime};base64,{base64.b64encode(f.read()).decode()}"


# ── Database ──

def db_connect():
    return psycopg2.connect(DATABASE_URL)


def _ensure_reels_table():
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""CREATE TABLE IF NOT EXISTS reels_posted (
                    article_id TEXT PRIMARY KEY,
                    posted_at  TIMESTAMPTZ DEFAULT NOW(),
                    fb_video_id TEXT,
                    tg_ok      BOOLEAN DEFAULT FALSE
                )""")
    except Exception as e:
        log.error(f"[db] create reels_posted table failed: {e}")


def get_reel_candidates(limit=3):
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT p.article_id, p.title_en, p.title_ar, p.headline_ar,
                              p.category, p.source, p.website_url, p.posted_at
                       FROM posts p
                       LEFT JOIN reels_posted r ON p.article_id = r.article_id
                       WHERE r.article_id IS NULL
                         AND p.website_url IS NOT NULL
                         AND p.posted_at > NOW() - INTERVAL '3 days'
                       ORDER BY p.importance DESC, p.posted_at DESC
                       LIMIT %s""",
                    (limit,),
                )
                cols = ["article_id", "title_en", "title_ar", "headline_ar",
                        "category", "source", "website_url", "posted_at"]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        log.error(f"[db] get_reel_candidates failed: {e}")
        return []


def mark_reel_posted(article_id, fb_video_id=None, tg_ok=False):
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO reels_posted (article_id, fb_video_id, tg_ok)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (article_id) DO UPDATE
                       SET fb_video_id = EXCLUDED.fb_video_id, tg_ok = EXCLUDED.tg_ok""",
                    (article_id, fb_video_id, tg_ok),
                )
    except Exception as e:
        log.error(f"[db] mark_reel_posted failed: {e}")


# ── Gemini: Generate Reel Script ──

def generate_reel_script(article):
    prompt = f"""You are a video script writer for "ساي تك" (SciTech), an Arabic science & tech news channel.
Write a short script for a 15-second Facebook Reel about this article.

Article:
- Title (EN): {article.get('title_en', '')}
- Title (AR): {article.get('title_ar', '')}
- Headline (AR): {article.get('headline_ar', '')}
- Category: {article.get('category', '')}

Return JSON with these fields:
- "hook": A short attention-grabbing Arabic question or statement (max 8 words) - this appears first in the video
- "headline": The main Arabic headline for the video (max 15 words)
- "detail": One key detail or interesting fact about the article in Arabic (max 20 words)
- "cta": A call-to-action in Arabic like "تابعونا" or "اقرأ المزيد" (max 6 words)

Keep all text short — it will be displayed on a vertical video. Shorter = more impactful."""

    return gemini_text(prompt, label="reel-script", max_output_tokens=1024)


# ── Background Audio ──

def generate_ambient_audio(duration, sr=44100):
    """Generate a soft ambient news-style background audio using synthesized tones."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Deep warm pad
    pad = 0.12 * np.sin(2 * np.pi * 80 * t)
    pad += 0.08 * np.sin(2 * np.pi * 120 * t)
    pad += 0.05 * np.sin(2 * np.pi * 160 * t)

    # Shimmering high tone with slow modulation
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
    shimmer = 0.04 * np.sin(2 * np.pi * 440 * t) * mod
    shimmer += 0.03 * np.sin(2 * np.pi * 554 * t) * (1 - mod)
    shimmer += 0.02 * np.sin(2 * np.pi * 660 * t) * mod

    # Gentle rising tone for urgency/news feel
    sweep_freq = 200 + 100 * (t / duration)
    sweep = 0.03 * np.sin(2 * np.pi * sweep_freq * t)
    sweep *= np.sin(np.pi * t / duration)  # fade in/out envelope

    audio = pad + shimmer + sweep

    # Smooth fade in (1.5s) and fade out (2s)
    fade_in_samples = int(1.5 * sr)
    fade_out_samples = int(2.0 * sr)
    audio[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
    audio[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.35

    # Stereo
    stereo = np.column_stack([audio, audio])

    def make_frame(t_arr):
        t_arr = np.atleast_1d(np.asarray(t_arr, dtype=np.float64))
        indices = (t_arr * sr).astype(int)
        indices = np.clip(indices, 0, len(stereo) - 1)
        return stereo[indices]

    return AudioClip(make_frame, duration=duration, fps=sr)


# ── Frame Rendering (Playwright) ──

def _html_esc(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _logo_b64():
    logo = Path("logo.png")
    if logo.exists():
        with open(logo, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    return None


FRAME_CSS_BASE = """
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{width:{W}px;height:{H}px;overflow:hidden;font-family:'Noto Kufi Arabic',sans-serif;
     position:relative;display:flex;align-items:center;justify-content:center;}}
"""

def _frame_bg_css(accent, glow):
    return f"""
body{{background:#080c18;}}
.mesh{{position:absolute;inset:0;
  background:
    radial-gradient(ellipse 90% 50% at 20% 80%, {accent}44 0%, transparent 55%),
    radial-gradient(ellipse 70% 60% at 80% 20%, {glow}44 0%, transparent 50%),
    radial-gradient(ellipse 50% 70% at 50% 50%, {accent}15 0%, transparent 65%);}}
.grid{{position:absolute;inset:0;opacity:0.05;
  background-image:
    linear-gradient(rgba(255,255,255,0.3) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.3) 1px, transparent 1px);
  background-size:54px 54px;}}
.orb1{{position:absolute;width:400px;height:400px;border-radius:50%;
  background:radial-gradient(circle, {accent}55 0%, transparent 70%);
  top:-100px;right:-80px;filter:blur(80px);}}
.orb2{{position:absolute;width:350px;height:350px;border-radius:50%;
  background:radial-gradient(circle, {glow}44 0%, transparent 70%);
  bottom:-80px;left:-60px;filter:blur(70px);}}
.orb3{{position:absolute;width:250px;height:250px;border-radius:50%;
  background:radial-gradient(circle, {accent}30 0%, transparent 70%);
  top:35%;left:50%;filter:blur(50px);}}
.corner-tl,.corner-br{{position:absolute;width:50px;height:50px;z-index:5;}}
.corner-tl{{top:30px;left:30px;border-top:2px solid {accent}55;border-left:2px solid {accent}55;}}
.corner-br{{bottom:30px;right:30px;border-bottom:2px solid {accent}55;border-right:2px solid {accent}55;}}
"""


def _bg_divs():
    return '<div class="mesh"></div><div class="grid"></div><div class="orb1"></div><div class="orb2"></div><div class="orb3"></div><div class="corner-tl"></div><div class="corner-br"></div>'


def render_frame_intro(accent, glow, logo_b64):
    logo_img = f'<img src="{logo_b64}" style="width:180px;margin-bottom:40px;filter:drop-shadow(0 4px 30px {accent}66);">' if logo_b64 else ""
    html = f"""<!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@400;700;900&display=swap" rel="stylesheet">
<style>
{FRAME_CSS_BASE.format(W=W, H=H)}
{_frame_bg_css(accent, glow)}
.content{{position:relative;z-index:10;text-align:center;display:flex;flex-direction:column;align-items:center;}}
.brand{{font-size:72px;font-weight:900;color:#fff;margin-bottom:16px;
  text-shadow:0 4px 50px {accent}66, 0 2px 20px rgba(0,0,0,0.5);}}
.tagline{{font-size:28px;color:rgba(255,255,255,0.75);font-weight:500;}}
.bar{{width:120px;height:4px;background:linear-gradient(90deg,{accent},{glow});border-radius:2px;
  margin:30px auto;box-shadow:0 0 30px {accent}88;}}
</style></head><body>
{_bg_divs()}
<div class="content">
  {logo_img}
  <div class="brand">ساي تك</div>
  <div class="bar"></div>
  <div class="tagline">أخبار العلوم والتكنولوجيا</div>
</div>
</body></html>"""
    return html


def render_frame_hook(hook_text, category, accent, glow):
    icon = CATEGORY_ICONS.get(category, "📰")
    html = f"""<!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@400;700;900&display=swap" rel="stylesheet">
<style>
{FRAME_CSS_BASE.format(W=W, H=H)}
{_frame_bg_css(accent, glow)}
.content{{position:relative;z-index:10;text-align:center;padding:60px;display:flex;flex-direction:column;align-items:center;justify-content:center;}}
.icon{{font-size:100px;margin-bottom:40px;filter:drop-shadow(0 4px 30px rgba(0,0,0,0.4));}}
.hook{{font-size:52px;font-weight:900;color:#fff;line-height:1.4;max-width:900px;
  text-shadow:0 4px 40px rgba(0,0,0,0.6), 0 0 80px {accent}33;}}
.cat-badge{{position:absolute;top:60px;right:60px;background:{accent};color:#000;padding:12px 28px;
  border-radius:12px;font-weight:900;font-size:20px;letter-spacing:1px;}}
</style></head><body>
{_bg_divs()}
<div class="content">
  <div class="cat-badge">{_html_esc(category)}</div>
  <div class="icon">{icon}</div>
  <div class="hook">{_html_esc(hook_text)}</div>
</div>
</body></html>"""
    return html


def render_frame_article_image(article_img_path, headline_text, source, category, accent, glow):
    """Frame that shows the actual article image as full background with headline overlay."""
    bg_data = _img_to_data_url(article_img_path)
    hl_len = len(headline_text or "")
    font_size = 40 if hl_len > 60 else 48 if hl_len > 35 else 56
    html = f"""<!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@400;700;900&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{width:{W}px;height:{H}px;overflow:hidden;font-family:'Noto Kufi Arabic',sans-serif;position:relative;background:#000;}}
.bg{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center;}}
.overlay{{position:absolute;inset:0;
  background:linear-gradient(180deg,
    rgba(0,0,0,0.1) 0%,
    rgba(0,0,0,0.05) 20%,
    rgba(0,0,0,0.15) 40%,
    rgba(0,0,0,0.6) 65%,
    rgba(0,0,0,0.92) 85%,
    rgba(0,0,0,0.98) 100%);}}
.cat-badge{{position:absolute;top:50px;right:50px;background:{accent};color:#000;padding:12px 28px;
  border-radius:12px;font-weight:900;font-size:22px;letter-spacing:1px;
  box-shadow:0 4px 20px rgba(0,0,0,0.4);z-index:20;}}
.source-tag{{position:absolute;top:50px;left:50px;background:rgba(0,0,0,0.65);color:#fff;
  padding:10px 22px;border-radius:10px;font-weight:700;font-size:16px;
  backdrop-filter:blur(10px);z-index:20;}}
.bottom{{position:absolute;bottom:0;left:0;right:0;padding:50px 55px 80px;z-index:15;text-align:right;}}
.accent-line{{width:80px;height:5px;background:{accent};margin:0 0 20px auto;border-radius:3px;
  box-shadow:0 0 25px {accent}aa;}}
.headline{{font-size:{font_size}px;font-weight:900;color:#fff;line-height:1.35;
  text-shadow:0 3px 20px rgba(0,0,0,0.9);}}
.domain{{position:absolute;bottom:30px;left:55px;font-size:16px;color:rgba(255,255,255,0.6);
  font-weight:600;letter-spacing:2px;z-index:20;}}
.corner-tl,.corner-br{{position:absolute;width:50px;height:50px;z-index:5;}}
.corner-tl{{top:30px;left:30px;border-top:2px solid {accent}55;border-left:2px solid {accent}55;}}
.corner-br{{bottom:30px;right:30px;border-bottom:2px solid {accent}55;border-right:2px solid {accent}55;}}
</style></head><body>
<img src="{bg_data}" class="bg" alt="bg">
<div class="overlay"></div>
<div class="corner-tl"></div>
<div class="corner-br"></div>
<div class="source-tag">{_html_esc(source)}</div>
<div class="cat-badge">{_html_esc(category)}</div>
<div class="bottom">
  <div class="accent-line"></div>
  <div class="headline">{_html_esc(headline_text)}</div>
</div>
<div class="domain">scitech.top</div>
</body></html>"""
    return html


def render_frame_headline(headline_text, accent, glow):
    """Fallback headline frame when no article image is available."""
    hl_len = len(headline_text or "")
    font_size = 44 if hl_len > 50 else 52 if hl_len > 30 else 60
    html = f"""<!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@400;700;900&display=swap" rel="stylesheet">
<style>
{FRAME_CSS_BASE.format(W=W, H=H)}
{_frame_bg_css(accent, glow)}
.content{{position:relative;z-index:10;text-align:center;padding:80px 60px;display:flex;flex-direction:column;align-items:center;justify-content:center;}}
.bar{{width:80px;height:5px;background:linear-gradient(90deg,{accent},{glow});border-radius:3px;
  margin-bottom:40px;box-shadow:0 0 30px {accent}88;}}
.headline{{font-size:{font_size}px;font-weight:900;color:#fff;line-height:1.45;max-width:920px;
  text-shadow:0 4px 40px rgba(0,0,0,0.6), 0 0 60px {accent}22;}}
</style></head><body>
{_bg_divs()}
<div class="content">
  <div class="bar"></div>
  <div class="headline">{_html_esc(headline_text)}</div>
</div>
</body></html>"""
    return html


def render_frame_detail(detail_text, accent, glow):
    html = f"""<!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@400;700;900&display=swap" rel="stylesheet">
<style>
{FRAME_CSS_BASE.format(W=W, H=H)}
{_frame_bg_css(accent, glow)}
.content{{position:relative;z-index:10;text-align:center;padding:80px 60px;display:flex;flex-direction:column;align-items:center;justify-content:center;}}
.detail-box{{background:rgba(255,255,255,0.06);backdrop-filter:blur(20px);
  border:1px solid {accent}44;border-radius:24px;padding:50px 45px;max-width:900px;
  box-shadow:0 0 40px {accent}15, inset 0 0 30px {accent}08;}}
.detail{{font-size:38px;font-weight:700;color:#fff;line-height:1.5;
  text-shadow:0 2px 20px rgba(0,0,0,0.4);}}
.quote-mark{{font-size:80px;color:{accent};opacity:0.4;margin-bottom:10px;}}
</style></head><body>
{_bg_divs()}
<div class="content">
  <div class="detail-box">
    <div class="quote-mark">❝</div>
    <div class="detail">{_html_esc(detail_text)}</div>
  </div>
</div>
</body></html>"""
    return html


def render_frame_cta(cta_text, url, accent, glow, logo_b64):
    logo_img = f'<img src="{logo_b64}" style="width:120px;margin-bottom:30px;filter:drop-shadow(0 4px 20px {accent}55);">' if logo_b64 else ""
    html = f"""<!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@400;700;900&display=swap" rel="stylesheet">
<style>
{FRAME_CSS_BASE.format(W=W, H=H)}
{_frame_bg_css(accent, glow)}
.content{{position:relative;z-index:10;text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;}}
.cta{{font-size:48px;font-weight:900;color:#fff;margin-bottom:30px;
  text-shadow:0 4px 40px rgba(0,0,0,0.5);}}
.btn{{background:linear-gradient(135deg,{accent},{glow});color:#000;font-weight:900;
  font-size:26px;padding:20px 50px;border-radius:16px;
  box-shadow:0 8px 40px {accent}55;letter-spacing:1px;}}
.domain{{margin-top:40px;font-size:22px;color:rgba(255,255,255,0.6);font-weight:600;letter-spacing:3px;}}
.bar{{width:80px;height:4px;background:linear-gradient(90deg,{accent},{glow});border-radius:2px;
  margin:30px auto;box-shadow:0 0 20px {accent}88;}}
</style></head><body>
{_bg_divs()}
<div class="content">
  {logo_img}
  <div class="cta">{_html_esc(cta_text)}</div>
  <div class="bar"></div>
  <div class="btn">scitech.top</div>
  <div class="domain">أخبار العلوم والتكنولوجيا</div>
</div>
</body></html>"""
    return html


def render_html_to_png(html, out_path):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": W, "height": H})
            page.set_content(html, wait_until="networkidle")
            page.wait_for_timeout(2000)
            page.screenshot(path=str(out_path))
            browser.close()
        return out_path
    except Exception as e:
        log.error(f"[render] frame error: {e}")
        return None


def render_all_frames(article, script, article_img_path=None):
    category = article.get("category", "Technology")
    source = article.get("source", "SciTech")
    accent = CATEGORY_COLORS.get(category, "#00e5c0")

    glow_map = {
        "#c084fc": "#f472b6", "#f87171": "#fbbf24", "#38bdf8": "#818cf8",
        "#00e5c0": "#38bdf8", "#818cf8": "#c084fc", "#fbbf24": "#f87171",
        "#34d399": "#38bdf8", "#60a5fa": "#c084fc", "#94a3b8": "#60a5fa",
    }
    glow = glow_map.get(accent, "#38bdf8")
    logo_b64 = _logo_b64()
    aid = article.get("article_id", "x")[:12]
    ts = datetime.now().strftime("%H%M%S")

    hook = script.get("hook", "خبر جديد")
    headline = script.get("headline", article.get("headline_ar", "خبر عاجل"))
    detail = script.get("detail", "")
    cta = script.get("cta", "تابعونا على ساي تك")

    frames_spec = [
        ("intro",    render_frame_intro(accent, glow, logo_b64)),
        ("hook",     render_frame_hook(hook, category, accent, glow)),
    ]

    if article_img_path and Path(article_img_path).exists():
        frames_spec.append(("article", render_frame_article_image(
            article_img_path, headline, source, category, accent, glow)))
    else:
        frames_spec.append(("headline", render_frame_headline(headline, accent, glow)))

    if detail:
        frames_spec.append(("detail", render_frame_detail(detail, accent, glow)))
    frames_spec.append(("cta", render_frame_cta(cta, article.get("website_url", ""), accent, glow, logo_b64)))

    paths = []
    for name, html in frames_spec:
        out = FRAMES_DIR / f"{aid}_{ts}_{name}.png"
        result = render_html_to_png(html, out)
        if result:
            paths.append(result)
        else:
            log.warning(f"[render] frame '{name}' failed, skipping")

    return paths


# ── Video Stitching (MoviePy) ──

def create_reel_video(frame_paths, article_id):
    if len(frame_paths) < 3:
        log.error("[video] need at least 3 frames to create a reel")
        return None

    durations = []
    n = len(frame_paths)
    if n == 4:
        durations = [2.5, 3.0, 5.0, 3.5]
    elif n == 5:
        durations = [2.0, 2.5, 4.5, 3.5, 3.0]
    else:
        durations = [3.0] * n

    fade = 0.6

    clips = []
    for i, (fpath, dur) in enumerate(zip(frame_paths, durations)):
        clip = ImageClip(str(fpath), duration=dur)
        clip = clip.resized((W, H))
        zoom_factor = 1.06
        clip = clip.resized(lambda t, d=dur: 1 + (zoom_factor - 1) * (t / d))
        clip = clip.with_position("center")
        if i > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(fade)])
        if i < n - 1:
            clip = clip.with_effects([vfx.CrossFadeOut(fade)])
        clips.append(clip)

    final = concatenate_videoclips(clips, method="compose", padding=-fade)
    final = final.resized((W, H))

    total_duration = final.duration

    # Add background audio
    audio = None
    try:
        bgm_path = Path("bgm.mp3")
        if bgm_path.exists():
            log.info("[audio] using custom bgm.mp3")
            audio = AudioFileClip(str(bgm_path)).subclipped(0, min(total_duration, 60))
            if audio.duration < total_duration:
                audio = audio.with_effects([vfx.Loop(duration=total_duration)])
            audio = audio.with_effects([vfx.MultiplyVolume(0.4)])
        else:
            log.info("[audio] generating ambient background audio")
            audio = generate_ambient_audio(total_duration)
    except Exception as e:
        log.warning(f"[audio] audio generation failed, continuing without: {e}")
        audio = None

    if audio is not None:
        final = final.with_audio(audio)

    out_path = REELS_DIR / f"reel_{article_id[:12]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

    final.write_videofile(
        str(out_path),
        fps=30,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="128k",
        preset="medium",
        bitrate="4000k",
        logger=None,
    )

    log.info(f"[video] reel created: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


# ── Facebook Reels API ──

def post_facebook_reel(video_path, title, description):
    try:
        file_size = Path(video_path).stat().st_size
        log.info(f"[facebook] initializing reel upload ({file_size / 1024:.0f} KB)")
        init_resp = requests.post(
            f"https://graph.facebook.com/v19.0/{FB_PAGE_ID}/video_reels",
            params={"access_token": FB_PAGE_ACCESS_TOKEN},
            json={"upload_phase": "start"},
            timeout=30,
        )
        init_resp.raise_for_status()
        init_data = init_resp.json()
        video_id = init_data["video_id"]
        upload_url = init_data.get("upload_url")
        log.info(f"[facebook] upload initialized, video_id={video_id}")

        if upload_url:
            with open(video_path, "rb") as f:
                upload_resp = requests.post(
                    upload_url,
                    headers={
                        "Authorization": f"OAuth {FB_PAGE_ACCESS_TOKEN}",
                        "offset": "0",
                        "file_size": str(file_size),
                    },
                    data=f,
                    timeout=120,
                )
            upload_resp.raise_for_status()
            log.info("[facebook] video binary uploaded")
        else:
            with open(video_path, "rb") as f:
                upload_resp = requests.post(
                    f"https://rupload.facebook.com/video-upload/v19.0/{video_id}",
                    headers={
                        "Authorization": f"OAuth {FB_PAGE_ACCESS_TOKEN}",
                        "offset": "0",
                        "file_size": str(file_size),
                    },
                    data=f,
                    timeout=120,
                )
            upload_resp.raise_for_status()
            log.info("[facebook] video binary uploaded (rupload)")

        publish_resp = requests.post(
            f"https://graph.facebook.com/v19.0/{FB_PAGE_ID}/video_reels",
            params={"access_token": FB_PAGE_ACCESS_TOKEN},
            json={
                "upload_phase": "finish",
                "video_id": video_id,
                "title": title[:80],
                "description": description[:2000],
            },
            timeout=60,
        )
        publish_resp.raise_for_status()
        pub_data = publish_resp.json()
        log.info(f"[facebook] reel published! response: {pub_data}")
        return video_id

    except Exception as e:
        log.error(f"[facebook] reel upload failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            log.error(f"[facebook] response body: {e.response.text}")
        return None


# ── Telegram ──

def send_reel_to_telegram(video_path, caption):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
        with open(video_path, "rb") as f:
            resp = requests.post(url, data={
                "chat_id": TELEGRAM_PROMO_CHAT,
                "caption": caption[:1024],
                "parse_mode": "HTML",
                "supports_streaming": "true",
            }, files={"video": f}, timeout=120)
        if resp.ok:
            log.info("[telegram] reel video sent")
            return True
        else:
            log.error(f"[telegram] send failed: {resp.status_code} {resp.text}")
    except Exception as e:
        log.error(f"[telegram] error: {e}")
    return False


# ── Cleanup ──

def cleanup_old_files(max_age_hours=24):
    cutoff = time.time() - (max_age_hours * 3600)
    for d in [FRAMES_DIR, REELS_DIR, IMAGES_DIR]:
        for f in d.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)


# ── Main ──

def run():
    log.info("=" * 50)
    log.info("  SciTech Reels Agent - Starting")
    log.info("=" * 50)

    _ensure_reels_table()

    candidates = get_reel_candidates(limit=2)
    if not candidates:
        log.info("[reels] no new articles to make reels from, done")
        return

    log.info(f"[reels] found {len(candidates)} candidates")

    for i, article in enumerate(candidates):
        log.info(f"\n--- Reel {i+1}/{len(candidates)}: {article.get('title_en', '')[:60]} ---")

        script = generate_reel_script(article)
        if not script:
            log.error(f"[reels] script generation failed for {article['article_id']}, skipping")
            continue

        log.info(f"[reels] script: hook='{script.get('hook', '')[:40]}' headline='{script.get('headline', '')[:40]}'")

        article_img = fetch_article_image(article.get("website_url"))
        if article_img:
            log.info(f"[reels] article image ready: {article_img.name}")
        else:
            log.info("[reels] no article image, using styled headline frame")

        frame_paths = render_all_frames(article, script, article_img)
        if len(frame_paths) < 3:
            log.error(f"[reels] only {len(frame_paths)} frames rendered, need at least 3, skipping")
            continue

        log.info(f"[reels] rendered {len(frame_paths)} frames")

        video_path = create_reel_video(frame_paths, article["article_id"])
        if not video_path:
            log.error(f"[reels] video creation failed, skipping")
            continue

        fb_video_id = post_facebook_reel(
            video_path,
            title=script.get("headline", article.get("headline_ar", "")),
            description=f"{article.get('headline_ar', '')}\n\n{article.get('website_url', '')}\n\n#ساي_تك #SciTech #أخبار_التكنولوجيا",
        )

        tg_caption = f"🎬 <b>ريل جديد — {article.get('category', '')}</b>\n\n{script.get('headline', '')}\n\n🔗 {article.get('website_url', '')}"
        tg_ok = send_reel_to_telegram(video_path, tg_caption)

        mark_reel_posted(article["article_id"], fb_video_id, tg_ok)

        if fb_video_id:
            log.info(f"[reels] ✓ Reel posted to Facebook (video_id={fb_video_id})")
        else:
            log.warning("[reels] ✗ Facebook post failed")
        if tg_ok:
            log.info("[reels] ✓ Reel sent to Telegram")

        if i < len(candidates) - 1:
            time.sleep(10)

    cleanup_old_files()
    log.info("\n[reels] all done!")


if __name__ == "__main__":
    run()
