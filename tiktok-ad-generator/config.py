"""
Central configuration for the TikTok Shop Ad Video Generator.

Everything you might want to tweak (video size, colors per product type,
voices, tones) lives here so you don't have to dig through the code.
"""

import os

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FONT_DIR = os.path.join(ASSETS_DIR, "fonts")
MUSIC_DIR = os.path.join(ASSETS_DIR, "music")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
UPLOAD_DIR = os.path.join(BASE_DIR, "output", "_uploads")
TMP_DIR = os.path.join(BASE_DIR, "output", "_tmp")

for _d in (OUTPUT_DIR, UPLOAD_DIR, TMP_DIR, MUSIC_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Video format  (9:16 vertical, the TikTok standard)
# ---------------------------------------------------------------------------
WIDTH = 1080
HEIGHT = 1920
FPS = 30

# How long (seconds) each spoken line is padded by, and the min/max a scene
# can last. Keeps the final video in the 15-35s "sweet spot" for TikTok Shop.
SCENE_PAD = 0.5
SCENE_MIN = 2.3
SCENE_MAX = 6.0

# Quick crossfade between scenes (frames). Set to 0 for hard cuts.
CROSSFADE_FRAMES = 7

# ---------------------------------------------------------------------------
# Colour palettes per product type.
#   bg_top / bg_bottom -> background gradient
#   accent             -> highlight colour (underline, stars, progress bar)
# Most of these lean soft / feminine because the audience is mostly women.
# ---------------------------------------------------------------------------
PALETTES = {
    "cosmetics":  {"bg_top": (255, 224, 233), "bg_bottom": (255, 190, 211), "accent": (233, 30, 99)},
    "skincare":   {"bg_top": (255, 235, 224), "bg_bottom": (255, 206, 193), "accent": (255, 111, 97)},
    "makeup":     {"bg_top": (246, 224, 236), "bg_bottom": (220, 160, 193), "accent": (194, 71, 155)},
    "haircare":   {"bg_top": (237, 230, 248), "bg_bottom": (201, 186, 234), "accent": (126, 87, 194)},
    "fashion":    {"bg_top": (246, 233, 227), "bg_bottom": (222, 199, 189), "accent": (191, 116, 96)},
    "jewelry":    {"bg_top": (247, 240, 226), "bg_bottom": (226, 206, 167), "accent": (190, 150, 70)},
    "home":       {"bg_top": (236, 241, 233), "bg_bottom": (203, 220, 198), "accent": (96, 150, 104)},
    "wellness":   {"bg_top": (224, 244, 240), "bg_bottom": (183, 223, 216), "accent": (38, 166, 154)},
    "baby":       {"bg_top": (247, 241, 224), "bg_bottom": (211, 226, 241), "accent": (255, 167, 38)},
    "general":    {"bg_top": (240, 230, 246), "bg_bottom": (208, 193, 231), "accent": (171, 71, 188)},
}

# Friendly labels shown in the web app, mapped to palette keys above.
NICHES = [
    ("cosmetics", "Cosmetics / Beauty"),
    ("skincare", "Skincare"),
    ("makeup", "Makeup"),
    ("haircare", "Hair care"),
    ("fashion", "Fashion / Clothing"),
    ("jewelry", "Jewelry / Accessories"),
    ("home", "Home / Kitchen"),
    ("wellness", "Health / Wellness"),
    ("baby", "Baby / Kids"),
    ("general", "Other / General"),
]

# ---------------------------------------------------------------------------
# Voices (free, from Microsoft Edge TTS — no API key needed).
# Mostly warm female voices because the audience is mostly women.
# ---------------------------------------------------------------------------
VOICES = [
    ("en-US-JennyNeural", "Jenny — US female, warm & friendly"),
    ("en-US-AriaNeural", "Aria — US female, upbeat"),
    ("en-US-MichelleNeural", "Michelle — US female, soft"),
    ("en-US-AnaNeural", "Ana — US female, youthful"),
    ("en-GB-SoniaNeural", "Sonia — UK female, elegant"),
    ("en-GB-LibbyNeural", "Libby — UK female, young"),
    ("en-AU-NatashaNeural", "Natasha — AU female, bright"),
    ("en-US-GuyNeural", "Guy — US male"),
]
DEFAULT_VOICE = "en-US-JennyNeural"

# Tone -> how the voice is delivered (Edge TTS rate/pitch) + copy style.
TONES = {
    "energetic": {"label": "Energetic / Hype", "rate": "+12%", "pitch": "+10Hz"},
    "friendly":  {"label": "Friendly / Casual", "rate": "+4%", "pitch": "+2Hz"},
    "luxury":    {"label": "Luxury / Premium", "rate": "-6%", "pitch": "-2Hz"},
    "calm":      {"label": "Calm / Soft", "rate": "-8%", "pitch": "0Hz"},
}
DEFAULT_TONE = "friendly"

# Video styles offered in the UI.
STYLES = [
    ("animation", "Dynamic Animation (recommended)"),
    ("hype", "Bold & Hype (max motion + sparkles)"),
    ("showcase", "Clean Showcase (slower, premium)"),
    ("human", "AI Human Presenter (needs paid add-on)"),
]
DEFAULT_STYLE = "animation"
