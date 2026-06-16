"""
Font loading.

The nice fonts (Poppins + Anton) are bundled in assets/fonts so the app works
offline. If for some reason they're missing, we fall back to common system
fonts, and finally to Pillow's built-in font so the app never crashes.
"""

import os
from functools import lru_cache

from PIL import ImageFont

import config

# role -> bundled file name (in assets/fonts)
_BUNDLED = {
    "hook": "Anton-Regular.ttf",            # big punchy headline
    "headline": "Poppins-ExtraBold.ttf",    # main captions
    "sub": "Poppins-SemiBold.ttf",          # secondary lines
    "body": "Poppins-Medium.ttf",           # small text / handle
}

# Fallback system fonts to try (Windows, macOS, Linux) per role.
_SYSTEM = {
    "hook": [
        "C:/Windows/Fonts/impact.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "headline": [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/seguibl.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "sub": [
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "body": [
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
}


def _resolve_path(role):
    bundled = os.path.join(config.FONT_DIR, _BUNDLED.get(role, _BUNDLED["headline"]))
    if os.path.exists(bundled):
        return bundled
    for p in _SYSTEM.get(role, _SYSTEM["headline"]):
        if os.path.exists(p):
            return p
    return None


@lru_cache(maxsize=128)
def get_font(role, size):
    """Return a PIL ImageFont for the given role ('hook','headline','sub','body')."""
    path = _resolve_path(role)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    # absolute last resort
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()
