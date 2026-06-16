"""
Video rendering.

Takes the prepared scene background + the script, then for every frame:
  * applies a gentle Ken Burns zoom/pan to the background (so it feels alive)
  * pops an animated caption card on top, with a highlighted keyword
  * sprinkles soft floating sparkles (beauty vibe) and an animated CTA arrow
  * draws a top progress bar and optional @handle watermark
  * crossfades between scenes and fades in/out at the very start/end

Frames are streamed straight into ffmpeg (no giant temp files) and encoded to
an H.264 mp4 that TikTok accepts directly.
"""

import math
import random
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw

import config
from engine.assets import get_font

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = config.WIDTH, config.HEIGHT, config.FPS
BLACK = Image.new("RGB", (W, H), (0, 0, 0))
_SCRATCH = ImageDraw.Draw(Image.new("RGB", (10, 10)))

# Words worth highlighting (drawn with an accent background, TikTok-style).
POWER = {"free", "glow", "glowing", "new", "now", "viral", "sale", "obsessed",
         "best", "love", "sold", "off", "secret", "instantly", "amazing", "game",
         "changer", "results", "stock", "gift", "perfect", "soft", "trending",
         "magic", "dreamy", "luxury", "must", "wow", "today", "sells", "regret"}


# ---------------------------------------------------------------------------
# easing
# ---------------------------------------------------------------------------
def _smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _ease_out_back(x):
    x = max(0.0, min(1.0, x))
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * pow(x - 1, 3) + c1 * pow(x - 1, 2)


def _clean(s):
    return "".join(c for c in s.lower() if c.isalnum())


# ---------------------------------------------------------------------------
# sparkle sprite (built once)
# ---------------------------------------------------------------------------
def _make_sparkle(size=52):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = size / 2
    for i in range(14):                       # soft glow
        rr = c * (i + 1) / 14
        a = int(46 * (1 - i / 14))
        d.ellipse([c - rr, c - rr, c + rr, c + rr], fill=(255, 255, 255, a))
    d.polygon([(c, 3), (c + 5, c), (c, size - 3), (c - 5, c)], fill=(255, 255, 255, 235))
    d.polygon([(3, c), (c, c + 5), (size - 3, c), (c, c - 5)], fill=(255, 255, 255, 235))
    return img


_SPARKLE = _make_sparkle()


def _sparkles(frame, t, n, seed):
    for i in range(n):
        rnd = random.Random(seed * 97 + i)
        bx, by = rnd.uniform(0.06, 0.94), rnd.uniform(0.08, 0.92)
        speed, phase = rnd.uniform(16, 52), rnd.uniform(0, 6.283)
        base = rnd.uniform(0.5, 1.0)
        y = (by * H - t * speed) % H
        x = bx * W + 14 * math.sin(t * 0.8 + phase)
        tw = 0.40 + 0.60 * (0.5 + 0.5 * math.sin(t * 3.0 + phase))
        sz = int(_SPARKLE.size[0] * (0.45 + 1.25 * base * (0.5 + 0.5 * tw)))
        if sz < 7:
            continue
        spr = _SPARKLE.resize((sz, sz), Image.BILINEAR)
        a = spr.split()[3].point(lambda v: int(v * tw))
        frame.paste(spr.convert("RGB"), (int(x - sz / 2), int(y - sz / 2)), a)


def _cta_arrow(frame, t, palette):
    d = ImageDraw.Draw(frame)
    cx = W // 2
    y = int(H * 0.82) + int(13 * math.sin(t * 6))
    acc = palette["accent"]
    for col, w, off in (((0, 0, 0), 22, 3), (acc, 14, 0)):   # dark stroke then accent
        d.line([(cx - 46, y - 22 + off), (cx, y + 18 + off)], fill=col, width=w)
        d.line([(cx + 46, y - 22 + off), (cx, y + 18 + off)], fill=col, width=w)


# ---------------------------------------------------------------------------
# caption card  (word layout + keyword highlight)
# ---------------------------------------------------------------------------
_KIND = {
    "hook":    {"size": 78, "cy": 0.71, "bar": True,  "pop": 0.32, "long": True},
    "problem": {"size": 60, "cy": 0.72, "bar": False, "pop": 0.22, "long": False},
    "reveal":  {"size": 72, "cy": 0.71, "bar": True,  "pop": 0.40, "long": True},
    "benefit": {"size": 62, "cy": 0.72, "bar": False, "pop": 0.22, "long": False},
    "proof":   {"size": 60, "cy": 0.71, "bar": False, "pop": 0.26, "long": False, "stars": True},
    "cta":     {"size": 68, "cy": 0.71, "bar": True,  "pop": 0.40, "long": True},
}


def make_caption_card(scene, palette):
    """Render a caption to an RGBA card. Returns (card, (cx, cy), pop_strength)."""
    kind = scene.get("kind", "benefit")
    spec = _KIND.get(kind, _KIND["benefit"])
    text = (scene.get("caption") or "").strip()
    anchor = (W // 2, int(H * spec["cy"]))
    if not text:
        return None, anchor, spec["pop"]

    font = get_font("headline", spec["size"])
    pad_x, pad_y = 46, 32
    max_text_w = 940 - 2 * pad_x
    space_w = _SCRATCH.textlength(" ", font=font)
    asc, desc = font.getmetrics()
    line_h = asc + desc
    gap = int(line_h * 0.14)

    tokens = text.split()
    # choose a keyword to highlight
    hi = -1
    for i, tok in enumerate(tokens):
        if _clean(tok) in POWER:
            hi = i
            break
    if hi < 0 and spec.get("long"):
        cand = [(len(_clean(t)), i) for i, t in enumerate(tokens) if len(_clean(t)) > 3]
        if cand:
            hi = max(cand)[1]

    # wrap into lines of (token, is_highlight, width)
    lines, cur, curw = [], [], 0.0
    for i, tok in enumerate(tokens):
        w = _SCRATCH.textlength(tok, font=font)
        add = w if not cur else space_w + w
        if cur and curw + add > max_text_w:
            lines.append((cur, curw))
            cur, curw = [], 0.0
            add = w
        cur.append((tok, i == hi, w))
        curw += add
    if cur:
        lines.append((cur, curw))

    text_w = max(w for _, w in lines)
    star_h = 70 if spec.get("stars") else 0
    card_w = int(min(W - 56, text_w + 2 * pad_x))
    card_h = int(line_h * len(lines) + gap * (len(lines) - 1) + 2 * pad_y + star_h)

    card = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle([0, 0, card_w, card_h], radius=34, fill=(15, 15, 20, 170))

    y = pad_y
    if spec.get("stars"):
        total = 5 * 56 + 4 * 10
        sx = (card_w - total) // 2 + 28
        for i in range(5):
            _star(d, sx + i * 66, y + 28, 30, palette["accent"])
        y += star_h

    acc = palette["accent"]
    for line, lw in lines:
        x = (card_w - lw) / 2
        for tok, is_hl, w in line:
            if is_hl:
                d.rounded_rectangle([x - 9, y + 6, x + w + 9, y + line_h - 2],
                                    radius=13, fill=acc + (255,))
                d.text((x, y), tok, font=font, fill=(255, 255, 255, 255),
                       stroke_width=1, stroke_fill=acc + (255,))
            else:
                d.text((x, y), tok, font=font, fill=(255, 255, 255, 255),
                       stroke_width=2, stroke_fill=(0, 0, 0, 230))
            x += w + space_w
        y += line_h + gap

    if spec.get("bar"):
        bw = int(card_w * 0.30)
        bx = (card_w - bw) // 2
        d.rounded_rectangle([bx, card_h - 15, bx + bw, card_h - 7], radius=4,
                            fill=acc + (255,))
    return card, anchor, spec["pop"]


def _star(draw, cx, cy, r, fill):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.42
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    draw.polygon(pts, fill=fill)


# ---------------------------------------------------------------------------
# Ken Burns
# ---------------------------------------------------------------------------
_ZOOM = {"small": (1.04, 1.12, 0.18), "med": (1.02, 1.16, 0.5), "big": (1.0, 1.2, 0.7)}


def _kb_params(scene_index, zoom):
    lo, hi, pan = _ZOOM.get(zoom, _ZOOM["med"])
    zoom_in = (scene_index % 2 == 0)
    k0, k1 = (hi, lo) if zoom_in else (lo, hi)
    px0, px1 = (-pan, pan) if scene_index % 2 == 0 else (pan, -pan)
    py0, py1 = (pan * 0.4, -pan * 0.4)
    return k0, k1, px0, px1, py0, py1


def _ken_burns(base, p, params):
    bw, bh = base.size
    k0, k1, px0, px1, py0, py1 = params
    e = _smoothstep(p)
    k = k0 + (k1 - k0) * e
    cw, ch = min(bw, W * k), min(bh, H * k)
    mx, my = (bw - cw) / 2, (bh - ch) / 2
    cx = bw / 2 + (px0 + (px1 - px0) * e) * mx
    cy = bh / 2 + (py0 + (py1 - py0) * e) * my
    left = max(0, min(bw - cw, cx - cw / 2))
    top = max(0, min(bh - ch, cy - ch / 2))
    crop = base.crop((int(left), int(top), int(left + cw), int(top + ch)))
    return crop.resize((W, H), Image.BILINEAR)


def _paste_card(frame, card, anchor, alpha, scale, dy):
    if card is None or alpha <= 0.01:
        return
    cw, ch = card.size
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    c = card.resize((nw, nh), Image.BILINEAR)
    if alpha < 1.0:
        a = c.split()[3].point(lambda v: int(v * alpha))
        c.putalpha(a)
    x = anchor[0] - nw // 2
    y = anchor[1] - nh // 2 + dy
    frame.paste(c.convert("RGB"), (x, y), c.split()[3])


def _overlay(frame, global_p, palette, handle):
    d = ImageDraw.Draw(frame)
    margin, top, th = 40, 64, 10
    d.rounded_rectangle([margin, top, W - margin, top + th], radius=5, fill=(255, 255, 255))
    fill_w = int((W - 2 * margin) * max(0.0, min(1.0, global_p)))
    if fill_w > th:
        d.rounded_rectangle([margin, top, margin + fill_w, top + th], radius=5,
                            fill=palette["accent"])
    if handle:
        h = handle if handle.startswith("@") else "@" + handle
        f = get_font("sub", 36)
        d.text((margin, top + 26), h, font=f, fill=(255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
_STYLE_FX = {
    "showcase":  {"zoom": "small", "sparkles": 0,  "cf": 9, "pop_mul": 0.85},
    "animation": {"zoom": "med",   "sparkles": 7,  "cf": 7, "pop_mul": 1.0},
    "hype":      {"zoom": "big",   "sparkles": 11, "cf": 5, "pop_mul": 1.3},
}


def render_video(base_img, scenes, cards, durations, out_path,
                 style="animation", handle="", palette=None, progress_cb=None):
    palette = palette or config.PALETTES["general"]
    if style == "human":          # paid avatar not configured -> animated look
        style = "animation"
    fx = _STYLE_FX.get(style, _STYLE_FX["animation"])

    scene_frames = [max(1, int(round(d * FPS))) for d in durations]
    total = sum(scene_frames)
    gfade = max(1, int(FPS * 0.3))
    seed = (sum(ord(c) for c in (handle or "x")) + len(scenes)) % 9999

    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           "-crf", "20", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    prev_last = None
    gi = 0
    try:
        for si, scene in enumerate(scenes):
            sf = scene_frames[si]
            params = _kb_params(si, fx["zoom"])
            card, anchor, pop = cards[si]
            pop = pop * fx["pop_mul"]
            xfade = min(fx["cf"], sf // 2) if si > 0 else 0
            is_cta = scene.get("kind") == "cta"

            for f in range(sf):
                p = f / max(1, sf - 1)
                t = gi / FPS
                frame = _ken_burns(base_img, p, params)

                if fx["sparkles"]:
                    _sparkles(frame, t, fx["sparkles"], seed)

                # Caption holds off until the crossfade finishes, so the old and
                # new captions never overlap during a transition.
                local_t = f / FPS
                if f >= xfade:
                    lct = (f - xfade) / FPS
                    prog = min(1.0, lct / 0.42)
                    s = _ease_out_back(prog)
                    scale = (1 - pop) + pop * s
                    alpha = _smoothstep(min(1.0, lct / 0.30))
                    dy = int(34 * (1 - _smoothstep(prog)))
                    _paste_card(frame, card, anchor, alpha, scale, dy)

                _overlay(frame, gi / max(1, total - 1), palette, handle)
                if is_cta:
                    _cta_arrow(frame, local_t, palette)

                if xfade and f < xfade and prev_last is not None:
                    frame = Image.blend(prev_last, frame, (f + 1) / (xfade + 1))
                if gi < gfade:
                    frame = Image.blend(BLACK, frame, _smoothstep(gi / gfade))
                elif gi >= total - gfade:
                    frame = Image.blend(BLACK, frame, _smoothstep((total - 1 - gi) / gfade))

                proc.stdin.write(frame.tobytes())
                prev_last = frame
                gi += 1
                if progress_cb and gi % 10 == 0:
                    progress_cb(gi / total)
    finally:
        proc.stdin.close()
        proc.wait()
    if progress_cb:
        progress_cb(1.0)
    return out_path
