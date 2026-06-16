"""
Video rendering.

Takes the prepared scene background + the script, then for every frame:
  * applies a gentle Ken Burns zoom/pan to the background (so it feels alive)
  * pops an animated caption card on top
  * draws a top progress bar and optional @handle watermark
  * crossfades between scenes and fades in/out at the very start/end

Frames are streamed straight into ffmpeg (no giant temp files) and encoded to
an H.264 mp4 that TikTok accepts directly.
"""

import math
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw

import config
from engine.assets import get_font

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = config.WIDTH, config.HEIGHT, config.FPS
BLACK = Image.new("RGB", (W, H), (0, 0, 0))
_SCRATCH = ImageDraw.Draw(Image.new("RGB", (10, 10)))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _wrap(text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if _SCRATCH.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _star(draw, cx, cy, r, fill):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.42
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    draw.polygon(pts, fill=fill)


# Per-scene caption styling.
_KIND = {
    "hook":    {"role": "headline", "size": 78, "cy": 0.71, "accent_bar": True},
    "problem": {"role": "headline", "size": 60, "cy": 0.72, "accent_bar": False},
    "reveal":  {"role": "headline", "size": 72, "cy": 0.71, "accent_bar": True},
    "benefit": {"role": "headline", "size": 62, "cy": 0.72, "accent_bar": False},
    "proof":   {"role": "headline", "size": 60, "cy": 0.71, "accent_bar": False, "stars": True},
    "cta":     {"role": "headline", "size": 68, "cy": 0.71, "accent_bar": True},
}


def make_caption_card(scene, palette):
    """Render a caption to an RGBA card. Returns (card_image, (anchor_cx, anchor_cy))."""
    kind = scene.get("kind", "benefit")
    spec = _KIND.get(kind, _KIND["benefit"])
    text = (scene.get("caption") or "").strip()
    if not text:
        return None, (W // 2, int(H * spec["cy"]))

    font = get_font(spec["role"], spec["size"])
    pad_x, pad_y = 46, 34
    max_text_w = 940 - 2 * pad_x
    lines = _wrap(text, font, max_text_w)

    asc, desc = font.getmetrics()
    line_h = asc + desc
    gap = int(line_h * 0.16)
    text_w = max(int(_SCRATCH.textlength(ln, font=font)) for ln in lines)
    text_h = line_h * len(lines) + gap * (len(lines) - 1)

    star_h = 70 if spec.get("stars") else 0
    card_w = min(W - 60, text_w + 2 * pad_x)
    card_h = text_h + 2 * pad_y + star_h

    card = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    # frosted dark box for guaranteed readability over any photo
    d.rounded_rectangle([0, 0, card_w, card_h], radius=34, fill=(15, 15, 20, 165))

    y = pad_y
    if spec.get("stars"):
        sw = 56
        total = 5 * sw + 4 * 10
        sx = (card_w - total) // 2 + sw // 2
        for i in range(5):
            _star(d, sx + i * (sw + 10), y + 28, 30, palette["accent"])
        y += star_h

    for ln in lines:
        lw = _SCRATCH.textlength(ln, font=font)
        d.text(((card_w - lw) // 2, y), ln, font=font, fill=(255, 255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0, 230))
        y += line_h + gap

    if spec.get("accent_bar"):
        bar_w = int(card_w * 0.32)
        bx = (card_w - bar_w) // 2
        d.rounded_rectangle([bx, card_h - 16, bx + bar_w, card_h - 8],
                            radius=4, fill=palette["accent"] + (255,))

    return card, (W // 2, int(H * spec["cy"]))


# ---------------------------------------------------------------------------
# Ken Burns
# ---------------------------------------------------------------------------
def _kb_params(scene_index, style):
    zoom_in = (scene_index % 2 == 0)
    if style == "showcase":
        lo, hi, pan = 1.04, 1.12, 0.18
    else:
        lo, hi, pan = 1.02, 1.16, 0.5
    (k0, k1) = (hi, lo) if zoom_in else (lo, hi)
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
    # top progress bar
    margin, top, th = 40, 64, 10
    d.rounded_rectangle([margin, top, W - margin, top + th], radius=5, fill=(255, 255, 255, 90))
    fill_w = int((W - 2 * margin) * max(0.0, min(1.0, global_p)))
    if fill_w > th:
        d.rounded_rectangle([margin, top, margin + fill_w, top + th], radius=5,
                            fill=palette["accent"])
    # optional @handle
    if handle:
        h = handle if handle.startswith("@") else "@" + handle
        f = get_font("sub", 36)
        d.text((margin, top + 26), h, font=f, fill=(255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def render_video(base_img, scenes, cards, durations, out_path,
                 style="animation", handle="", palette=None, progress_cb=None):
    palette = palette or config.PALETTES["general"]
    scene_frames = [max(1, int(round(d * FPS))) for d in durations]
    total = sum(scene_frames)
    cf = config.CROSSFADE_FRAMES
    gfade = max(1, int(FPS * 0.3))

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
            params = _kb_params(si, style)
            card, anchor = cards[si]
            xfade = min(cf, sf // 2) if si > 0 else 0

            for f in range(sf):
                p = f / max(1, sf - 1)
                frame = _ken_burns(base_img, p, params)

                # caption pop-in
                local_t = f / FPS
                e = _smoothstep(min(1.0, local_t / 0.35))
                _paste_card(frame, card, anchor, alpha=e,
                            scale=0.88 + 0.12 * e, dy=int(26 * (1 - e)))

                _overlay(frame, gi / max(1, total - 1), palette, handle)

                # crossfade from previous scene
                if xfade and f < xfade and prev_last is not None:
                    frame = Image.blend(prev_last, frame, (f + 1) / (xfade + 1))

                # global fade in / out
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
