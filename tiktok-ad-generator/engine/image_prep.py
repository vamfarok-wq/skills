"""
Turns the uploaded product photo into a polished scene background:

  * a soft brand-coloured gradient
  * a big blurred copy of the product filling the frame (premium "studio" look)
  * the product itself, neatly fitted, with rounded corners and a soft shadow

We render this slightly LARGER than the video frame so the Ken Burns zoom/pan
in render.py always has room to move without showing empty edges.
"""

from PIL import Image, ImageDraw, ImageFilter, ImageOps

import config

# Oversized canvas so the zoom never runs off the edge.
OVERSCAN = 1.18
BW = int(config.WIDTH * OVERSCAN)
BH = int(config.HEIGHT * OVERSCAN)


def load_product(path, max_side=1400):
    """Load the product image, fix orientation, flatten transparency to white."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def _gradient(size, top, bottom):
    w, h = size
    base = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(1, h - 1)
        base.putpixel((0, y), tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    return base.resize((w, h))


def _cover(img, size):
    """Resize-crop img to completely cover `size` (like CSS background-size:cover)."""
    w, h = size
    iw, ih = img.size
    scale = max(w / iw, h / ih)
    img = img.resize((int(iw * scale) + 1, int(ih * scale) + 1), Image.LANCZOS)
    iw, ih = img.size
    left, top = (iw - w) // 2, (ih - h) // 2
    return img.crop((left, top, left + w, top + h))


def _rounded(img, radius):
    """Return img (RGBA) with rounded corners."""
    img = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.size[0], img.size[1]],
                                           radius=radius, fill=255)
    img.putalpha(mask)
    return img


def build_scene_base(product_path, palette):
    """
    Build the oversized (BW x BH) RGB background used for every scene.
    Returns a PIL.Image in RGB mode.
    """
    product = load_product(product_path)

    # 1) brand gradient
    canvas = _gradient((BW, BH), palette["bg_top"], palette["bg_bottom"]).convert("RGB")

    # 2) big blurred product fill for depth
    blurred = _cover(product, (BW, BH)).filter(ImageFilter.GaussianBlur(38))
    # blend the blur with the gradient so the brand colour still shows through
    canvas = Image.blend(canvas, blurred, 0.55)
    # gentle vignette / brighten centre by overlaying a soft white radial
    glow = Image.new("L", (BW, BH), 0)
    gd = ImageDraw.Draw(glow)
    gd.ellipse([BW * 0.06, BH * 0.04, BW * 0.94, BH * 0.78], fill=70)
    glow = glow.filter(ImageFilter.GaussianBlur(160))
    white = Image.new("RGB", (BW, BH), (255, 255, 255))
    canvas = Image.composite(white, canvas, glow)

    # 3) hero product card, centred in the upper-middle area
    target_w = int(BW * 0.70)
    pw, ph = product.size
    scale = target_w / pw
    hero_w, hero_h = int(pw * scale), int(ph * scale)
    # keep it from getting too tall
    max_h = int(BH * 0.52)
    if hero_h > max_h:
        scale = max_h / ph
        hero_w, hero_h = int(pw * scale), int(ph * scale)
    hero = product.resize((hero_w, hero_h), Image.LANCZOS)
    hero = _rounded(hero, radius=int(min(hero_w, hero_h) * 0.07))

    cx = BW // 2
    cy = int(BH * 0.40)
    hx, hy = cx - hero_w // 2, cy - hero_h // 2

    # soft drop shadow
    shadow = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    sd = Image.new("RGBA", (hero_w, hero_h), (0, 0, 0, 0))
    ImageDraw.Draw(sd).rounded_rectangle([0, 0, hero_w, hero_h],
                                         radius=int(min(hero_w, hero_h) * 0.07),
                                         fill=(0, 0, 0, 120))
    shadow.paste(sd, (hx, hy + 22), sd)
    shadow = shadow.filter(ImageFilter.GaussianBlur(28))

    out = canvas.convert("RGBA")
    out = Image.alpha_composite(out, shadow)
    out.paste(hero, (hx, hy), hero)
    return out.convert("RGB")
