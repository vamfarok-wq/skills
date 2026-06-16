"""
Copywriting: turns a product name + category into a short, punchy TikTok Shop
ad script (hook -> problem -> reveal -> benefits -> social proof -> call to action).

Two ways to generate copy:
  1. Templates (default, 100% free, no internet, no API key). Lots of variety so
     your videos don't all sound the same.
  2. Optional AI (only if you set an ANTHROPIC_API_KEY) which can also *look at
     the product photo* to write tailored copy. Completely optional.

Whatever is generated, you can edit every line in the web app before rendering,
so the final words are always yours.
"""

import os
import random
import base64

# ---------------------------------------------------------------------------
# Template word banks
# ---------------------------------------------------------------------------
HOOKS = [
    "Stop scrolling — you NEED to see this.",
    "POV: you finally found the one that actually works.",
    "I was today years old when I found this.",
    "This is your sign to finally try it.",
    "Why is nobody talking about this?!",
    "TikTok made me buy it… and I'm obsessed.",
    "Okay, this might be my best find ever.",
    "If you've been thinking about it — this is your sign.",
    "I can't gatekeep this any longer.",
    "Run, don't walk, to grab this one.",
]

PROBLEMS = {
    "cosmetics": ["Tired of products that promise everything and do nothing?",
                  "Sick of wasting money on stuff that just sits in a drawer?"],
    "skincare":  ["Tired of dull, tired-looking skin?",
                  "Struggling with dryness no matter what you try?"],
    "makeup":    ["Tired of makeup that fades by lunchtime?",
                  "Can't find a shade that actually suits you?"],
    "haircare":  ["Dealing with dry, frizzy, lifeless hair?",
                  "Tired of bad hair days you can't fix?"],
    "fashion":   ["Nothing in your closet feels right lately?",
                  "Want to look put-together without the effort?"],
    "jewelry":   ["Want a little everyday sparkle that lasts?",
                  "Tired of jewelry that turns your skin green?"],
    "home":      ["Tired of clutter and things that just don't work?",
                  "Wish your daily routine felt a little easier?"],
    "wellness":  ["Struggling to keep up with your wellness goals?",
                  "Tired of feeling low on energy every day?"],
    "baby":      ["Looking for something safe and gentle for your little one?",
                  "Tired of products that are more hassle than help?"],
    "general":   ["Tired of products that just don't deliver?",
                  "Wish you'd found this sooner?"],
}

REVEALS = [
    "Meet {name}.",
    "This is {name} — and it's a total game changer.",
    "Say hello to {name}.",
    "{name} is about to change your routine.",
    "Let me introduce you to {name}.",
]

BENEFITS = {
    "cosmetics": ["Made with gentle, skin-loving ingredients.",
                  "A little goes a long way, so it lasts for ages.",
                  "Cruelty-free and kind to sensitive skin.",
                  "Feels luxurious without the luxury price tag."],
    "skincare":  ["Hydrates deeply for a healthy, dewy glow.",
                  "Smooths and softens skin in days, not weeks.",
                  "Lightweight, non-greasy, absorbs in seconds.",
                  "Gentle enough for sensitive skin, every day."],
    "makeup":    ["Long-lasting, all-day stay-put formula.",
                  "Buildable color that looks natural or bold.",
                  "Blends like a dream — no patchiness.",
                  "Flattering on every skin tone."],
    "haircare":  ["Leaves hair soft, shiny, and frizz-free.",
                  "Nourishes from root to tip.",
                  "Lightweight — no greasy, weighed-down feeling.",
                  "Smells amazing and lasts all day."],
    "fashion":   ["Flattering fit that hugs in all the right places.",
                  "Soft, comfy fabric you'll live in.",
                  "Dress it up or down — endlessly versatile.",
                  "Looks way more expensive than it is."],
    "jewelry":   ["Tarnish-free and skin-friendly.",
                  "Dainty and elegant for everyday wear.",
                  "The perfect finishing touch to any outfit.",
                  "Looks high-end without the high-end price."],
    "home":      ["Saves you time every single day.",
                  "Simple, smart, and actually makes life easier.",
                  "Sturdy, well-made, and built to last.",
                  "The little upgrade you didn't know you needed."],
    "wellness":  ["Supports your daily wellness routine.",
                  "Easy to fit into even the busiest day.",
                  "Feel-good results you'll actually notice.",
                  "Thoughtfully made with quality you can trust."],
    "baby":      ["Gentle and safe for delicate skin.",
                  "Makes everyday parenting just a little easier.",
                  "Soft, thoughtful, and beautifully made.",
                  "Loved by parents and little ones alike."],
    "general":   ["Thoughtfully designed to actually work.",
                  "Great quality that lasts and lasts.",
                  "Simple to use and genuinely useful.",
                  "Amazing value for what you get."],
}

PROOF = [
    "Thousands sold with glowing 5-star reviews.",
    "It's going viral for a reason.",
    "So many people are switching to this.",
    "Selling out fast — and it keeps coming back.",
    "The reviews honestly speak for themselves.",
]

CTAS = [
    "Tap the orange cart and grab yours before it sells out!",
    "Hit the yellow basket and treat yourself today.",
    "Tap the link below — you won't regret it.",
    "Get yours now while it's still in stock!",
    "Add it to cart and thank yourself later.",
]

HASHTAGS = {
    "cosmetics": ["#tiktokmademebuyit", "#beautytok", "#cosmetics", "#tiktokshopfinds", "#beautyfinds"],
    "skincare":  ["#skincare", "#skincareroutine", "#glowup", "#tiktokmademebuyit", "#skintok"],
    "makeup":    ["#makeup", "#makeuptutorial", "#beautytok", "#tiktokmademebuyit", "#makeuphacks"],
    "haircare":  ["#haircare", "#hairtok", "#hairgoals", "#tiktokmademebuyit", "#healthyhair"],
    "fashion":   ["#fashiontok", "#outfitinspo", "#tiktokfashion", "#tiktokmademebuyit", "#ootd"],
    "jewelry":   ["#jewelry", "#accessories", "#daintyjewelry", "#tiktokmademebuyit", "#jewelrytok"],
    "home":      ["#homefinds", "#tiktokmademebuyit", "#kitchengadgets", "#homehacks", "#amazonfinds"],
    "wellness":  ["#wellness", "#selfcare", "#healthtok", "#tiktokmademebuyit", "#wellnessjourney"],
    "baby":      ["#momtok", "#babyessentials", "#momsoftiktok", "#tiktokmademebuyit", "#parentinghacks"],
    "general":   ["#tiktokmademebuyit", "#tiktokshopfinds", "#musthaves", "#viralproducts", "#tiktokfinds"],
}

# Emoji accents for the *post caption* only (not burned into the video, so they
# always render correctly when you paste them into TikTok).
EMOJI = {
    "cosmetics": "💄✨", "skincare": "🧴✨", "makeup": "💅💖", "haircare": "💇‍♀️✨",
    "fashion": "👗✨", "jewelry": "💍✨", "home": "🏠✨", "wellness": "🌿✨",
    "baby": "🍼💕", "general": "🛒✨",
}


def _pick(seq, rng):
    return seq[rng.randrange(len(seq))]


def generate_script(product_name, niche="general", tone="friendly",
                    details="", benefits=None, handle="", price="", seed=None):
    """
    Build the scene-by-scene ad script.

    Returns a dict:
      {
        "scenes": [ {kind, voice, caption}, ... ],
        "tiktok_caption": "...",
        "hashtags": [...],
      }
    `voice` is what gets spoken; `caption` is the (shorter) on-screen text.
    """
    rng = random.Random(seed)
    niche = niche if niche in BENEFITS else "general"
    name = (product_name or "this product").strip()

    # Benefit lines: prefer user-provided, else pick 3 distinct from the bank.
    if benefits:
        chosen_benefits = [b.strip() for b in benefits if b and b.strip()][:3]
    else:
        pool = BENEFITS[niche][:]
        rng.shuffle(pool)
        chosen_benefits = pool[:3]
    if details.strip():
        # Drop the user's own description in as the first benefit line.
        chosen_benefits = [details.strip()] + chosen_benefits
        chosen_benefits = chosen_benefits[:3]

    price_line = ""
    if price.strip():
        price_line = f" And it's only {price.strip()}."

    # On-screen caption == spoken line (what you edit is what you see & hear).
    scenes = []
    hook = _pick(HOOKS, rng)
    scenes.append({"kind": "hook", "voice": hook, "caption": hook})
    problem = _pick(PROBLEMS[niche], rng)
    scenes.append({"kind": "problem", "voice": problem, "caption": problem})
    reveal = _pick(REVEALS, rng).format(name=name)
    scenes.append({"kind": "reveal", "voice": reveal + price_line, "caption": reveal})
    for b in chosen_benefits:
        scenes.append({"kind": "benefit", "voice": b, "caption": b})
    proof = _pick(PROOF, rng)
    scenes.append({"kind": "proof", "voice": proof, "caption": proof})
    cta = _pick(CTAS, rng)
    scenes.append({"kind": "cta", "voice": cta, "caption": cta})

    tags = HASHTAGS.get(niche, HASHTAGS["general"])
    emoji = EMOJI.get(niche, EMOJI["general"])
    caption_bits = [f"{name} {emoji}"]
    if details.strip():
        caption_bits.append(details.strip())
    else:
        caption_bits.append(chosen_benefits[0])
    caption_bits.append(_pick(CTAS, rng))
    if handle.strip():
        caption_bits.append(f"Follow {handle.strip()} for more!")
    tiktok_caption = "  ".join(caption_bits) + "\n\n" + " ".join(tags)

    return {"scenes": scenes, "tiktok_caption": tiktok_caption, "hashtags": tags}


# ---------------------------------------------------------------------------
# Optional AI copywriting (only used if ANTHROPIC_API_KEY is set).
# Looks at the product photo + name and drafts a tailored script.
# If anything fails, we silently fall back to the templates above.
# ---------------------------------------------------------------------------
def generate_with_ai(image_path, product_name, niche, tone, handle="", price=""):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except Exception:
        return None
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.standard_b64encode(f.read()).decode()
        media = "image/png"
        low = image_path.lower()
        if low.endswith((".jpg", ".jpeg")):
            media = "image/jpeg"
        elif low.endswith(".webp"):
            media = "image/webp"

        prompt = (
            "You are a top TikTok Shop affiliate copywriter. Look at this product "
            f"photo. Product name (may be blank): '{product_name}'. Category: {niche}. "
            f"Tone: {tone}. Audience: mostly women.\n"
            "Write a short, punchy vertical-video ad script as STRICT JSON with this shape:\n"
            '{"scenes":[{"kind":"hook","voice":"...","caption":"..."},'
            '{"kind":"problem",...},{"kind":"reveal",...},'
            '{"kind":"benefit",...},{"kind":"benefit",...},{"kind":"benefit",...},'
            '{"kind":"proof",...},{"kind":"cta",...}],'
            '"tiktok_caption":"...","hashtags":["#..."]}\n'
            "Rules: 'voice' is spoken (one short sentence). 'caption' is on-screen "
            "text (max ~6 words). No emojis in voice or caption. Put emojis only in "
            "tiktok_caption. End on a strong call to action to tap the cart. "
            "Return ONLY the JSON, nothing else."
        )
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": media, "data": img_b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        import json
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1])
        if data.get("scenes"):
            return data
    except Exception:
        return None
    return None
