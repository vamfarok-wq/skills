"""
The orchestrator: takes the user's request and produces the finished video.

Pipeline:
  1. Get the script (from the edited script the user submitted, or generate one).
  2. Synthesize a voiceover clip per scene (free Edge TTS) -> gives scene timings.
  3. Build the prepared scene background from the product photo.
  4. Render all frames -> silent mp4.
  5. Mux the voiceover (+ optional music) onto the video.
  6. Write a matching .txt with the TikTok caption, hashtags and script.

Everything lands in the local `output/` folder. Nothing is uploaded anywhere.
"""

import os
import re
import glob
import time

import config
from engine import copywriter, voice, image_prep, render


def _safe_name(name):
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", (name or "product").strip()).strip("-")
    return (name or "product")[:50].lower()


def _find_music():
    """First audio file the user dropped into assets/music, if any."""
    for ext in ("mp3", "m4a", "wav", "aac", "ogg"):
        hits = sorted(glob.glob(os.path.join(config.MUSIC_DIR, f"*.{ext}")))
        if hits:
            return hits[0]
    return None


def scene_duration(spoken_text, audio_dur):
    if audio_dur and audio_dur > 0:
        return max(config.SCENE_MIN, min(config.SCENE_MAX, audio_dur + config.SCENE_PAD))
    return voice.estimate_duration(spoken_text)


def build(params, image_path, progress_cb=None):
    """
    params keys:
      product_name, niche, tone, voice, style, handle, price, details,
      benefits (list, optional), scenes (list, optional override),
      tiktok_caption (optional), hashtags (optional), use_music (bool)
    Returns dict: {video_path, caption_path, filename, duration, used_voice}
    """
    def report(frac, msg):
        if progress_cb:
            progress_cb(frac, msg)

    niche = params.get("niche", "general")
    palette = config.PALETTES.get(niche, config.PALETTES["general"])
    tone = params.get("tone", config.DEFAULT_TONE)
    voice_id = params.get("voice", config.DEFAULT_VOICE)
    style = params.get("style", config.DEFAULT_STYLE)
    handle = params.get("handle", "")

    # 1) script -------------------------------------------------------------
    report(0.04, "Writing your ad script…")
    if params.get("scenes"):
        scenes = params["scenes"]
        tiktok_caption = params.get("tiktok_caption", "")
        hashtags = params.get("hashtags", [])
    else:
        script = copywriter.generate_script(
            params.get("product_name", ""), niche, tone,
            details=params.get("details", ""), benefits=params.get("benefits"),
            handle=handle, price=params.get("price", ""))
        scenes = script["scenes"]
        tiktok_caption = script["tiktok_caption"]
        hashtags = script["hashtags"]

    # 2) voiceover ----------------------------------------------------------
    report(0.10, "Recording the voiceover…")
    segments, durations = [], []
    for i, s in enumerate(scenes):
        spoken = (s.get("voice") or s.get("caption") or "").strip()
        seg_path = os.path.join(config.TMP_DIR, f"seg_{i:02d}.mp3")
        seg, dur = voice.synth_line(spoken, voice_id, tone, seg_path)
        segments.append(seg)
        durations.append(scene_duration(spoken, dur))
        report(0.10 + 0.25 * (i + 1) / len(scenes), "Recording the voiceover…")

    used_voice = any(segments)

    # 3) background ---------------------------------------------------------
    report(0.38, "Designing the visuals…")
    base = image_prep.build_scene_base(image_path, palette)
    cards = [render.make_caption_card(s, palette) for s in scenes]

    # 4) render frames ------------------------------------------------------
    report(0.42, "Animating your video…")
    name = _safe_name(params.get("product_name", "product"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{name}_{stamp}.mp4"
    silent_path = os.path.join(config.TMP_DIR, "silent.mp4")
    final_path = os.path.join(config.OUTPUT_DIR, filename)

    render.render_video(
        base, scenes, cards, durations, silent_path,
        style=style, handle=handle, palette=palette,
        progress_cb=lambda fr: report(0.42 + 0.40 * fr, "Animating your video…"))

    # 5) audio + mux --------------------------------------------------------
    report(0.86, "Adding sound…")
    music_path = _find_music() if params.get("use_music") else None
    audio_path = os.path.join(config.TMP_DIR, "final_audio.mp3")
    audio = voice.build_track(segments, durations, music_path, audio_path)

    import subprocess
    if audio:
        subprocess.run([voice.FFMPEG, "-y", "-i", silent_path, "-i", audio,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        final_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        # no audio at all -> just keep the silent video
        subprocess.run([voice.FFMPEG, "-y", "-i", silent_path, "-c", "copy", final_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # 6) caption text file --------------------------------------------------
    report(0.96, "Finishing up…")
    caption_path = os.path.join(config.OUTPUT_DIR, filename.replace(".mp4", ".txt"))
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write("=== TIKTOK CAPTION (copy & paste this when you post) ===\n\n")
        f.write(tiktok_caption.strip() + "\n\n")
        f.write("=== HASHTAGS ===\n" + " ".join(hashtags) + "\n\n")
        f.write("=== SCRIPT (what the voice says) ===\n")
        for s in scenes:
            f.write(f"[{s.get('kind','')}] {s.get('voice','')}\n")
        f.write("\nTip: in the TikTok app you can add a trending sound on top of "
                "this video for an extra reach boost.\n")

    total_dur = sum(durations)
    report(1.0, "Done!")
    return {
        "video_path": final_path,
        "caption_path": caption_path,
        "filename": filename,
        "duration": round(total_dur, 1),
        "used_voice": used_voice,
    }
