"""
Voiceover generation.

Primary: Edge TTS (Microsoft) — free, no API key, very natural. Needs internet.
Fallback 1: pyttsx3 — fully offline (uses your OS voices). Optional.
Fallback 2: silence — if everything fails, the video is still made with captions.

We synthesize one audio clip per scene so we know exactly how long each scene
should last, then stitch them into one track that lines up with the video.
"""

import os
import asyncio
import subprocess

import imageio_ffmpeg

import config

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _duration(path):
    """Length of an audio file in seconds (0 if unreadable)."""
    if not path or not os.path.exists(path):
        return 0.0
    try:
        import mutagen
        m = mutagen.File(path)
        if m and m.info and m.info.length:
            return float(m.info.length)
    except Exception:
        pass
    # Fallback: ask ffmpeg.
    try:
        out = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True)
        for line in out.stderr.splitlines():
            if "Duration:" in line:
                hms = line.split("Duration:")[1].split(",")[0].strip()
                h, m_, s = hms.split(":")
                return int(h) * 3600 + int(m_) * 60 + float(s)
    except Exception:
        pass
    return 0.0


async def _edge_one(text, voice, rate, pitch, out_path):
    import edge_tts
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await comm.save(out_path)


def _try_edge(text, voice, rate, pitch, out_path):
    try:
        asyncio.run(_edge_one(text, voice, rate, pitch, out_path))
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True
    except Exception:
        pass
    return False


def _try_pyttsx3(text, out_path):
    try:
        import pyttsx3
        wav = out_path.rsplit(".", 1)[0] + ".wav"
        engine = pyttsx3.init()
        engine.save_to_file(text, wav)
        engine.runAndWait()
        if os.path.exists(wav) and os.path.getsize(wav) > 0:
            # convert to mp3 for consistency
            subprocess.run([FFMPEG, "-y", "-i", wav, out_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            os.remove(wav)
            return True
    except Exception:
        pass
    return False


def synth_line(text, voice, tone_key, out_path):
    """
    Speak one line to out_path (mp3). Returns (path_or_None, duration_seconds).
    Falls back gracefully; returns (None, 0) if no audio could be produced.
    """
    tone = config.TONES.get(tone_key, config.TONES[config.DEFAULT_TONE])
    rate, pitch = tone["rate"], tone["pitch"]

    if _try_edge(text, voice, rate, pitch, out_path):
        return out_path, _duration(out_path)
    if _try_pyttsx3(text, out_path):
        return out_path, _duration(out_path)
    return None, 0.0


def estimate_duration(text):
    """Rough spoken length when we have no real audio (~2.7 words/sec)."""
    words = max(1, len(text.split()))
    return max(config.SCENE_MIN, min(config.SCENE_MAX, words / 2.7 + 0.8))


def _silence(seconds, out_path):
    subprocess.run([FFMPEG, "-y", "-f", "lavfi", "-i",
                    f"anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-t", f"{seconds:.3f}", "-q:a", "9", out_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def build_track(segments, durations, music_path, out_path):
    """
    Stitch per-scene audio into one track that matches the video timeline.

    segments: list of mp3 paths or None (None -> silence for that scene)
    durations: list of scene durations (seconds) — each block is padded to this.
    music_path: optional background music mixed quietly underneath.

    Returns out_path if any audio exists, else None (caller makes a silent video).
    """
    has_voice = any(s for s in segments)
    blocks = []
    for seg, dur in zip(segments, durations):
        block = os.path.join(config.TMP_DIR, f"blk_{len(blocks):03d}.mp3")
        if seg and os.path.exists(seg):
            # Pad (or trim) the spoken clip to exactly the scene duration.
            subprocess.run([FFMPEG, "-y", "-i", seg, "-af", "apad",
                            "-t", f"{dur:.3f}", block],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            _silence(dur, block)
        blocks.append(block)

    # Concatenate the blocks.
    listfile = os.path.join(config.TMP_DIR, "concat.txt")
    with open(listfile, "w") as f:
        for b in blocks:
            f.write(f"file '{b}'\n")
    voice_track = os.path.join(config.TMP_DIR, "voice_track.mp3")
    subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                    "-c", "copy", voice_track],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    total = sum(durations)

    if not has_voice and not music_path:
        return None  # nothing to add; video stays silent

    if music_path and os.path.exists(music_path):
        if has_voice:
            # Mix: voice on top, music ducked underneath.
            subprocess.run([FFMPEG, "-y",
                            "-i", voice_track,
                            "-stream_loop", "-1", "-i", music_path,
                            "-filter_complex",
                            "[1:a]volume=0.18[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=0[a]",
                            "-map", "[a]", "-t", f"{total:.3f}", out_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            subprocess.run([FFMPEG, "-y", "-stream_loop", "-1", "-i", music_path,
                            "-af", "volume=0.5", "-t", f"{total:.3f}", out_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out_path

    # Voice only.
    os.replace(voice_track, out_path)
    return out_path
