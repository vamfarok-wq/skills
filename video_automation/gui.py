"""
Video Automation GUI — Flask web interface.
Usage: python gui.py [--port 5000] [--no-browser]
"""

import os
import sys
import json
import queue
import threading
import logging
import time
import webbrowser
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, jsonify, request, Response, stream_with_context

# ── Logging ───────────────────────────────────────────────────────────────────

log_queue: queue.Queue = queue.Queue(maxsize=1000)


class _QueueHandler(logging.Handler):
    def emit(self, record):
        try:
            log_queue.put_nowait(self.format(record))
        except queue.Full:
            pass


_h = _QueueHandler()
_h.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")
)
logging.getLogger().addHandler(_h)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────

_gen_status: dict = {
    "running": False,
    "step": "",
    "progress": 0.0,
    "output": [],
    "error": "",
}
_gen_lock = threading.Lock()

_scheduler_thread: threading.Thread | None = None
_scheduler_stop = threading.Event()

# ── .env helpers ──────────────────────────────────────────────────────────────

_BASE = Path(__file__).parent
ENV_FILE = _BASE / ".env"
ENV_EXAMPLE = _BASE / ".env.example"


def load_env() -> dict:
    result: dict = {}
    src = ENV_FILE if ENV_FILE.exists() else ENV_EXAMPLE
    if src.exists():
        for line in src.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip()
    return result


def save_env(updates: dict):
    existing = load_env()
    existing.update({k: v for k, v in updates.items() if v is not None})
    lines: list[str] = []
    if ENV_EXAMPLE.exists():
        for line in ENV_EXAMPLE.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k = stripped.split("=", 1)[0].strip()
                lines.append(f"{k}={existing.get(k, stripped.split('=',1)[1].strip())}")
            else:
                lines.append(line)
    else:
        for k, v in existing.items():
            lines.append(f"{k}={v}")
    ENV_FILE.write_text("\n".join(lines) + "\n")

# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")
app.secret_key = os.urandom(16)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    try:
        from src.database.manager import DatabaseManager
        return jsonify(DatabaseManager().get_stats())
    except Exception as exc:
        return jsonify({"total_videos": 0, "today": 0, "successful_uploads": 0, "by_niche": {}, "error": str(exc)})


@app.route("/api/recent_videos")
def api_recent_videos():
    try:
        import sqlite3
        conn = sqlite3.connect(_BASE / "automation.db")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT niche, platform, title, status, created_at FROM videos ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception:
        return jsonify([])


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify(load_env())


@app.route("/api/settings", methods=["POST"])
def api_save_settings():
    try:
        save_env(request.json or {})
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except Exception:
            pass
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/niches")
def api_niches():
    from src.niches.manager import NICHES
    return jsonify([
        {"name": n.name, "display_name": n.display_name,
         "keywords": n.keywords, "platforms": n.platforms}
        for n in NICHES
    ])


@app.route("/api/generate", methods=["POST"])
def api_generate():
    global _gen_status
    with _gen_lock:
        if _gen_status["running"]:
            return jsonify({"ok": False, "error": "Generation already running"}), 409
        _gen_status = {"running": True, "step": "Starting…", "progress": 0.0, "output": [], "error": ""}

    body = request.json or {}
    threading.Thread(
        target=_run_generate,
        args=(body.get("niche", "tech"), body.get("platform", "youtube")),
        daemon=True,
    ).start()
    return jsonify({"ok": True})


@app.route("/api/generate/stream")
def api_generate_stream():
    def _gen():
        while True:
            with _gen_lock:
                snapshot = dict(_gen_status)
                snapshot["output"] = list(_gen_status["output"])
            yield f"data: {json.dumps(snapshot)}\n\n"
            if not snapshot["running"]:
                break
            time.sleep(0.3)
    return Response(
        stream_with_context(_gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/scheduler/start", methods=["POST"])
def api_scheduler_start():
    global _scheduler_thread, _scheduler_stop
    if _scheduler_thread and _scheduler_thread.is_alive():
        return jsonify({"ok": False, "error": "Scheduler already running"})
    _scheduler_stop.clear()
    _scheduler_thread = threading.Thread(target=_run_scheduler, daemon=True)
    _scheduler_thread.start()
    return jsonify({"ok": True})


@app.route("/api/scheduler/stop", methods=["POST"])
def api_scheduler_stop():
    _scheduler_stop.set()
    logger.info("Scheduler stop requested.")
    return jsonify({"ok": True})


@app.route("/api/scheduler/status")
def api_scheduler_status():
    running = bool(
        _scheduler_thread
        and _scheduler_thread.is_alive()
        and not _scheduler_stop.is_set()
    )
    return jsonify({"running": running})


@app.route("/api/logs/stream")
def api_logs_stream():
    def _gen():
        while True:
            try:
                msg = log_queue.get(timeout=1.0)
                yield f"data: {json.dumps({'message': msg})}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'ping': True})}\n\n"
    return Response(
        stream_with_context(_gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── Background workers ────────────────────────────────────────────────────────

def _gen_step(msg: str, prog: float):
    with _gen_lock:
        _gen_status["step"] = msg
        _gen_status["progress"] = round(prog, 3)
        _gen_status["output"].append(f"[{time.strftime('%H:%M:%S')}]  {msg}")
    logger.info("[Generate] %s", msg)


def _run_generate(niche_name: str, platform: str):
    try:
        from config import Config
        from src.niches.manager import NicheManager
        from src.content.collector import DataCollector
        from src.content.generator import ContentGenerator
        from src.video.creator import VideoCreator
        from src.database.manager import DatabaseManager

        cfg = Config()
        if not cfg.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set — go to API Settings and save your key.")

        niche = NicheManager().get(niche_name)
        if not niche:
            raise ValueError(f"Unknown niche: {niche_name}")

        _gen_step("Collecting trending data…", 0.10)
        data = DataCollector(cfg).collect(niche)
        _gen_step(f"Collected {len(data['topics'])} topics, {len(data['facts'])} facts", 0.25)

        _gen_step("Generating AI script…", 0.40)
        script = ContentGenerator(cfg).generate(data, niche, platform)
        _gen_step(f'Script ready: {script["title"][:55]}', 0.55)
        with _gen_lock:
            _gen_status["output"].append(f"  Title : {script['title']}")
            _gen_status["output"].append(f"  Tags  : {', '.join(script['tags'][:6])}")

        _gen_step("Rendering video…", 0.70)
        path = VideoCreator(cfg).create_video(script, niche_name, platform)
        if not path:
            raise RuntimeError("Video creator returned None")

        mb = os.path.getsize(path) / 1024 / 1024
        _gen_step(f"Video ready — {mb:.1f} MB", 0.86)
        with _gen_lock:
            _gen_status["output"].append(f"  File  : {path}")

        db = DatabaseManager()
        vid_id = db.save_video(
            niche_name, platform, script["title"], script["description"], script["tags"], path
        )

        # Upload
        uploaded = False
        if platform == "youtube" and cfg.UPLOAD_TO_YOUTUBE and cfg.YOUTUBE_REFRESH_TOKEN:
            _gen_step("Uploading to YouTube…", 0.93)
            from src.upload.youtube import YouTubeUploader
            r = YouTubeUploader(cfg).upload(path, script, niche)
            db.save_upload(vid_id, "youtube", r["platform_id"], r["url"], "success")
            with _gen_lock:
                _gen_status["output"].append(f"  YouTube: {r['url']}")
            uploaded = True
        elif platform == "tiktok" and cfg.UPLOAD_TO_TIKTOK and cfg.TIKTOK_ACCESS_TOKEN:
            _gen_step("Uploading to TikTok…", 0.93)
            from src.upload.tiktok import TikTokUploader
            r = TikTokUploader(cfg).upload(path, script, niche)
            db.save_upload(vid_id, "tiktok", r["platform_id"], r["url"], "success")
            with _gen_lock:
                _gen_status["output"].append(f"  TikTok: {r['url']}")
            uploaded = True

        if not uploaded:
            with _gen_lock:
                _gen_status["output"].append("  Upload skipped (credentials not configured)")

        _gen_step("Done! ✓", 1.0)

    except Exception as exc:
        logger.exception("Generation failed")
        with _gen_lock:
            _gen_status["error"] = str(exc)
            _gen_status["step"] = f"Error: {exc}"
            _gen_status["output"].append(f"  ERROR: {exc}")
    finally:
        with _gen_lock:
            _gen_status["running"] = False


def _run_scheduler():
    try:
        from config import Config
        from src.scheduler.runner import DailyScheduler
        DailyScheduler(Config()).start()
    except Exception as exc:
        logger.error("Scheduler crashed: %s", exc)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Video Automation Web GUI")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    if not args.no_browser:
        threading.Timer(1.4, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    print(f"\n  🎬  Video Automation GUI")
    print(f"  ➜  http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True, use_reloader=False)
