"""
TikTok Shop Ad Video Generator — local web app.

Run it, open the page it prints, upload a product photo, and it generates a
ready-to-post vertical ad video into the local `output/` folder.

Start with:   python app.py
Then open:    http://127.0.0.1:5000
"""

import os
import json
import uuid
import threading
import webbrowser

from flask import (Flask, request, jsonify, send_from_directory,
                   send_file, abort)

import config
from engine import copywriter, builder

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB uploads

# In-memory job tracking (single user, local app — no database needed).
JOBS = {}
ALLOWED = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# static / page
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(config.WEB_DIR, "index.html")


@app.route("/web/<path:fname>")
def web_files(fname):
    return send_from_directory(config.WEB_DIR, fname)


@app.route("/output/<path:fname>")
def output_files(fname):
    return send_from_directory(config.OUTPUT_DIR, fname)


# ---------------------------------------------------------------------------
# options for the UI
# ---------------------------------------------------------------------------
@app.route("/api/options")
def options():
    return jsonify({
        "niches": config.NICHES,
        "voices": config.VOICES,
        "tones": [(k, v["label"]) for k, v in config.TONES.items()],
        "styles": config.STYLES,
        "default_voice": config.DEFAULT_VOICE,
        "default_tone": config.DEFAULT_TONE,
        "default_style": config.DEFAULT_STYLE,
        "ai_enabled": bool(os.environ.get("ANTHROPIC_API_KEY")),
    })


def _save_upload(file_storage):
    ext = os.path.splitext(file_storage.filename or "")[1].lower()
    if ext not in ALLOWED:
        return None
    path = os.path.join(config.UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    file_storage.save(path)
    return path


# ---------------------------------------------------------------------------
# step 1: generate (or AI-draft) the script so the user can edit it
# ---------------------------------------------------------------------------
@app.route("/api/script", methods=["POST"])
def make_script():
    product_name = request.form.get("product_name", "").strip()
    niche = request.form.get("niche", "general")
    tone = request.form.get("tone", config.DEFAULT_TONE)
    handle = request.form.get("handle", "").strip()
    price = request.form.get("price", "").strip()
    details = request.form.get("details", "").strip()
    use_ai = request.form.get("use_ai", "false") == "true"

    benefits = request.form.get("benefits", "").strip()
    benefits = [b for b in benefits.split("\n") if b.strip()] if benefits else None

    result = None
    if use_ai:
        img = request.files.get("image")
        img_path = _save_upload(img) if img else None
        if img_path:
            result = copywriter.generate_with_ai(img_path, product_name, niche, tone,
                                                 handle=handle, price=price)
    ai_used = result is not None
    if not result:
        result = copywriter.generate_script(product_name, niche, tone, details=details,
                                            benefits=benefits, handle=handle, price=price)
    result["ai_used"] = ai_used
    return jsonify(result)


# ---------------------------------------------------------------------------
# step 2: render the video (runs in a background thread, UI polls status)
# ---------------------------------------------------------------------------
def _run_job(job_id, params, image_path):
    def cb(frac, msg):
        JOBS[job_id].update(progress=round(frac * 100), message=msg)
    try:
        result = builder.build(params, image_path, progress_cb=cb)
        JOBS[job_id].update(
            state="done", progress=100, message="Done!",
            video_url=f"/output/{result['filename']}",
            caption_url=f"/output/{os.path.basename(result['caption_path'])}",
            video_path=result["video_path"],
            caption_path=result["caption_path"],
            duration=result["duration"], used_voice=result["used_voice"])
    except Exception as e:
        import traceback
        traceback.print_exc()
        JOBS[job_id].update(state="error", message=f"Something went wrong: {e}")


@app.route("/api/generate", methods=["POST"])
def generate():
    img = request.files.get("image")
    if not img:
        return jsonify({"error": "Please upload a product photo."}), 400
    image_path = _save_upload(img)
    if not image_path:
        return jsonify({"error": "Unsupported image type. Use JPG, PNG or WEBP."}), 400

    try:
        params = json.loads(request.form.get("params", "{}"))
    except Exception:
        return jsonify({"error": "Bad parameters."}), 400

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"state": "running", "progress": 0, "message": "Starting…"}
    threading.Thread(target=_run_job, args=(job_id, params, image_path),
                     daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


def _open_browser(url):
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    print("\n" + "=" * 60)
    print("  TikTok Shop Ad Video Generator is running!")
    print(f"  Open this in your browser:  {url}")
    print("  (Press CTRL+C here to stop.)")
    print("=" * 60 + "\n")
    threading.Timer(1.2, _open_browser, args=(url,)).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
