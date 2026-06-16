"use strict";

const $ = (id) => document.getElementById(id);
let photoFile = null;
let pollTimer = null;

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3000);
}
function show(id) { $(id).classList.remove("hidden"); }
function hide(id) { $(id).classList.add("hidden"); }

// ---- load dropdown options -------------------------------------------------
async function loadOptions() {
  const o = await (await fetch("/api/options")).json();
  const fill = (sel, items, def) => {
    sel.innerHTML = "";
    items.forEach(([val, label]) => {
      const opt = document.createElement("option");
      opt.value = val; opt.textContent = label;
      if (val === def) opt.selected = true;
      sel.appendChild(opt);
    });
  };
  fill($("niche"), o.niches);
  fill($("voice"), o.voices, o.default_voice);
  fill($("tone"), o.tones, o.default_tone);
  fill($("style"), o.styles, o.default_style);
  if (o.ai_enabled) $("aiRow").hidden = false;

  $("style").addEventListener("change", () => {
    const note = $("styleNote");
    if ($("style").value === "human") {
      note.hidden = false;
      note.textContent = "Note: a realistic AI human presenter needs a paid avatar " +
        "service (HeyGen/D-ID) and an API key. Without one, this falls back to the " +
        "animated style — which works great for TikTok Shop.";
    } else { note.hidden = true; }
  });
}

// ---- photo upload ----------------------------------------------------------
function setPhoto(file) {
  if (!file) return;
  photoFile = file;
  const img = $("preview");
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $("dropInner").hidden = true;
}
$("photo").addEventListener("change", (e) => setPhoto(e.target.files[0]));
const drop = $("drop");
["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.add("drag");
}));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.remove("drag");
}));
drop.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) setPhoto(e.dataTransfer.files[0]);
});

// ---- step 1 -> script ------------------------------------------------------
$("btnScript").addEventListener("click", async () => {
  if (!photoFile) { toast("Please upload a product photo first 📸"); return; }
  const btn = $("btnScript");
  btn.disabled = true; btn.textContent = "Writing…";
  try {
    const fd = new FormData();
    fd.append("product_name", $("product_name").value);
    fd.append("niche", $("niche").value);
    fd.append("tone", $("tone").value);
    fd.append("handle", $("handle").value);
    fd.append("price", $("price").value);
    fd.append("details", $("details").value);
    const useAi = $("use_ai") && $("use_ai").checked;
    fd.append("use_ai", useAi ? "true" : "false");
    if (useAi) fd.append("image", photoFile);

    const data = await (await fetch("/api/script", { method: "POST", body: fd })).json();
    renderScript(data);
    toast(data.ai_used ? "✨ AI wrote this from your photo — edit away!"
                       : "Script ready — edit any line you like!");
    hide("step1"); show("step2");
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (e) {
    toast("Could not write the script. Is the app still running?");
  } finally {
    btn.disabled = false; btn.textContent = "Write my ad script →";
  }
});

let currentScript = null;
function renderScript(data) {
  currentScript = data;
  const list = $("scriptList");
  list.innerHTML = "";
  data.scenes.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "scene";
    div.innerHTML = `<div class="kind">${s.kind}</div>`;
    const ta = document.createElement("textarea");
    ta.rows = 2; ta.value = s.voice || s.caption || "";
    ta.dataset.idx = i;
    div.appendChild(ta);
    list.appendChild(div);
  });
  $("tiktok_caption").value = data.tiktok_caption || "";
}

$("btnBack").addEventListener("click", () => { hide("step2"); show("step1"); });

// ---- step 2 -> generate ----------------------------------------------------
$("btnGenerate").addEventListener("click", async () => {
  // collect edited script
  const tas = $("scriptList").querySelectorAll("textarea");
  const scenes = currentScript.scenes.map((s, i) => ({
    kind: s.kind,
    voice: tas[i].value.trim(),
    caption: tas[i].value.trim(),   // what you edit is exactly what's shown & spoken
  }));
  const params = {
    product_name: $("product_name").value,
    niche: $("niche").value,
    tone: $("tone").value,
    voice: $("voice").value,
    style: $("style").value,
    handle: $("handle").value,
    price: $("price").value,
    details: $("details").value,
    use_music: $("use_music").checked,
    scenes,
    tiktok_caption: $("tiktok_caption").value,
    hashtags: currentScript.hashtags || [],
  };

  const fd = new FormData();
  fd.append("image", photoFile);
  fd.append("params", JSON.stringify(params));

  hide("step2"); show("step3");
  $("bar").style.width = "0%";
  $("progressMsg").textContent = "Starting…";

  try {
    const res = await (await fetch("/api/generate", { method: "POST", body: fd })).json();
    if (res.error) { toast(res.error); hide("step3"); show("step2"); return; }
    pollStatus(res.job_id);
  } catch (e) {
    toast("Could not start. Is the app still running?");
    hide("step3"); show("step2");
  }
});

function pollStatus(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    let job;
    try { job = await (await fetch("/api/status/" + jobId)).json(); }
    catch (e) { return; }
    $("bar").style.width = (job.progress || 0) + "%";
    if (job.message) $("progressMsg").textContent = job.message;

    if (job.state === "done") {
      clearInterval(pollTimer);
      showResult(job);
    } else if (job.state === "error") {
      clearInterval(pollTimer);
      toast(job.message || "Something went wrong.");
      hide("step3"); show("step2");
    }
  }, 900);
}

function showResult(job) {
  hide("step3"); show("step4");
  const v = $("resultVideo");
  v.src = job.video_url + "?t=" + Date.now();
  $("btnDownload").href = job.video_url;
  $("btnCaption").href = job.caption_url;
  $("savedPath").textContent = "Saved to: " + (job.video_path || "the output folder");
  if (!job.used_voice) {
    const w = $("voiceWarn");
    w.hidden = false;
    w.textContent = "Heads up: the voiceover couldn't be generated (no internet for the " +
      "free voice service). Your video was made with captions only — connect to the " +
      "internet and try again for a spoken voiceover.";
  } else { $("voiceWarn").hidden = true; }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$("btnAnother").addEventListener("click", () => {
  hide("step4"); show("step1");
  window.scrollTo({ top: 0, behavior: "smooth" });
});

loadOptions();
