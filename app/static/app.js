"use strict";
const $ = (id) => document.getElementById(id);
let selected = null, ready = false, busy = false, objectURL = null, sequence = 0;
const percent = n => `${(n * 100).toFixed(1)}%`;
function message(text, error = false) { $("upload-message").textContent = text; $("upload-message").classList.toggle("error", error); }
function updateButton() { $("analyze").disabled = !selected || !ready || busy; }
async function health() {
  try {
    const response = await fetch("/api/health", {cache: "no-store", signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error("Health check failed");
    const h = await response.json(); ready = h.model_loaded && h.models === 3;
    $("model-status").textContent = ready ? "3 trained checkpoints loaded" : "Checkpoints unavailable";
    $("model-detail").textContent = ready ? "Ensemble ready · CPU inference · threshold 0.5" : (h.detail || "Models are unavailable.") + ". Feature previews are available.";
  } catch (error) { ready = false; $("model-status").textContent = "Service unavailable"; $("model-detail").textContent = "Start python run.py, then refresh status."; }
  updateButton();
}
function resetPrediction() { $("prediction-result").hidden = true; $("prediction-empty").hidden = false; $("result-tag").textContent = "AWAITING AUDIO"; }
function resetFeatures() { $("metadata").hidden = true; $("wave-placeholder").hidden = false; $("mel-placeholder").hidden = false; $("wave-end").textContent = "—"; for (const id of ["waveform", "spectrogram"]) { const c = $(id); c.getContext("2d").clearRect(0, 0, c.width, c.height); } }
function drawWaveform(data) {
  const canvas = $("waveform"), ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h); ctx.strokeStyle = "#e2eae4"; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) { ctx.beginPath(); ctx.moveTo(0, h * i / 4); ctx.lineTo(w, h * i / 4); ctx.stroke(); }
  ctx.strokeStyle = "#087f76"; ctx.lineWidth = Math.max(1, w / data.min.length);
  const amplitude = Math.max(0.01, ...data.max.map(Math.abs), ...data.min.map(Math.abs));
  data.min.forEach((v, i) => { const x = (i + 0.5) / data.min.length * w; ctx.beginPath(); ctx.moveTo(x, h / 2 - data.max[i] / amplitude * h * 0.43); ctx.lineTo(x, h / 2 - v / amplitude * h * 0.43); ctx.stroke(); });
}
function drawMel(matrix) {
  const c = $("spectrogram"), ctx = c.getContext("2d"), flat = matrix.flat();
  const min = Math.min(...flat), max = Math.max(...flat), stops = [[20,44,80],[20,141,141],[227,238,134]];
  ctx.clearRect(0, 0, c.width, c.height);
  matrix.forEach((row, band) => row.forEach((value, frame) => {
    const t = Math.max(0, Math.min(1, (value - min) / (max - min || 1))), k = Math.min(1, Math.floor(t * 2)), mix = t * 2 - k;
    ctx.fillStyle = `rgb(${stops[k].map((v,j) => Math.round(v + (stops[k+1][j] - v) * mix)).join(",")})`;
    ctx.fillRect(frame / row.length * c.width, (matrix.length - 1 - band) / matrix.length * c.height, Math.ceil(c.width / row.length), Math.ceil(c.height / matrix.length));
  }));
}
function renderFeatures(data) {
  drawWaveform(data.waveform); drawMel(data.log_mel);
  $("wave-placeholder").hidden = true; $("mel-placeholder").hidden = true; $("metadata").hidden = false;
  $("wave-end").textContent = `${data.duration_seconds.toFixed(2)} s`;
  $("duration").textContent = `${data.duration_seconds.toFixed(2)} s`;
  $("sample-rate").textContent = `${data.sample_rate.toLocaleString()} Hz`;
  $("channels").textContent = data.channels === 1 ? "1 · Mono" : `${data.channels} channels`;
  $("audio-warning").textContent = data.warnings.join(" ");
}
async function requestAudio(endpoint, file) {
  const form = new FormData(); form.append("file", file);
  let response;
  try {
    response = await fetch(endpoint, {method: "POST", body: form, signal: AbortSignal.timeout(60000)});
  } catch (error) {
    throw new Error(error.name === "TimeoutError" ? "Analysis timed out. Try a shorter recording or restart the local service." : "Cannot reach the local service. Start DeepGuard and try again.");
  }
  let data;
  try { data = await response.json(); }
  catch { throw new Error("The service returned an invalid response. Restart DeepGuard and try again."); }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Audio request failed.");
  return data;
}
async function choose(file) {
  const ticket = ++sequence; selected = null; busy = false; resetPrediction(); resetFeatures(); updateButton();
  $("audio-file").value = ""; // Allow choosing the same file again after a failure or drop.
  if (objectURL) { URL.revokeObjectURL(objectURL); objectURL = null; }
  $("player").hidden = true; $("player").removeAttribute("src"); $("file-card").hidden = true;
  if (!file) { message("Choose an audio file to preview its features."); return; }
  if (file.size > 20 * 1024 * 1024) { message("Choose a file smaller than 20 MB.", true); return; }
  $("file-card").hidden = false; $("file-name").textContent = file.name; $("file-size").textContent = `${(file.size / 1024).toFixed(0)} KB`;
  objectURL = URL.createObjectURL(file); $("player").src = objectURL; $("player").hidden = false;
  busy = true; message("Reading audio and extracting Log-Mel features…");
  try {
    const data = await requestAudio("/api/features", file); if (ticket !== sequence) return;
    renderFeatures(data); selected = file; $("result-tag").textContent = ready ? "READY TO ANALYZE" : "WEIGHTS REQUIRED";
    message(ready ? "Features ready. Analyze Audio to run the ensemble." : "Features ready. Prediction requires the three original trained checkpoints.");
  } catch (error) { if (ticket === sequence) message(error.message, true); }
  finally { if (ticket === sequence) { busy = false; updateButton(); } }
}
$("audio-file").addEventListener("change", event => choose(event.target.files[0]));
$("clear").addEventListener("click", () => { $("audio-file").value = ""; choose(null); });
$("refresh").addEventListener("click", health);
$("analyze").addEventListener("click", async () => {
  if (!selected || !ready || busy) return;
  const ticket = sequence, file = selected; busy = true; updateButton(); message("Running all three CNNs and averaging probabilities…"); resetPrediction();
  try {
    const data = await requestAudio("/api/predict", file); if (ticket !== sequence) return;
    renderFeatures(data); $("prediction-empty").hidden = true; $("prediction-result").hidden = false; $("result-tag").textContent = "ANALYSIS COMPLETE";
    $("verdict").textContent = data.label; $("confidence").textContent = percent(data.confidence);
    $("fake-value").textContent = percent(data.fake_probability); $("real-value").textContent = percent(data.genuine_probability);
    $("fake-bar").style.width = percent(data.fake_probability); $("real-bar").style.width = percent(data.genuine_probability);
    const seeds = $("individual-predictions");
    seeds.replaceChildren();
    for (const prediction of data.individual_predictions) {
      const row = document.createElement("p");
      row.textContent = `CNN seed ${prediction.seed}: ${prediction.label} · Synthetic ${percent(prediction.fake_probability)} · Genuine ${percent(prediction.genuine_probability)}`;
      seeds.appendChild(row);
    }
    message("Analysis complete. Interpret the prediction alongside the known domain-shift limitations.");
  } catch (error) { if (ticket === sequence) { message(error.message, true); await health(); } }
  finally { if (ticket === sequence) { busy = false; updateButton(); } }
});
for (const event of ["dragover", "dragenter"]) $("dropzone").addEventListener(event, e => { e.preventDefault(); $("dropzone").classList.add("drag"); });
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("drag"));
$("dropzone").addEventListener("drop", e => { e.preventDefault(); $("dropzone").classList.remove("drag"); choose(e.dataTransfer.files[0]); });
health();
