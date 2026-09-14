/* FSD Spatial Reasoning Demo — vanilla-JS frontend (no framework, no CDN). */
"use strict";

const API = {
  scenes: "/api/scenes",
  demo: (id) => `/api/demo/${encodeURIComponent(id)}`,
  upload: "/api/upload",
};

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

const state = { sceneId: null };

function fmtBbox(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return "—";
  return bbox.map((v) => Number(v).toFixed(3)).join(", ");
}

function setSceneQuestion(question) {
  const el = $("#scene-question");
  el.textContent = question || "";
}

function setStatus(text, kind) {
  const el = $("#status");
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
}

function setStageBody(stageId, html) {
  const body = $(`#${stageId} .stage-body`);
  if (body) body.innerHTML = html;
}

function placeholder(msg) {
  return `<p class="placeholder">${msg}</p>`;
}

function setStagesLoading() {
  setStageBody("stage1", placeholder("Grounding entities…"));
  setStageBody("stage2", placeholder("Building scene graph…"));
  setStageBody("stage3", placeholder("Running staged reasoning…"));
  setStageBody("stage4", placeholder("Waiting for verdict…"));
}

async function loadScenes() {
  const res = await fetch(API.scenes);
  if (!res.ok) throw new Error(`Failed to load scenes (HTTP ${res.status})`);
  const data = await res.json();
  const sel = $("#scene-select");
  sel.innerHTML = "";
  for (const scene of data.scenes || []) {
    const opt = document.createElement("option");
    opt.value = scene.id;
    opt.dataset.question = scene.question || "";
    const classes = (scene.classes || []).join("/") || "?";
    opt.textContent = `${scene.id} · [${classes}] · ${(scene.question || "").slice(0, 58)}${(scene.question || "").length > 58 ? "…" : ""}`;
    sel.appendChild(opt);
  }
  if ((data.scenes || []).length) {
    state.sceneId = data.scenes[0].id;
    sel.value = state.sceneId;
    setSceneQuestion(data.scenes[0].question);
  }
  sel.addEventListener("change", () => {
    state.sceneId = sel.value;
    const opt = sel.selectedOptions[0];
    setSceneQuestion(opt ? opt.dataset.question : "");
  });
}

function renderStage1(entities) {
  const n = entities.length;
  if (!n) return placeholder("No valid entities grounded.");
  const rows = entities
    .map(
      (e) => `<tr>
        <td class="mono">${escapeHtml(e.name || "")}</td>
        <td><span class="badge${isVulnerable(e.class) ? " vulnerable" : ""}">${escapeHtml(e.class || "")}</span></td>
        <td class="mono">${fmtBbox(e.bbox)}</td>
        <td>${e.has_pose ? "yes" : "no"}</td>
      </tr>`
    )
    .join("");
  return `<div class="stage-head"><span class="count">${n} entit${n === 1 ? "y" : "ies"}</span></div>` +
    `<table><thead><tr><th>Name</th><th>Class</th><th>Bbox (x1,y1,x2,y2)</th><th>Pose</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderStage2(edges) {
  if (!edges.length) return placeholder("No relation triplets resolved.");
  const items = edges
    .map(
      (t) => `<li class="edge"><span class="mono">${escapeHtml(t.subject || "")}</span>` +
        `<span class="rel">${escapeHtml(t.relation || "")}</span>` +
        `<span class="mono">${escapeHtml(t.object || "")}</span></li>`
    )
    .join("");
  return `<div class="stage-head"><span class="count">${edges.length} triple${edges.length === 1 ? "" : "s"}</span></div>` +
    `<ul class="edges">${items}</ul>`;
}

function renderStage3(steps) {
  if (!steps.length) return placeholder("No reasoning steps produced.");
  const items = steps
    .map(
      (s) => `<li class="reason-step"><span class="step-num unlocked">${Number(s.step)}</span>` +
        `<span class="step-text">${escapeHtml(s.text || "")}</span></li>`
    )
    .join("");
  return `<div class="stage-head"><span class="count">${steps.length} steps</span></div>` +
    `<ol class="reason-steps">${items}</ol>`;
}

function renderStage4(answer) {
  const text = answer || "Unable to determine.";
  const upper = text.toUpperCase();
  const verdict = upper.includes("CANNOT") ? "verdict-cannot" : upper.includes("SAFE") ? "verdict-safe" : "";
  return `<p class="answer-text ${verdict}">${escapeHtml(text)}</p>` +
    `<span class="kbd">${verdict ? (verdict === "verdict-safe" ? "Verdict: SAFE" : "Verdict: CANNOT PROCEED") : "Verdict unknown"}</span>`;
}

function isVulnerable(cls) {
  return ["Pedestrian", "Cyclist", "Person_sitting"].includes(cls);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function runDemo() {
  if (!state.sceneId) return;
  const btn = $("#run-btn");
  btn.disabled = true;
  setStatus("Running 4-stage pipeline…", "running");
  setStagesLoading();
  try {
    const res = await fetch(API.demo(state.sceneId), { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    setStageBody("stage1", renderStage1(data.stage1_grounding || []));
    setStageBody("stage2", renderStage2(data.stage2_scene_graph || []));
    setStageBody("stage3", renderStage3(data.stage3_reasoning || []));
    setStageBody("stage4", renderStage4(data.stage4_answer));
    setStatus(`Scene ${data.scene_id} completed (${(data.stage3_reasoning || []).length} reasoning steps).`, "ok");
  } catch (err) {
    setStatus("Error: " + err.message, "error");
    setStageBody("stage4", placeholder("Pipeline failed — see status above."));
  } finally {
    btn.disabled = false;
  }
}

function setUploadMsg(text, kind) {
  const el = $("#upload-result");
  el.textContent = text;
  el.className = "upload-msg" + (kind ? " " + kind : "");
}

async function runUpload() {
  const input = $("#upload-input");
  const btn = $("#upload-btn");
  const file = input.files && input.files[0];
  if (!file) {
    setUploadMsg("Select a file to upload first.", "error");
    return;
  }
  btn.disabled = true;
  setUploadMsg("Uploading…", "");
  try {
    const fd = new FormData();
    fd.append("file", file, file.name);
    const res = await fetch(API.upload, { method: "POST", body: fd });
    if (res.status === 400) {
      const data = await res.json().catch(() => ({}));
      setUploadMsg(data.detail || "Not a valid image file — upload PNG/JPG", "error");
    } else if (res.status === 501) {
      const data = await res.json().catch(() => ({}));
      const msg = data.message || "Text-only adapter — raw-image upload not supported. Use the Scene Demo above instead.";
      setUploadMsg(`vision_mode_pending (501): ${msg}`, "adapter");
    } else if (res.status === 503) {
      const data = await res.json().catch(() => ({}));
      const msg = data.message || "Spatial adapter not trained yet — run T7 then switch pipeline backend.";
      setUploadMsg(`adapter_not_ready (503): ${msg}`, "adapter");
    } else if (res.ok) {
      const data = await res.json().catch(() => ({}));
      setUploadMsg(`Upload accepted → ${data.status || "ok"}`, "ok");
    } else {
      const data = await res.json().catch(() => ({}));
      setUploadMsg(data.detail || `HTTP ${res.status}`, "error");
    }
  } catch (err) {
    setUploadMsg("Upload failed: " + err.message, "error");
  } finally {
    btn.disabled = false;
    input.value = "";
  }
}

function init() {
  $("#run-btn").addEventListener("click", runDemo);
  $("#upload-btn").addEventListener("click", runUpload);
  loadScenes().catch((err) => setStatus("Error: " + err.message, "error"));
}

document.addEventListener("DOMContentLoaded", init);