"""Playwright E2E tests for the FSD spatial reasoning demo app (T9).

Starts uvicorn on port 8011 from the repo root, then exercises:

  Case 1 (happy):   pick a scene -> Run Pipeline -> 4 stage cards rendered
                    and a final SAFE/CANNOT verdict is visible.
  Case 2 (failure): non-image upload (.txt, and a text file named .png) ->
                    HTTP 400 error surfaced in the UI; valid tiny PNG ->
                    adapter_not_ready (HTTP 503) surfaced in the UI.

Evidence: .omo/evidence/task-9-spatial-multimodal-2mo.png (screenshot)
          .omo/evidence/task-9-spatial-multimodal-2mo.log  (server + E2E output)

Run with the project venv:
    /opt/anaconda3/envs/venv/bin/python tests/e2e_fsd.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402

HOST = "127.0.0.1"
PORT = 8011
BASE_URL = f"http://{HOST}:{PORT}"
HEALTH_URL = f"{BASE_URL}/health"

EVIDENCE_DIR = REPO_ROOT / ".omo" / "evidence"
SCREENSHOT_PATH = EVIDENCE_DIR / "task-9-spatial-multimodal-2mo.png"
LOG_PATH = EVIDENCE_DIR / "task-9-spatial-multimodal-2mo.log"

TIMEOUT_MS = 30_000
VERDICT_SCENE = "kitti-scene-70"  # pedestrian crossing -> CANNOT verdict


def wait_for_server(timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy: {last_err}")


def main() -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_PATH.open("a", encoding="utf-8")

    def log(line: str) -> None:
        log_handle.write(f"[{time.strftime('%H:%M:%S')}] {line}\n")
        log_handle.flush()

    server: subprocess.Popen | None = None
    tmp_dir = tempfile.mkdtemp(prefix="fsd-e2e-")
    pw = None
    browser = None
    page = None
    try:
        log("=== T9 FSD E2E start ===")
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", HOST, "--port", str(PORT)],
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=log_handle,
        )
        log(f"uvicorn pid={server.pid} on {BASE_URL}")
        wait_for_server()
        log("server healthy (GET /health -> 200)")

        # --- fixtures ----------------------------------------------------
        from PIL import Image  # noqa: PLC0415

        tiny_png = Path(tmp_dir) / "tiny.png"
        Image.new("RGB", (1, 1), (210, 25, 25)).save(tiny_png, format="PNG")
        log(f"fixture: valid 1x1 PNG -> {tiny_png} ({tiny_png.stat().st_size} bytes)")

        not_image_txt = Path(tmp_dir) / "not-image.txt"
        not_image_txt.write_text("this is definitely not an image", encoding="utf-8")

        fake_png = Path(tmp_dir) / "fake.png"
        fake_png.write_text("still not an image despite the .png name", encoding="utf-8")

        # --- browser ------------------------------------------------------
        pw = sync_playwright().start()
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1500})

        console_lines: list[str] = []
        page.on("console", lambda msg: console_lines.append(f"[page:{msg.type}] {msg.text}"))
        page.on("pageerror", lambda exc: console_lines.append(f"[pageerror] {exc}"))

        # ---- CASE 1 (happy) ---------------------------------------------
        log("CASE 1 (happy): select scene -> Run -> 4 stages -> verdict")
        page.goto(BASE_URL, wait_until="load", timeout=TIMEOUT_MS)
        assert page.title() == "FSD Spatial Reasoning Demo", f"title={page.title()!r}"
        log(f"title OK: {page.title()!r}")

        page.wait_for_selector("#scene-select option[value]", state="attached", timeout=TIMEOUT_MS)
        page.select_option("#scene-select", VERDICT_SCENE)
        page.click("#run-btn")

        page.wait_for_function(
            "() => { const el = document.querySelector('#stage4 .answer-text');"
            " return el && el.textContent.trim().length > 0; }",
            timeout=TIMEOUT_MS,
        )
        answer = (page.text_content("#stage4 .answer-text") or "").strip()
        log(f"answer = {answer[:110]!r}")
        assert "SAFE" in answer or "CANNOT" in answer, f"verdict missing: {answer!r}"

        stage_count = page.locator(".stage-card").count()
        log(f"stage cards present = {stage_count}")
        assert stage_count == 4, f"expected 4 stage cards, got {stage_count}"

        ground_rows = page.locator("#stage1 tbody tr").count()
        graph_edges = page.locator("#stage2 .edge").count()
        reason_steps = page.locator("#stage3 .reason-step").count()
        log(f"grounding rows={ground_rows} graph edges={graph_edges} reasoning steps={reason_steps}")
        assert ground_rows > 0, "no grounded entities rendered"
        assert graph_edges > 0, "no scene graph edges rendered"
        assert reason_steps > 0, "no reasoning steps rendered"

        question = (page.text_content("#scene-question") or "").strip()
        log(f"scene question = {question[:90]!r}")
        assert question, "scene question not rendered"

        page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
        log(f"screenshot saved -> {SCREENSHOT_PATH}")

        # ---- CASE 2 (failure) --------------------------------------------
        log("CASE 2 (failure): non-image upload -> 400 surfaced")
        page.set_input_files("#upload-input", str(not_image_txt))
        page.click("#upload-btn")
        page.wait_for_function(
            "() => { const el = document.querySelector('#upload-result');"
            " return el && el.textContent.trim().length > 0; }",
            timeout=TIMEOUT_MS,
        )
        err_text = (page.text_content("#upload-result") or "").strip()
        log(f"txt upload msg = {err_text!r}")
        assert "Not a valid image file" in err_text, f"400 error not surfaced: {err_text!r}"

        log("CASE 2b: text file named .png -> magic-byte rejection (400)")
        page.set_input_files("#upload-input", str(fake_png))
        page.click("#upload-btn")
        page.wait_for_function(
            "() => { const el = document.querySelector('#upload-result');"
            " return el && el.textContent.includes('Not a valid image file'); }",
            timeout=TIMEOUT_MS,
        )
        fake_text = (page.text_content("#upload-result") or "").strip()
        log(f"fake-png upload msg = {fake_text!r}")
        assert "Not a valid image file" in fake_text, f"magic-byte 400 not surfaced: {fake_text!r}"

        log("CASE 2 (adapter): valid tiny PNG -> 503 adapter_not_ready surfaced")
        page.set_input_files("#upload-input", str(tiny_png))
        page.click("#upload-btn")
        page.wait_for_function(
            "() => { const el = document.querySelector('#upload-result');"
            " return el && el.textContent.includes('adapter_not_ready'); }",
            timeout=TIMEOUT_MS,
        )
        ok_text = (page.text_content("#upload-result") or "").strip()
        log(f"valid png upload msg = {ok_text!r}")
        assert "adapter_not_ready" in ok_text, f"503 not surfaced: {ok_text!r}"
        assert "503" in ok_text, f"503 status code not shown: {ok_text!r}"

        if console_lines:
            log("browser console:")
            log("\n".join(console_lines))

        log("E2E PASS: both cases exit 0")
        log("=== T9 FSD E2E done ===")
        return 0
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:  # noqa: BLE001
                pass
        if browser is not None:
            try:
                browser.close()
            except Exception:  # noqa: BLE001
                pass
        if pw is not None:
            try:
                pw.stop()
            except Exception:  # noqa: BLE001
                pass
        if server is not None:
            log(f"terminating uvicorn pid={server.pid}")
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
            log("uvicorn terminated")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log(f"removed temp dir {tmp_dir}")
        log_handle.close()


if __name__ == "__main__":
    sys.exit(main())