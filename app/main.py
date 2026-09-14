"""FSD Spatial Reasoning Demo — standalone FastAPI backend.

Serves a static vanilla-JS frontend plus a 4-stage spatial reasoning API:

    stage1 grounding -> stage2 scene graph -> stage3 reasoning -> stage4 answer

The demo runs on the T7 Qwen2.5-VL LoRA adapter when ``outputs/lora_adapter``
is present (rule-based :class:`MockBackend` fallback otherwise). Image upload
returns ``vision_mode_pending`` (HTTP 501) because the adapter is text-only:
it requires the scene metadata emitted by the demo scenes, not a raw image.

Run from the repo root::

    uvicorn app:app --port 8011
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.pipeline import MockBackend, QwenVLBackend, SpatialPipeline

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
VAL_JSONL = REPO_ROOT / "data" / "kitti_scene_val.jsonl"
ADAPTER_DIR = REPO_ROOT / "outputs" / "lora_adapter"

# Curated demo selection from the T6 val split — mixed safe/unsafe and
# minimal/dense scenes so the 4-stage visualization is visibly diverse.
DEMO_SCENE_IDS: tuple[str, ...] = (
    "kitti-scene-70",   # Pedestrian crossing        -> CANNOT
    "kitti-scene-380",  # stop line                  -> SAFE
    "kitti-scene-256",  # lane change, 3 cars        -> SAFE
    "kitti-scene-385",  # payment vs pedestrian      -> CANNOT
    "kitti-scene-164",  # dense 4 entities           -> CANNOT
    "kitti-scene-241",  # truck/van arrangement      -> SAFE
    "kitti-scene-106",  # stop before pedestrian     -> CANNOT
    "kitti-scene-292",  # wet road                   -> SAFE
    "kitti-scene-336",  # pedestrians crossing lanes -> CANNOT
    "kitti-scene-190",  # traffic light              -> SAFE
    "kitti-scene-172",  # person sitting             -> CANNOT
    "kitti-scene-365",  # stop line (minimal)        -> SAFE
)

ALLOWED_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})

app = FastAPI(title="FSD Spatial Reasoning Demo")

def _build_pipeline() -> SpatialPipeline:
    """Select QwenVLBackend when the T7 adapter is present, else MockBackend."""
    if ADAPTER_DIR.exists():
        return SpatialPipeline(QwenVLBackend(str(ADAPTER_DIR)))
    print("[INFO] outputs/lora_adapter not found — falling back to MockBackend.")
    return SpatialPipeline(MockBackend())


_pipeline = _build_pipeline()
_scenes: dict[str, dict[str, Any]] | None = None
_result_cache: dict[str, dict[str, Any]] = {}


def _load_val_scenes() -> dict[str, dict[str, Any]]:
    """Lazily load the T6 KITTI validation scene JSONL once."""
    global _scenes
    if _scenes is None:
        scenes: dict[str, dict[str, Any]] = {}
        with VAL_JSONL.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                scene = json.loads(line)
                scenes[scene["id"]] = scene
        _scenes = scenes
    return _scenes


def _magic_image_bytes(content: bytes) -> bool:
    """Magic-byte sniff for common raster formats (pure stdlib)."""
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if content.startswith(b"\xff\xd8\xff"):
        return True  # JPEG
    if content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):
        return True  # GIF
    if content.startswith(b"BM"):
        return True  # BMP
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return True  # WebP
    return False


def _is_image(content: bytes, filename: str) -> bool:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        return False
    if _magic_image_bytes(content):
        return True
    try:
        from PIL import Image  # type: ignore[import-not-found]
        from io import BytesIO

        with Image.open(BytesIO(content)) as img:
            img.verify()
        return True
    except Exception:
        return False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/scenes")
def api_scenes() -> dict[str, Any]:
    scenes = _load_val_scenes()
    items: list[dict[str, Any]] = []
    for sid in DEMO_SCENE_IDS:
        scene = scenes.get(sid)
        if scene is None:
            continue
        entities = scene.get("entities", [])
        classes = sorted({e.get("class", "") for e in entities if e.get("class", "")})
        items.append(
            {
                "id": sid,
                "question": scene.get("question", ""),
                "image_path": scene.get("image_path", ""),
                "entity_count": len(entities),
                "classes": classes,
            }
        )
    return {"scenes": items}


@app.post("/api/demo/{scene_id}")
def api_demo(scene_id: str) -> dict[str, Any]:
    scenes = _load_val_scenes()
    scene = scenes.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"Unknown scene: {scene_id}")
    cached = _result_cache.get(scene_id)
    if cached is not None:
        return cached
    result = _pipeline.run(scene)
    _result_cache[scene_id] = result
    return result


@app.post("/api/upload")
async def api_upload(file: UploadFile) -> Any:
    content = await file.read()
    name = file.filename or ""
    if not content or not _is_image(content, name):
        raise HTTPException(status_code=400, detail="Not a valid image file — upload PNG/JPG")
    return JSONResponse(
        status_code=501,
        content={
            "status": "vision_mode_pending",
            "code": 501,
            "message": (
                "Text-only adapter is ready; raw-image upload requires the "
                "vision branch (object detection -> entities). Use /api/demo/{scene_id}."
            ),
        },
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))