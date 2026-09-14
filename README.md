# spatial-multimodal-2mo

Spatial multimodal understanding for autonomous driving — a two-month (2MO) capstone project covering the full pipeline from perception knowledge extraction, synthetic spatial instruction data, structured scene reasoning, and vision-language fine-tuning, through to deployment benchmarks and an investor-ready demo.

> **Status**: core pipeline + demo complete. T7 Qwen2.5-VL LoRA fine-tuning is executed on Colab (see [docs/colab-runbook.md](docs/colab-runbook.md)); the web demo currently runs on the deterministic `MockBackend` until the adapter is available.

---

## Tasks at a Glance

| Task | Scope | Evidence |
|---|---|---|
| T1 | Environment baseline (venv, Colab CLI, HF token, repo scaffold) | `.omo/evidence/task-1-*.md` |
| T2 | **DET** knowledge extraction — KITTI + RetinaNet/ResNet50 | `.omo/evidence/task-2-*.md` |
| T3 | **SEG** knowledge extraction — KITTI + UNet/UNet++ | `.omo/evidence/task-3-*.md` |
| T4 | **POSE** knowledge extraction — MPII + Stacked Hourglass / SimpleBaseline | `task-4-spatial-multimodal-2mo.md` |
| T5 | **LANG** knowledge extraction — Korean→English Seq2Seq + Bahdanau attention | `task-5-spatial-multimodal-2mo.md` |
| T6 | Spatial instruction dataset — 400 synthesized KITTI-style samples | `.omo/evidence/task-6-*` |
| T7 | Qwen2.5-VL-3B LoRA fine-tuning on Colab | `docs/colab-runbook.md` |
| T8 | DRScaffold-style 4-stage spatial reasoning pipeline | `.omo/evidence/task-8-*.md` |
| T9 | FSD web demo E2E (FastAPI + vanilla JS) | `.omo/evidence/task-9-*` |
| T10 | Cloud deployment benchmark (Colab / ONNX / CPU) | `.omo/evidence/task-10-*.md` |
| T11 | Mobile/web deployment benchmark (WebGPU / GGUF / MLX) | `.omo/evidence/task-11-*.md` |
| T12 | US investor pitch package (demo script + license notice) | `.omo/evidence/task-12-*.md` |
| T13 | Bonus — yoga pose vision E2E (exact pose checkpoint path verified) | `.omo/evidence/task-13-*.png` |

---

## Repository Structure

```
spatial-multimodal-2mo/
├── app/                     # Standalone FastAPI demo (vanilla-JS static frontend)
│   ├── main.py              #   /health, /api/scenes, /api/demo, /api/upload
│   └── static/              #   index.html, styles.css, app.js
├── pipeline/                # DRScaffold-style 4-stage spatial reasoning
│   └── pipeline.py          #   grounding → scene graph → reasoning → answer
├── data/                    # Synthetic spatial instruction dataset
│   ├── kitti_scene_train.jsonl   # 340 samples
│   ├── kitti_scene_val.jsonl     # 60 samples
│   └── schema.json               # JSON schema for the dataset
├── scripts/                 # Dataset gen/validation, training, knowledge-extraction smoke
│   ├── generate_spatial_dataset.py
│   ├── validate_spatial_dataset.py
│   ├── train_lora.py            # T7 Qwen2.5-VL LoRA (text-only / vision modes)
│   ├── train_fixed.py / train_fs10.py
│   ├── reconstruct_pose_model.py # T4 pose architecture smoke
│   ├── seq2seq_ko_en.py          # T5 translation smoke
│   ├── smoke_det.py / smoke_seg.py
│   └── probe_transformers_compat.py
├── tools/                   # Operator tools
│   ├── fsd_spatial_demo.py      # FSD spatial-understanding v2 live/evidence demo
│   ├── colab_fsd_finish.ipynb   # Self-contained Colab completion notebook (resume kit)
│   ├── resume_train.sh / finish_training.sh / resume_1epoch.sh
├── tests/                   # Unit + E2E
│   ├── test_pipeline.py         # 3 fixtures: safe / unsafe / minimal KITTI scenes
│   └── e2e_fsd.py               # boots uvicorn and probes the HTTP API
├── docs/colab-runbook.md    # T7 Colab runbook (T4/A100, CU budget, recovery)
├── .omo/evidence/           # Per-task evidence records
├── task-4-*.md / task-5-*.md  # POSE / LANG knowledge extraction reports
```

## The 4-Stage Reasoning Pipeline

`pipeline/pipeline.py` implements a DRScaffold-style structured spatial reasoning chain:

1. **Entity grounding** — normalise bounding boxes, drop `DontCare`/invalid entries, filter to valid KITTI classes.
2. **Scene graph** — materialise relation triplets, resolve `ego_vehicle` anchor.
3. **Staged reasoning** — rule-based safety chain: inventory → graph count → vulnerable-road-user analysis → verdict.
4. **Final answer** — one English sentence summarising the decision (`SAFE — payment may proceed` / `CANNOT proceed … must stop and yield`).

Backends behind the `LLMBackend` protocol:

- `MockBackend` — deterministic, rule-based; used for unit tests, CI, and the current web demo.
- `QwenVLBackend` — T7 Qwen2.5-VL adapter stub; raises a clear `FileNotFoundError` until `outputs/lora_adapter` exists.

## FSD Spatial Under-standing Demo (`tools/fsd_spatial_demo.py`)

Runs the CrossSpatialFusion **v2** spatial-understanding model (`fsd_v2.pth`, camera RGB + LiDAR BEV fusion → 128-D spatial embedding) with an A/B improvement of **+79.5 %** in spatial MSE (v1 `1.2448` → v2 `0.2550`, same-budget A/B measurement).

```bash
python3 tools/fsd_spatial_demo.py          # honest evidence mode (works on any host)
python3 tools/fsd_spatial_demo.py --live    # real inference (requires torch-OK host / Colab)
```

## Run the Web Demo

```bash
# from the repo root
uvicorn app:app --port 8011
open http://127.0.0.1:8011
```

Image upload currently returns a structured `adapter_not_ready` (HTTP 503) until the T7 adapter is produced; scene replay endpoints (`/api/scenes`, `/api/demo/{scene_id}`) are fully functional on `MockBackend`.

## T7 Qwen2.5-VL-3B LoRA Fine-Tuning (Colab)

```bash
colab run --gpu T4 --conda base --repo /Users/robotjang/spatial-multimodal-2mo \
  -- "bash .colab/setup_and_train.sh"
```

Full prerequisite checklist, text-only vs vision mode, CU budget, and failure recovery are in [docs/colab-runbook.md](docs/colab-runbook.md).

## Tests

```bash
pytest tests/                  # unit tests (safe / unsafe / minimal scenes)
python tests/e2e_fsd.py        # boots uvicorn on :8011 and probes the API
```

## Notes

- The dataset is **synthesized** instruction data consistent with KITTI scene graphs — no real KITTI label distribution is claimed (see `data/schema.json`).
- Model weights (`.pth`/`.pt`/LoRA adapters) are gitignored; `fsd_v2.pth` external artifacts are resolved by `tools/fsd_spatial_demo.py` from live candidate paths.
- Evidence for every task lives in `.omo/evidence/`; test/metric claims there are measured, not estimated.