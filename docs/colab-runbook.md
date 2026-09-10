# Colab Runbook — T7: Qwen2.5-VL-3B LoRA Fine-Tuning

**Quick Start (single command after auth):**
```bash
colab run --gpu T4 --conda base --repo /Users/robotjang/spatial-multimodal-2mo \
  -- "bash .colab/setup_and_train.sh"
```

---

## Prerequisites

| Requirement | Status | Notes |
|---|---|---|
| HF Token | **USER MUST SET** | `export HF_TOKEN=hf_xxx` (gated model access required) |
| Colab OAuth | **USER MUST RUN** | `colab auth` (one-time browser OAuth) |
| colab CLI | ✅ installed | `/opt/anaconda3/envs/venv/bin/colab` v0.6.0 |

**Verify before proceeding:**
```bash
export HF_TOKEN=hf_<your_token>          # REQUIRED
colab auth                                # one-time OAuth
colab status                             # must exit 0
```

---

## Step 1 — Create Colab Runtime + Install Dependencies

```bash
/opt/anaconda3/envs/venv/bin/colab run --gpu T4 \
  --repo /Users/robotjang/spatial-multimodal-2mo \
  -- "pip install -q transformers>=4.49 peft bitsandbytes accelerate datasets sentencepiece && echo '[OK] deps installed'"
```

> **A100 alternative:** replace `--gpu T4` with `--gpu A100` (faster, uses ~13.5 CU/hr vs T4 ~1.75 CU/hr).

---

## Step 2 — Transformers Compatibility Probe

```bash
/opt/anaconda3/envs/venv/bin/colab exec \
  -- "cd /content/spatial-multimodal-2mo && \
      KMP_DUPLICATE_LIB_OK=TRUE python scripts/probe_transformers_compat.py"
```

**Expected output:**
```
VERDICT: PASS — Qwen2.5-VL is supported in this transformers installation.
```

**If FAIL** → pin compatible version before proceeding:
```bash
/opt/anaconda3/envs/venv/bin/colab exec \
  -- "pip install 'transformers>=4.49,<5' && \
      cd /content/spatial-multimodal-2mo && \
      KMP_DUPLICATE_LIB_OK=TRUE python scripts/probe_transformers_compat.py"
```

---

## Step 3 — Download KITTI (for vision mode only; skip for text-only)

> **Text-only mode (default):** No KITTI images needed. Structured text from entities/scene_graph/staged_reasoning is used as training input. **Skip to Step 4.**

> **Vision mode:** Requires real KITTI images. Run this step first.

```bash
/opt/anaconda3/envs/venv/bin/colab exec \
  -- "mkdir -p /data/kitti && \
      cd /data/kitti && \
      wget -q https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip && \
      unzip -q data_object_image_2.zip && \
      echo '[OK] KITTI images at /data/kitti/training/image_2/'"
```

---

## Step 4 — Run Training

**Text-only mode (default — works without images):**
```bash
/opt/anaconda3/envs/venv/bin/colab exec \
  -- "cd /content/spatial-multimodal-2mo && \
      HF_TOKEN=$HF_TOKEN \
      KMP_DUPLICATE_LIB_OK=TRUE \
      python scripts/train_lora.py \
        --mode text-only \
        --epochs 3 \
        --batch-size 1 \
        --grad-accum 16"
```

**Vision mode (after KITTI download):**
```bash
/opt/anaconda3/envs/venv/bin/colab exec \
  -- "cd /content/spatial-multimodal-2mo && \
      HF_TOKEN=$HF_TOKEN \
      KMP_DUPLICATE_LIB_OK=TRUE \
      python scripts/train_lora.py \
        --mode vision \
        --epochs 3"
```

**Expected output:**
```
[INFO] Model loaded in ~Xs
[INFO] Starting training...
[INFO] Training complete in X.XX hours
[INFO] Adapter saved to outputs/lora_adapter/
[INFO] Metrics saved to outputs/eval_metrics.json
[OK] CU estimate (X.X) within 550 monthly budget
```

---

## Step 5 — Copy Outputs Back to Repo

```bash
/opt/anaconda3/envs/venv/bin/colab exec \
  -- "cd /content/spatial-multimodal-2mo && \
      ls -la outputs/lora_adapter/ && \
      cat outputs/eval_metrics.json"
```

```bash
# Local: pull outputs from Colab
/opt/anaconda3/envs/venv/bin/colab pull \
  /content/spatial-multimodal-2mo/outputs/ \
  /Users/robotjang/spatial-multimodal-2mo/outputs/
```

---

## CU Budget Reference

| GPU | CU/hr | 3-epoch T6 (~X samples) est. | Budget |
|---|---|---|---|
| T4 (16GB) | ~1.75 | ~2-4 CU | ✅ well within 550 |
| A100 (40GB) | ~13.5 | ~15-30 CU | ✅ within 550 |

> Monitor via `colab status` during run. Stop with `colab stop` if approaching limit.

---

## Failure Recovery

| Error | Fix |
|---|---|
| **OOM** | `--batch-size 1 --grad-accum 32 --max-length 512` |
| **transformers import error** | `pip install "transformers>=4.49,<5"` in Colab step |
| **HF gated model 403** | Verify `HF_TOKEN` is set and you accepted the model card at huggingface.co |
| **Auth expired** | `colab auth` again locally |
| **OOM on eval** | Add `--max-length 512` flag |
