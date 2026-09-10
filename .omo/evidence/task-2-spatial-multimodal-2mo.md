# Task 2: DET 지식 추출 — KITTI + RetinaNet/ResNet50

> Notebook: `~/Downloads/KITTI_RetinaNet_Autonomous_Driving_Submission-3.ipynb`
> Executed: 2026-09-10 | venv: `/opt/anaconda3/envs/venv` (Python 3.14, torch 2.12.1, torchvision 0.27.1)

---

## 1. 모델 구조

### RetinaNet-ResNet50-FPN (torchvision built-in)

**Architecture** (Cell 14):
```python
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)

old_head = model.head.classification_head
model.head.classification_head = RetinaNetClassificationHead(
    in_channels=model.backbone.out_channels,
    num_anchors=old_head.num_anchors,
    num_classes=NUM_CLASSES,  # 8
    prior_probability=0.01
)
```

| Component | Detail |
|---|---|
| Backbone | ResNet50 + Feature Pyramid Network (FPN) |
| Pretrained weights | COCO (`retinanet_resnet50_fpn_coco-eeacb38b.pth`, 130 MB) |
| Detection head | `RetinaNetClassificationHead` — replaced for 8 KITTI classes |
| Anchor config | Default torchvision anchors (`old_head.num_anchors`, typically 9 per FPN level) |
| FPN levels | C2-C5 (P2-P5) + P6 (extra conv on C5) |
| Classification prior | 0.01 (Focal loss bias initialization) |
| Loss | Focal loss (classification) + Smooth L1 (bbox regression) — built-in in torchvision RetinaNet |

**KITTI Classes (Cell 6):**
```
Car=0, Van=1, Truck=2, Pedestrian=3, Person_sitting=4, Cyclist=5, Tram=6, Misc=7
```

**Notebook class distribution (1000 samples, Cell 6):**
| Class | Count |
|---|---|
| Car | 3943 |
| Pedestrian | 562 |
| Van | 406 |
| Cyclist | 188 |
| Truck | 165 |
| Misc | 138 |
| Tram | ~80 |
| Person_sitting | ~7 |

---

## 2. 데이터 준비

### KITTIRetinaDataset (Cell 11)

Wraps `torchvision.datasets.Kitti` and converts annotations to RetinaNet format:

```python
class KITTIRetinaDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices=None, train=False):
        self.base = base_dataset
        self.indices = list(range(len(base_dataset))) if indices is None else list(indices)
        self.train = train

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, raw_target = self.base[real_idx]
        image = torchvision.transforms.functional.to_tensor(image)

        boxes, labels = [], []
        for obj in raw_target:
            name = obj["type"]
            if name not in CLASS_TO_ID:    # skip DontCare
                continue
            x1, y1, x2, y2 = obj["bbox"]
            if x2 <= x1 or y2 <= y1:      # skip invalid boxes
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_ID[name])

        # Convert to tensors or create empty
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([real_idx])}
        return image, target
```

**Key data details:**
- Image transform: `torchvision.transforms.functional.to_tensor()` (PIL → float tensor [0,1], CHW)
- No data augmentation applied (raw images only)
- Annotation: `{type, bbox(x1,y1,x2,y2), truncated, occluded, alpha}` — only `type` and `bbox` used
- `DontCare` objects excluded
- Invalid/zero-area boxes filtered

**Train/Val Split (Cell 12):**
```python
FAST_MODE = True
TRAIN_LIMIT = 600    # (or 0.9 * 7481 = 6732 in full mode)
VAL_LIMIT = 150      # (or 749 in full mode)
```
- DataLoader: `batch_size=2`, `shuffle=True`, `num_workers=2`, `collate_fn=tuple(zip(*batch))`
- Val DataLoader: `batch_size=1`, `shuffle=False`

**Dataset source:** `torchvision.datasets.Kitti` downloads KITTI 2D Object Detection from `gs://kitti-data/object_detection/` (12.6 GB images + 5.6 MB devkit).

---

## 3. 메트릭 로그 (from notebook outputs)

### Training (Cell 16, 20 epochs, FAST_MODE, GPU Tesla T4)

| Epoch | Loss | LR | Time |
|---|---|---|---|
| 1 | 1.3197 | 0.000100 | 104.7s |
| 2 | 1.3117 | 0.000100 | 104.2s |
| 3 | 1.3034 | 0.000100 | 103.9s |
| 4 | 1.2933 | 0.000100 | — |
| 5-20 | (truncated in output) | — | — |

**Training config:**
- Optimizer: SGD, lr=0.0001, momentum=0.9, weight_decay=0.0005
- Scheduler: `StepLR(step_size=10, gamma=0.1)` — LR drops to 0.00001 at epoch 11
- Gradient clipping: `clip_grad_norm_(max_norm=10.0)`

### Detection Evaluation (Cell 22, IoU≥0.5)

```
samples: 100, TP: 281, FP: 177, FN: 281
precision: 0.6135
recall: 0.5000
f1: 0.5510
```

### GO/STOP Test (Cell 27, 10 evaluation images)

```
Test images: 10/10
Correct: 8
Accuracy: 80.0%
```
- stop_1.png → Go (FP — car detected but below size threshold)
- stop_2.png → Go (FP — max score 0.256, below 0.25 threshold)
- stop_3~5.png → Stop (TP)
- go_1~5.png → Go (TP)

### Raw model output (Cell 30, stop_2.png)

```
123 raw detections, min_score=0.0504, max_score=0.2563
```
The notebook concludes that 80% was below the 90% rubric target, with tuning attempts (cells 28-30) adjusting thresholds.

---

## 4. 재현 smoke 로그

### Local smoke (CPU, `/opt/anaconda3/envs/venv`)

**Command:**
```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/venv/bin/python scripts/smoke_det.py
```

**Output:**
```
PyTorch     : 2.12.1
Torchvision : 0.27.1
Downloading: "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth" to cache
Model: RetinaNet
KITTI classes: 8
Output keys: ['boxes', 'scores', 'labels']
boxes shape : torch.Size([0, 4])
labels shape: torch.Size([0])
scores shape: torch.Size([0])
Head num_classes: 8
Dataset len: 2
Image shape: torch.Size([3, 375, 1242])
Target boxes: torch.Size([1, 4])
Target labels: tensor([0])
Forward on dataset sample OK — 0 detections
Training loss: 1.9269
Loss components: classification=1.1519, bbox_regression=0.7750

=== SMOKE PASSED ===
RetinaNet-ResNet50-FPN | 8 KITTI classes | loss=1.9269
```

**Exit code:** 0

**Verification:**
- RetinaNet-ResNet50-FPN instantiated with COCO pretrained weights ✓
- Classification head swapped to 8 KITTI classes ✓
- Forward pass on random 3×375×1242 tensor produces valid output dict (boxes/scores/labels) ✓
- KITTIRetinaDataset with fabricated 2-sample mini-dataset loads and converts correctly ✓
- Single training step computes positive loss (classification=1.1519 + bbox_regression=0.7750 = 1.9269) ✓
- All assertions passed ✓

### Notes
- Smoke uses fabricated dummy data (2 images, 1 box each) — no real KITTI needed
- Loss value 1.9269 (1 batch, random init head) vs notebook 1.3197 (600 samples, epoch mean) — expected difference since notebook averages over full epoch after 1 epoch of warm-up from COCO pretrained head
- KITTI original dataset NOT present locally (~16 GB, Colab download route required for full reproduction)
- Colab smoke deferred — OAuth auth required (user action: `colab auth`)

---

## T6/T9 Spatial Data Insights

| Insight | Source | T6 Usage |
|---|---|---|
| 8-class taxonomy: Car/Van/Truck/Pedestrian/Person_sitting/Cyclist/Tram/Misc | Cell 6 | Grounding labels |
| KITTI bbox format: (x1,y1,x2,y2) pixel coords | Cell 11 | Box normalization for VLM input |
| Class imbalance: Car=3943 >> Misc=138 | Cell 6 | Oversample rare classes in instruction data |
| FPN handles multi-scale: small pedestrians + large trucks | Cell 14 | Multi-scale grounding QA |
| Focal loss handles class imbalance natively | Built-in | Loss weighting reference |
| Self-drive rules: person height>150px → Stop, vehicle max(w,h)>150px → Stop | Cell 24 | Reasoning chain template for DRScaffold |
| score_threshold=0.35 for eval, 0.25-0.50 range for driving | Cells 20,27 | Confidence QA calibration |

---

## Blockers & Fallback

1. **KITTI dataset NOT present locally** — ~/Downloads has no image_2/label_2 dirs. Colab route for full dataset download (12.6 GB) is the planned path for T6/T9. License: KITTI (CC BY-NC-SA 3.0 — non-commercial use only, attribution required).
2. **Colab smoke deferred** — `google-colab-cli` installed but not authenticated (OAuth browser login required). User action: `colab auth`.
3. **80% GO/STOP accuracy** — Notebook achieved 80% (below 90% rubric target). The notebook attempts threshold tuning in cells 28-30 but does not reach 90%. This limitation is documented for T9 demo expectations.

---

## Files

| File | Description |
|---|---|
| `.omo/evidence/task-2-spatial-multimodal-2mo.md` | This report |
| `scripts/smoke_det.py` | Local smoke test script (RetinaNet + dummy KITTI data) |
