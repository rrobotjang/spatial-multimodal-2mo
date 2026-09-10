# Task 3: SEG 지식 추출 — KITTI + UNet/UNet++

> Notebook: `~/Downloads/segmentation_project_with_unetpp_overlay.ipynb` (21 code cells)
> Executed: 2026-09-10 | venv: `/opt/anaconda3/envs/venv` (Python 3.14, torch 2.12.1, torchvision 0.27.1)

---

## 1. 모델 구조

### 1.1 UNet (notebook Cell 30, "binary road segmentation")

| Component | Detail |
|---|---|
| Task | Binary road segmentation (KITTI semantic label 7 = road) |
| Input | 3-channel RGB, 224×224, normalized `/255.0` |
| Output | 1-channel sigmoid mask, 224×224 |
| Encoder | 4 × `double_conv` (64→128→256→512) + MaxPool2d(2) between stages |
| Bottleneck | `double_conv(512,1024)` + `Dropout(0.5)` |
| Decoder | 4 × ConvTranspose2d(2×2, stride 2) + skip-concat + `double_conv` (1024→512→256→128→64) |
| Skip connections | Direct `torch.cat([up, enc_feat], dim=1)` at each decoder level |
| Output head | `Conv2d(64, 1, kernel_size=1)` + `sigmoid` |
| Params | **31,031,745** (smoke-measured) |
| `double_conv` | `Conv3×3 pad=1 + ReLU(inplace) + Conv3×3 pad=1 + ReLU(inplace)` |

```python
self.enc1 = self.double_conv(input_channels, 64)
self.pool1 = nn.MaxPool2d(2)   # ×4 (enc2:64→128, enc3:128→256, enc4:256→512)
self.bottleneck = self.double_conv(512, 1024)
self.dropout = nn.Dropout(0.5)
self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
self.dec6 = self.double_conv(1024, 512)   # up6 결과를 c4와 concat
# ... dec7(512,256) dec8(256,128) dec9(128,64)
self.final = nn.Conv2d(64, output_channels, kernel_size=1)  # → sigmoid
```

**Decoder (forward):**
```python
u6 = self.up6(c5); u6 = torch.cat([u6, c4], dim=1); c6 = self.dec6(u6)
u7 = self.up7(c6); u7 = torch.cat([u7, c3], dim=1); c7 = self.dec7(u7)
u8 = self.up8(c7); u8 = torch.cat([u8, c2], dim=1); c8 = self.dec8(u8)
u9 = self.up9(c8); u9 = torch.cat([u9, c1], dim=1); c9 = self.dec9(u9)
output = torch.sigmoid(self.final(c9))
```

### 1.2 UNetPlusPlus (notebook Cell 59, Zhou et al. 2018/2019, nested dense skip)

| Component | Detail |
|---|---|
| Filters | `nb_filter = [64, 128, 256, 512, 1024]` |
| Backbone (j=0) | Same as UNet encoder: `conv0_0` … `conv4_0` |
| Upsample | `nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)` (vs UNet's ConvTranspose) |
| Skip pathway | Grid nodes `X_{i,j}` — concat of same-depth previous nodes `X_{i,0..j-1}` + `up(X_{i+1,j-1})` |
| Dense connections | j=1: 2 inputs, j=2: 3 inputs, j=3: 4 inputs, j=4: 5 inputs (most dense) |
| Deep supervision | Optional `deep_supervision=True` → avg of sigmoid outputs from `final1..final4` (X0_1..X0_4); default False → single `final` on X0_4 |
| Params (no DS) | **36,615,041** (smoke-measured; vs UNet 31.0M, +18%) |
| Dropout | `Dropout(0.5)` on bottleneck `conv4_0` output |

```python
# j=1 열: X_{i,1} = double_conv( cat[ X_{i,0}, up(X_{i+1,0}) ] )
self.conv0_1 = self.double_conv(nb_filter[0] + nb_filter[1], nb_filter[0])
# j=2 열: cat[ X_{i,0}, X_{i,1}, up(X_{i+1,1}) ]
self.conv0_2 = self.double_conv(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
# j=3 열, j=4 열 (가장 dense: X0_4 가 최종 출력)
self.conv0_4 = self.double_conv(nb_filter[0]*4 + nb_filter[1], nb_filter[0])
```

```python
x0_0 = self.conv0_0(x)
x1_0 = self.conv1_0(self.pool(x0_0))
x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
# ...
x4_0 = self.dropout(self.conv4_0(self.pool(x3_0)))
x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))
output = torch.sigmoid(self.final(x0_4))   # or mean of final1..final4 if deep_supervision
```

**UNet vs UNet++ key differences for fair comparison (notebook intent):**
- Same `double_conv` block style, same loss/optimizer/epochs
- UNet++ replaces ConvTransposeT2d skip bridges with **dense nested skip + bilinear upsample**
- UNet++ batch_size halved (16→8) because of higher parameter/memory count

---

## 2. 데이터 준비

### 2.1 KittiDataset (notebook Cell 26)

```python
class KittiDataset(Dataset):
    def __init__(self, dir_path, img_size=(224,224,3), output_size=(224,224),
                 is_train=True, augmentation=None):
        self.data = self.load_dataset()

    def load_dataset(self):
        input_images = sorted(glob(os.path.join(self.dir_path, "image_2", "*.png")))
        label_images = sorted(glob(os.path.join(self.dir_path, "semantic", "*.png")))
        assert len(input_images) == len(label_images)
        data = list(zip(input_images, label_images))
        if self.is_train:
            return data[:-30]      # train = all but last 30
        return data[-30:]          # test  = last 30

    def __getitem__(self, index):
        _input  = imread(input_img_path)
        _output = imread(output_path)
        _output = (_output == 7).astype(np.uint8) * 1    # label 7 = road → binary mask
        data = {"image": _input, "mask": _output}
        if self.augmentation:
            augmented = self.augmentation(**data)
            _input  = augmented["image"] / 255.0         # normalize
            _output = augmented["mask"]
        _output = np.expand_dims(_output, axis=0)        # (H,W) → (1,H,W)
        return (torch.tensor(_input, dtype=torch.float32).permute(2,0,1),  # (C,H,W)
                torch.tensor(_output, dtype=torch.float32))                # (1,H,W)
```

**Dataset key facts:**
- Source: KITTI `data_semantics` (`s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip`, 313 MB) — semantic labels from city scenes
- Task reduces to **binary road segmentation**: `semantic == 7` → road mask (0/1 uint8)
- Train/test split: last 30 samples are held out as test (`data[:-30]` / `data[-30:]`)
- Image: RGB 1242×375→224×224, float32 `[0,1]` via `/255.0`, channel-first `(C,H,W)`
- Mask: `(1,H,W)` float32 binary
- `shuffle_data()` re-shuffles `self.data` after each epoch (np.random)

### 2.2 Augmentation (notebook Cell 18, albumentations 2.0.8)

```python
def build_augmentation(is_train=True):
    if is_train:
        return Compose([
            HorizontalFlip(p=0.5),
            RandomSizedCrop(min_max_height=(300, 370), size=(224, 224),
                            w2h_ratio=370/1242, p=0.5),
            Resize(height=224, width=224)])
    return Compose([Resize(height=224, width=224)])   # test: resize only
```

| Transform | Train | Test |
|---|---|---|
| HorizontalFlip | p=0.5 | — |
| RandomSizedCrop (300-370×224, w2h=370/1242) | p=0.5 | — |
| Resize 224×224 | always | always |

**Verification note from notebook:** label must be passed as `mask` (nearest-neighbor interpolation), NOT as `image` (bilinear would smear label value 7 and contaminate IoU ground truth).

### 2.3 DataLoaders

```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)
# UNet++: batch_size=8 (memory) — train_loader_unetpp / test_loader_unetpp
```

---

## 3. 메트릭 로그 (notebook output cells)

### 3.1 UNet training loss (Cell 43, 100 epochs, full notebook run)

```
Epoch 1/100,  Loss: 0.7183
Epoch 2/100,  Loss: 0.6983
Epoch 3/100,  Loss: 0.5975
Epoch 4/100,  Loss: 0.5254
Epoch 5/100,  Loss: 0.4348
Epoch 6/100,  Loss: 0.3751
Epoch 7/100,  Loss: 0.3544
Epoch 8/100,  Loss: 0.3155
Epoch 9/100,  Loss: 0.3146
Epoch 10/100, Loss: 0.2707
Epoch 11/100, Loss: 0.2641
Epoch 12/100, Loss: 0.2552
Epoch 13/100, Loss: 0.2356
... (수렴, 뒤로 갈수록 완만)
```

**Training config (Cell 41/43):** loss=`BCELoss`, optimizer=`Adam(lr=1e-4)`, `num_epochs=100`, threshold=0.5 → binary mask.

### 3.2 UNet++ training loss (Cell 74, 100 epochs)

```
Epoch 1/100,  Loss: 0.6627
Epoch 2/100,  Loss: 0.4833
Epoch 3/100,  Loss: 0.3379
Epoch 4/100,  Loss: 0.3180
Epoch 5/100,  Loss: 0.2650
Epoch 6/100,  Loss: 0.2400
Epoch 7/100,  Loss: 0.2571
Epoch 8/100,  Loss: 0.2086
Epoch 9/100,  Loss: 0.1948
Epoch 10/100, Loss: 0.1957
Epoch 11/100, Loss: 0.1878
Epoch 12/100, Loss: 0.1760
Epoch 13/100, Loss: 0.1906
...
```

UNet++ converges faster and lower in early epochs (0.176 vs UNet ~0.24 at epoch 12).

### 3.3 Mean IoU over full test set (Cell 79)

```
U-Net   Mean IoU (test set 전체): 0.7607
U-Net++ Mean IoU (test set 전체): 0.7630
-> U-Net++가 U-Net보다 더 나은 성능을 보였습니다.
```

### 3.4 Per-sample IoU (Cell 54 & 81, sample i=1, 000001_10.png)

```
=== U-Net ===
IoU : 0.894967
=== U-Net++ ===
IoU : 0.921453      → (better than U-Net on sample 1)
```

### 3.5 ioU definition (notebook Cell 51)

```python
def calculate_iou_score(target, prediction):
    if target.shape != prediction.shape:
        prediction = resize(prediction, target.shape, mode='constant', preserve_range=True)
    intersection = np.logical_and(target, prediction).sum()
    union = np.logical_or(target, prediction).sum()
    iou_score = intersection / (union + 1e-7)
    return iou_score
```

Full-test IoU (Cell 76 `evaluate_mean_iou`): threshold 0.5, per-sample `intersection/(union+1e-7)`, then mean.

### 3.6 Environment (Cell 66)

```
torch 2.11.0+cu128 | numpy 2.1.3 | PIL 11.3.0 | skimage 0.25.2 | albumentations 2.0.8
```

### 3.7 Overlay visualization (Cell 81)

Green=GT only, Red=Pred only, Orange/Yellow=Overlap (GT∩Pred); alpha=0.55 blend on original RGB.

---

## 4. 재현 smoke 로그

### Command

```bash
cd /Users/robotjang/spatial-multimodal-2mo
KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/venv/bin/python scripts/smoke_seg.py
```

### Output (exit code **0**)

```
PyTorch: 2.12.1, CUDA: False

=== UNet forward ===
Output shape: torch.Size([2, 1, 224, 224])
Value range: [0.5142, 0.5276] (sigmoid)
Mean pixel: 0.5210
Parameters: 31,031,745

=== UNetPlusPlus forward ===
Output shape: torch.Size([2, 1, 224, 224])
Parameters: 36,615,041
Deep supervision output shape: torch.Size([2, 1, 224, 224])

=== Loading local weights: /Users/robotjang/Downloads/segmentation/seg_model_unet.pth (118.4 MB) ===
Loaded UNet forward shape: torch.Size([2, 1, 224, 224])
Loaded model prediction range: [0.0000, 0.1208]
Loaded model mean prediction: 0.0000
Binary threshold(0.5) road pixels: 0.00%
WEIGHT_LOAD_OK

=== IoU smoke (random GT vs UNet pred) ===
IoU: 0.498147
Intersection pixels: 24995, Union pixels: 50176

=== ALL SMOKE TESTS PASSED ===
```

### Verification

- UNet instantiated (3→1 channels), forward `2×3×224×224` → `torch.Size([2,1,224,224])` ✓ (matches mask resolution 224×224 + 1 class)
- Outputs in sigmoid range [0,1], mean ≈0.52 on random input (untrained init) ✓
- Params: UNet 31,031,745 / UNet++ 36,615,041 (+18%) — matches notebook's "UNet++는 파라미터 수가 많아" rationale ✓
- Deep supervision UNet++ also produces valid `(2,1,224,224)` ✓
- **Local trained weights** `~/Downloads/segmentation/seg_model_unet.pth` (118.4 MB) loaded successfully → forward shape correct, predictions in [0.0000, 0.1208] on random noise (model trained on real roads outputs ≈0 road pixels for noise input — plausible; real KITTI image inference untestable without dataset) ✓
- IoU calc (`intersection/(union+1e-7)`) reproduces notebook metric function ✓

---

## T6/T9 Spatial Data Insights

| Insight | Source | T6 Usage |
|---|---|---|
| Road mask = KITTI semantic class **7** | `KittiDataset.__getitem__` | Binary drivable-area grounding target |
| Input preproc: RGB `/255.0` → float [0,1], 224×224, CHW | dataset | VLM input normalization parity |
| Mask is `(1,H,W)` float binary; threshold 0.5 at inference | dataset/eval | QA binary-class probability calibration |
| UNet skip bridges vs UNet++ dense nested skips (2-5 input concat) | Cell 30/59 | Architecture knowledge: dense skip improves small-class IoU |
| Augmentation: hflip(0.5) + RandomSizedCrop + Resize(224) | Cell 18 | Spatial-jitter augmentation template for instruction data variety |
| Test = **last 30** KITTI semantic samples (deterministic split) | `load_dataset` | Stable eval split for T9 demo scenes |
| Mean IoU: UNet 0.7607 vs UNet++ 0.7630 (full test); sample-1 0.895/0.921 | Cells 79/81 | Realistic IoU expectations for demo quality claims |
| Road label via nearest (mask) not bilinear (image) interpolation | Cell 61 comment | Metric-hygiene rule for any future custom metric |
| 100-epoch BCELoss+Adam(1e-4), batch 16 (UNet++) / 8 (UNet++) | Cells 43/74 | Train config reference if LoRA adapter needs dense baseline |

---

## Blockers & Fallback

1. **KITTI semantic dataset NOT present locally** — `data_semantics.zip` (313 MB) not in `~/Downloads`; notebook source was S3 (`s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip`). Full-data reproduction deferred to Colab (T6/T9 need it for real-scene instruction data). License: KITTI semantic (CC BY-NC-SA 3.0 — non-commercial, attribution required).
2. **Colab smoke deferred — auth required** — `google-colab-cli` 0.6.0 installed but unauthenticated; `colab status` demands OAuth browser login. User action: `colab auth`. Not a blocker for local smoke (fully exercised above).
3. **Weight-loaded forward on random noise** yields ~0% road (expected — model trained on real roads). Real-image inference contract verified only by shape/range; full per-sample IoU requires KITTI data on a GPU Colab node.

---

## Files

| File | Description |
|---|---|
| `.omo/evidence/task-3-spatial-multimodal-2mo.md` | This report |
| `scripts/smoke_seg.py` | Local smoke test (exact notebook UNet/UNet++ code + dummy batch + weight load + IoU) |