# POSE 지식 추출 리포트 — MPII + Stacked Hourglass / ResNet50 (SimpleBaseline)

**Task**: T4 (보너스 요가 T13의 전제)
**Notebook**: `~/Downloads/pose_estimation_submission_final_v2_transferChatgpt5.6.ipynb` (56 code cells)
**Repository**: `/Users/robotjang/spatial-multimodal-2mo`

---

## 1. 모델 구조

노트북에 구현된 두 가지 자세 추정 모델:

### 1-a. Stacked Hourglass (메인, `best`/`epoch-2` 가중치가 이 구조)

셀 21–24에서 클래스 4개를 순서대로 정의 (`BottleneckBlock` → `HourglassModule` → `LinearLayer` → `StackedHourglassNetwork`).

**`BottleneckBlock(in_channels, filters, stride, downsample)`** — 셀 21:
```python
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1, downsample=False):
        super(BottleneckBlock, self).__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_conv = nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, filters // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(filters // 2, momentum=0.9)
        self.conv2 = nn.Conv2d(filters // 2, filters // 2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters // 2, momentum=0.9)
        self.conv3 = nn.Conv2d(filters // 2, filters, kernel_size=1, stride=1, padding=0, bias=False)
    # forward: identity(=downsample_conv) + BN-ReLU-Conv ×3, 마지막에 identity 더함
```
- BN `momentum=0.9`, ReLU는 identity 더한 **뒤**가 아니라 branch 내부. Residual 연결은 `out += identity` (post-activation bottleneck).

**`HourglassModule(order, filters, num_residual)`** — 셀 22:
```python
self.up1_0 = BottleneckBlock(filters→filters)
self.up1_blocks = Sequential(BottleneckBlock × num_residual)
self.pool = MaxPool2d(2, stride 2)
self.low1_blocks = Sequential(BottleneckBlock × num_residual)
if order > 1: self.low2 = HourglassModule(order - 1, ...)   # 재귀
else: self.low2_blocks = Sequential(BottleneckBlock × num_residual)
self.low3_blocks = Sequential(BottleneckBlock × num_residual)
self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
# forward: up1 + upsample(low3(low2(low1(pool(x)))))
```

**`LinearLayer(in, out)`** — 셀 23: `Conv2d(1×1, bias=False) + BN + ReLU`, `kaiming_normal_(fan_out, relu)` 초기화.

**`StackedHourglassNetwork(input_shape=(256,256,3), num_stack=4, num_residual=1, num_heatmap=16)`** — 셀 24:
```python
conv1 = Conv2d(3, 64, k7, s2, p3) → BN → ReLU
bottleneck1 = BottleneckBlock(64, 128, downsample=True)   # 128 채널
pool = MaxPool2d(2, s2)
bottleneck2 = BottleneckBlock(128, 128, downsample=False)
bottleneck3 = BottleneckBlock(128, 256, downsample=True)  # 256 채널
for i in range(num_stack):                                # num_stack=4
    HourglassModule(order=4, filters=256, num_residual=1)
    residual_modules.append(BottleneckBlock(256→256) × 1)
    linear_layers.append(LinearLayer(256, 256))
    heatmap_convs.append(Conv2d(256, num_heatmap, k1))    # k1 conv → heatmap
    if i < num_stack-1:
        intermediate_convs.append(Conv2d(256,256,k1))     # 다음 스택 입력
        intermediate_outs.append(Conv2d(num_heatmap,256,k1))
# forward: conv1→bn1→relu→bottleneck1→pool→bottleneck2→bottleneck3
#   각 스택: hg → res → lin → heatmap_convs (ออกput list, 마지막 스택 미포함 intermediate)
#   heatmap 등록: x_{i+1} = intermediate_convs(lin) + intermediate_outs(heatmap)
```

**입출력 규격 (확정값):**
- 입력 이미지: **256×256×3**, 정규화 `x/127.5 - 1.0` → **[−1, 1]**
- heatmap: **64×64×16** (`HEATMAP_SIZE = (64, 64)`, `IMAGE_SHAPE = (256, 256, 3)` — 셀 26)
- 포워드 출력: **4개 스택 각각 `(B, 16, 64, 64)` 리스트** — `outputs[-1]` (마지막 스택) 사용
- 키포인트 수: **16 (MPII 표준 16개)**, 셀 31에서 순서 명시:
  `R_ANKLE=0, R_KNEE=1, R_HIP=2, L_HIP=3, L_KNEE=4, L_ANKLE=5, PELVIS=6, THORAX=7, UPPER_NECK=8, HEAD_TOP=9, R_WRIST=10, R_ELBOW=11, R_SHOULDER=12, L_SHOULDER=13, L_ELBOW=14, L_WRIST=15` + `MPII_BONES`(15개 뼈 연결) — 셀 31.
- 파라미터 수(실측): **16,251,392** (StackedHourglass, smoke 스크립트 출력)

### 1-b. SimpleBaseline (ResNet-50 백본, 비교 실험용) — 셀 43
```python
self.resnet = Sequential(*list(resnet50(weights=ImageNet).children())[:-2])  # 끝까지 ResNet50
self.reduce_channels = Conv2d(2048, 256, k1)
self.upconv = DeconvLayer(3)   # ConvTranspose2d(256,256,k4,s2,p1) ×3 + BN + ReLU
self.final_layer = Conv2d(256, num_heatmap, k1)
# forward → [x] (list로 반환해 Hourglass와 인터페이스 통일)
```
- `SimpleBaselineTransfer`(셀 52): `freeze_backbone` 옵션(Feature extraction) + 동일 deconv 헤드.

### 1-c. heatmap → 키포인트 후처리 — 셀 32-33
- `find_max_coordinates`: `(H,W,C)` reshape → 채널별 `argmax`, `y = idx // W, x = idx % W` (W 기준 복원 — 사각형 아닌 heatmap 버그 수정됨).
- `extract_keypoints_from_heatmap`: argmax 좌표 + 3×3 패치 이차 최대로 **sub-pixel 보정** `delta = (next-1)/4`, clamp 후 `/H`로 정규화 → `(16,2)` 텐서.

---

## 2. 데이터 준비

- **MPII human pose v1** 다운로드(셀 0): `mpii_human_pose_v1.tar.gz` (이미지) + `mpii_human_pose_v1_u12_2.zip` (annots), train/validation.json(셀 1, cloudfront) — **원본 데이터는 로컬에 없음** (T1 확인).
- `parse_one_annotation`(셀 6): `image/joints/joints_vis/center/scale` → dict.
- `generate_ptexample`(셀 8): image→JPEG bytes, x/y는 `int()`만 변환(검수 수정: -1 invisible 보존), `v = 0 if vis==0 else 2`.
- `ray` 병렬로 `.ptrecords` 샤드 생성(셀 9-16): train 64 shard / val 8 shard.
- `crop_roi`(셀 18): keypoint bbox에 `margin=0.2`(train은 random 0.1~0.3) 확장 크롭, 좌표 0~1 정규화.
- `generate_2d_gaussian` / `make_heatmaps`(셀 19): `sigma=1, scale=12` 가우시안 heatmap 64×64, invisible(vis==0)은 전부 0.
- **`Preprocessor`(셀 20)**: `image_shape=(256,256,3)`, `heatmap_shape=(64,64,16)`; resize 후 `uint8 → /127.5 − 1`. 기본 `heatmap_shape[2]=16`.
- **`MPIIDataset`(셀 13)**: JSON → `(image, heatmaps)` 튜플 반환.
- **`create_dataloader`(셀 26)**: `batch_size`, shuffle=is_train, `num_workers=4, pin_memory=True, prefetch_factor=2`.

---

## 3. 메트릭 로그 (notebook outputs)

- **노트북 셀에 저장된 출력 메트릭 없음** — nbformat 검사 결과 56개 코드 셀 전부 `outputs=[]` (실행 결과 미저장). 따라서 아래 값은 **코드 소스에서 확정된 하이퍼파라미터/손실 정의**이고, 실행 메트릭은 T4 smoke 로그(§4)가 유일한 실측이다.
- **Trainer 손실(원본, 셀 25 & 28 하이퍼파라미터)**: `nn.MSELoss(reduction='none')`, 가중치 `(labels>0)*81 + 1` (positive 위치 82× 가중), `global_batch_size`로 나눔. Epoch 2, batch 16, `lr=0.0007`, Adam. LR decay: patience 10 초과 시 /10, 또는 epoch 25/50/75에서 /10. Best model 저장명 `model-epoch-{epoch}-loss-{loss:.4f}.pt` → **로컬 `~/Downloads/poseEstimate/model-epoch-2-loss-1.2050.pt`** (셀 29 gdown에서 다운로드 — `file_id=1-F6ztKRSL7Lp0kVT7nPuSIfEpBWz1abZ`, epoch 2, val loss 1.2050).
- **최종 비교 실험(셀 39-55)**: `NUM_EPOCHS=5`, `BATCH_SIZE=16`, `NUM_HEATMAP=16`, `LEARNING_RATE=2.5e-4`, weighted MSE(85-87): `comparison_weighted_mse` = 각 스택 heatmap에 대해 `(target>0)*81+1` 가중 MSE, 마지막 스택만이 아닌 **전체 스택 평균**. 두 모델(StackedHourglass vs SimpleBaseline) + Transfer(full FT vs frozen backbone) 비교.
- **PCK/OKS/mAP는 미구현** — 셀 47은 loss/epoch-time/parameter 수만 요약(정량 결과), 시각화는 scatter/line(셀 35, 50). 평가 = **weighted MSE(val) + epoch time + 파라미터 수**.

---

## 4. 재현 smoke 로그

스크립트: `scripts/reconstruct_pose_model.py` (셀 21-24 + 32,33 클래스 1:1 전사)
가중치 로드: `~/Downloads/best_model.pt`(계획 지정) → **실패(정체 불일치)**, `~/Downloads/poseEstimate/model-epoch-2-loss-1.2050.pt` → **성공(전체 strict 로드)**

```
$ KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/venv/bin/python scripts/reconstruct_pose_model.py

[model] StackedHourglassNetwork(num_stack=4, num_residual=1, num_heatmap=16)
[model] parameters: 16,251,392

=== loading /Users/robotjang/Downloads/best_model.pt ===
[load] /Users/robotjang/Downloads/best_model.pt: NOT a pose checkpoint (keys like
  'encoder.embedding.weight'/'decoder.attention.W1.weight' = Seq2Seq translation model).
  SKIP structural load attempt on pose arch.

=== loading /Users/robotjang/Downloads/poseEstimate/model-epoch-2-loss-1.2050.pt ===
[load] /Users/robotjang/Downloads/poseEstimate/model-epoch-2-loss-1.2050.pt: STRICT load OK

[smoke] dummy forward OK: 4 stacks; last heatmap shape = (1, 16, 64, 64) (B, C=num_heatmap, H, W)
[smoke] output channel count == 16 == num_heatmap (16 MPII keypoints)  [PASS]

[image] person.jpg -> predicted keypoints (16 MPII joints, normalized [0,1]):
  R_ANKLE      (0.0039, 0.0000)     R_KNEE       (0.0039, 0.0000)
  R_HIP        (0.0039, 0.0000)     L_HIP        (0.0000, 0.0039)
  L_KNEE       (0.0000, 0.0039)     L_ANKLE      (0.0000, 0.0039)
  PELVIS       (0.0000, 0.0039)     THORAX       (0.5117, 0.4102)
  UPPER_NECK   (0.5156, 0.3789)     HEAD_TOP     (0.5195, 0.0664)
  R_WRIST      (0.4375, 0.5820)     R_ELBOW      (0.3477, 0.6250)
  R_SHOULDER   (0.3555, 0.4062)     L_SHOULDER   (0.6562, 0.3789)
  L_ELBOW      (0.6875, 0.6211)     L_WRIST      (0.4375, 0.5820)
[image] keypoint tensor shape (16, 2) == (16, 2)  [PASS]

[smoke] EXIT 0 - pose model loads and runs inference
```

**해석**: `~/Downloads/person.jpg`(800×533 얼굴/상반신 클로즈업)라 하체 키포인트(R_ANKLE 등)는 heatmap이 비어 corner argmax로 수렴(가시성 없는 관절 → (0,0) 근처). 반면 **상반신 관절은 해부학적으로 타당**: HEAD_TOP(0.52, 0.07)가 가장 위, UPPER_NECK→THORAX 순, 양 어깨(좌 0.66 / 우 0.36)가 퍼져 있고 팔꿈치·손목이 아래로 이어짐. 계획된 `best_model.pt` 로드는 **의도한 자세 가중치가 아니었음** — §아드버서리 노트 참조.

### 가중치 별 로드 결과 요약
| 가중치 | 크기 | 정체(첫 키샘플) | 자세모델 로드 |
|---|---|---|---|
| `~/Downloads/best_model.pt` | 42,310,517 B | `encoder.embedding.weight`/`decoder.attention.W1.weight` (Seq2Seq 번역=LANG 모델) | ❌ 구조 불일치(의도와 다름) |
| `~/Downloads/poseEstimate/model-epoch-2-loss-1.2050.pt` | 65,862,380 B | `conv1.weight`/`bottleneck1.*` (StackedHourglassNetwork) | ✅ **STRICT load OK** |

---

## 아드버서리 노트 (T4/T13 핵심 전달)

1. **`best_model.pt`는 POSE 가중치가 아님.** 키 이름(`encoder.embedding.weight`, `decoder.attention.W1/W2/v`)과 17개 키만 존재 → **Seq2Seq+Bahdanau 한영번역 모델**(T5 LANG 도메인). 실제 동작하는 자세 모델은 `~/Downloads/poseEstimate/model-epoch-2-loss-1.2050.pt` (StackedHourglass, epoch 2, val loss 1.2050).
2. **T13(요가)은 `poseEstimate/model-epoch-2-loss-1.2050.pt` 경로를 사용해야 함** — 계획의 best_model.pt 기반 설계 시 로드 실패로 MediaPipe fallback에 빠질 위험. smoke가 실제 추론까지 통과해 T13 전제 충족.
3. MPII 원본 데이터 로컬 부재(T1 확인) → 재학습/데이터센트릭 데모는 Colab 경유 필요(550 CU). 학습은 안 함(MUST NOT 준수).
4. 노트북 셀 출력(메트릭) 미저장 → 성능 비교 지표는 smoke 로그가 유일 실측. 손실 정의만 소스에서 추출함.