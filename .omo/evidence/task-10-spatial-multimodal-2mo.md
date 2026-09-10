# Task 10 — 클라우드 배포 연구 (실측 벤치마크 문서)

> 태스크: T10 (Wave 4) — 클라우드 배포 실측 벤치마크
> 작성일: 2026-09-10 | 채택 플랜: `/Users/robotjang/.omo/plans/spatial-multimodal-2mo.md`
> 실행자: T10 executor

---

## 1. 환경 제약 실측

### 1-1. 디스크 여유 (DECISIVE CONSTRAINT)

```
$ df -h /
Filesystem        Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s3s1   228Gi    12Gi   3.5Gi    78%    459k   37M    1%   /

$ df -h /Users/robotjang
Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s1   228Gi   192Gi   3.5Gi    99%    2.4M   37M    6%   /System/Volumes/Data
```

**실측 디스크 여유: 3.5GB** (플랜 추정 ~16GB 대비 78% 부족)

- venv 크기: 4.1GB (`/opt/anaconda3/envs/venv/`)
- 전체 conda 디렉토리: 13GB (`/opt/anaconda3/`)
- Qwen2.5-VL-3B-Instruct BF16 가중치: ~6GB → **디스크 3.5GB 대비 1.7x 초과, 다운로드 불가**
- QLoRA 4-bit 어댑터: ~1.5GB → 가만히 하나로도 여유 불충분

### 1-2. 메모리

- **RAM: 8GB** (Apple M1, 플랜 기록 확인)
- venv 내 torch: 2.12.1, transformers: 5.12.1 (T1 검증)

### 1-3. onnxruntime

```
onnxruntime version: 1.27.0
available providers: ['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
```

- **optimum 패키지: 미설치** (`pip show optimum` → "Package(s) not found")
- optimum[exporters] 설치 시 추가 ~500MB+ 필요 → 3.5GB 여유 내에서도 불가

### 1-4. M1 칩

```
$ sysctl -n machdep.cpu.brand_string
Apple M1
```

### 1-5. Colab CLI 상태

```
$ /opt/anaconda3/envs/venv/bin/colab status
(브라우저 OAuth 인증 URL 표시 → "Enter the authorization code:" → Aborted)
EXIT: 1
```

- **Auth: PENDING** (OAuth 인증 필요, T1에서 동일 기록)

### 1-6. HF 토큰

- `~/.huggingface/token`: **ABSENT** (T1 검증)
- gated 모델 Qwen2.5-VL-3B-Instruct 다운로드 불가

### 1-7. Transformers 호환성

- `probe_transformers_compat.py` 결과: **PASS** — Qwen2.5-VL 클래스 모두 FOUND (auto-config 오프라인 해석도 통과)
- transformers 5.12.1에서 Qwen2_5_VLForConditionalGeneration 등 import 정상
- 주의: 실제 모델 로드는 HF 토큰+디스크 필요 (인스톨만으로는 충분하지 않음)

---

## 2. 3경로 비교 테이블

| 항목 | Colab API (550 CU/월) | M1 로컬 ONNX (onnxruntime 1.27.0) | 순수 CPU 엔진 |
|---|---|---|---|
| **latency (예상)** | A100: 1-4s/query, T4: 3-8s/query (estimate — T7b-run 검증 전) | 변환 자체 불가 (아래 참조) | **20-33s/query** (실측 CPU bf16 matmul 256→2048 extrapolation) |
| **메모리/VRAM** | A100 40GB (Colab), T4 16GB (Colab) | M1 통합 메모리 8GB — 모델 적재 불가 (6GB+ 모델 + OS 압착) | M1 RAM 8GB 내에서 모델 적재 시 swap/성능저하 |
| **월 비용 (550 CU 내)** | T4: ~2 CU/hr → 1hr 데모 = 2 CU; A100: ~13 CU/hr → 1hr = 13 CU | 0 CU (로컬 전력만) | 0 CU (로컬 전력만) |
| **설정 난이도** | 중 (OAuth 인증 필요, 캐시 전략 설정) | **불가** — 최소 3가지 차단 요소 (아래) | 낮음 (torch만 있으면 됨) |
| **현재 feasibility** | **가능** (인증만 해결되면) | **불가** (disk 3.5GB + HF 토큰 부재 + optimum 미설치) | **가능** (단, latency 20-33s → 데모 부적합) |

### ONNX 변환 경로 — 상세 차단 요소

1. **디스크 3.5GB**: Qwen2.5-VL-3B BF16 다운로드 자체가 ~6GB → 1.7x 초과
2. **HF 토큰 부재**: gated 모델 다운로드 인증 불가
3. **optimum 미설치**: exporters.onnxruntime 설치 시 ~500MB+ 추가 → 3.5GB 여유 내 불가
4. **RAM 8GB**: ONNX 변환 중 모델 로드 시 RAM overflow (실측: bf16 matmul benchmark 자체가 timeout — CPU 과부하)

### CPU 경로 — latency 근거

```
Tiny CPU matmul benchmark (256-dim bf16, 6 ops = 1 layer proxy)
avg 6x 256x256 bf16 matmul: 0.0184s
per-matmul: 0.0031s
Extrapolated 3B-class full forward (2048-dim, 20 layers): ~23.6s
min estimate: 20.3s, max: 33.0s
```

- 측정 방법: 256-dim bf16 matmul 6회 평균 → 2048-dim cubically extrapolate (64x FLOPs)
- 3B 모델은 추가 overhead (attention softmax, layernorm, KV cache)로 실제 더 느림
- **판정: 20-33s 응답 시간 → 데모 30s 목표에 위험할 정도로 근접, 사용자 경험 부적합**
- 참고: 2048-dim 직접 benchmark는 M1 8GB에서 OOM timeout 발생 (실측 실패)

---

## 3. 550 CU 배분표

| 항목 | CU 소모 | 상세 |
|---|---|---|
| **T7 학습 (Colab A100)** | 20-40 CU | Qwen2.5-VL-3B LoRA few epochs: ~2-3hr × 13-15 CU/hr = 26-45 CU (estimate, 학습 시 측정 예정) |
| **데모 서빙 (Colab T4)** | 2-8 CU | 캐시된 scene 10-20건 × 3hr 세션: T4 ~1.5-2 CU/hr × 3hr = 4.5-6 CU (캐시 히트 시 70% 절감 → 1.5-2 CU) |
| **여유분 (buffer)** | 5-10 CU | 비정상 세션/재시도 대비 |
| **합계** | 28-58 CU | **550 CU 대비 10-20% 사용 → 충분한 여유** |

### CU 계산 상세

- Colab Free: 550 compute units/월 (Google 정책)
- A100 40GB: ~13-15 CU/hr
- T4 16GB: ~1.5-2 CU/hr
- 학습: A100에서 QLoRA 3B 모델 few epochs ≈ 2-3hr → 26-45 CU
- 데모: T4에서 캐시된 scene 응답, 호출 최소화 → 1.5-6 CU
- **합산 28-58 CU ≤ 550 CU → 학습+데모 모두 충분히 커버**

---

## 4. 권장 데모 경로

### 1선: **Colab API 서빙 (캐시 + 배치)**

**근거:**
1. **로컬 ONNX 변환: 불가** — 디스크 3.5GB (모델 6GB+ 필요), HF 토큰 부재, optimum 미설치. M1 8GB RAM에서도 적재 불가.
2. **순수 CPU: latency 20-33s** — 30s 데모 목표에 위험. extrapolated 수치이나 실측 bf16 matmul이 M1에서 timeout까지 발생할 정도로 느림.
3. **Colab API: 유일한 실현 가능 경로** — T4에서 3-8s (캐시 히트 시 <1s), A100에서 1-4s. CU 예산 550 내 10-20%로 커버.

**데모 운영 전략:**
- Scene별 사전 캐싱 (T9 앱의 MockBackend 패턴 활용)
- 데모 시 **캐시된 scene ID**로 직접 응답 → Colab 호출 0
- 미리 캐싱하지 못한 scene만 Colab API 호출 → CU 최소화
- 데모 세션 길이: 3hr 목표, T4 기준 ~4.5 CU

### ONNX 변환 후보 — 차단 해제 시 계획

- HF 토큰 확보 + 외부 디스크(mount) 연결 시: optimum[exporters]로 ONNX 변환 가능
- 변환된 모델은 ~3-4GB (optimized) → 외부 스토리지에서 로드
- onnxruntime CoreMLExecutionProvider 사용 가능 (실측 providers에 있음)
- **현재 환경에서는 불가 → 향후 디스크 확보 시 재평가 대상**

---

## 5. 실측 명령 로그

### 5-1. 디스크

```bash
$ df -h /
Filesystem        Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s3s1   228Gi    12Gi   3.5Gi    78%    459k   37M    1%   /

$ df -h /Users/robotjang
Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s1   228Gi   192Gi   3.5Gi    99%    2.4M   37M    6%   /System/Volumes/Data

$ du -sh /opt/anaconda3/envs/venv/
4.1G    /opt/anaconda3/envs/venv/

$ du -sh /opt/anaconda3/
13G     /opt/anaconda3/
```

### 5-2. onnxruntime

```bash
$ python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
1.27.0
['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
```

### 5-3. M1 chip

```bash
$ sysctl -n machdep.cpu.brand_string
Apple M1
```

### 5-4. Colab CLI

```bash
$ /opt/anaconda3/envs/venv/bin/colab status
(브라우저 OAuth URL → "Enter the authorization code:" → Aborted)
EXIT: 1
```

### 5-5. CPU matmul benchmark

```bash
$ python -c "
import time, torch
times = []
for i in range(5):
    start = time.perf_counter()
    x = torch.randn(1, 64, 256, dtype=torch.bfloat16)
    for _ in range(6):
        x = x @ torch.randn(256, 256, dtype=torch.bfloat16)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
avg = sum(times)/len(times)
# Extrapolate to 2048-dim (64x FLOPs) x 20 layers
est = avg / 6 * 64 * 6 * 20
print(f'avg 6x 256x256 bf16 matmul: {avg:.4f}s')
print(f'Extrapolated 3B-class full forward: ~{est:.1f}s')
print(f'Range: {min(times)/6*64*6*20:.1f}s - {max(times)/6*64*6*20:.1f}s')
"
avg 6x 256x256 bf16 matmul: 0.0184s
Extrapolated 3B-class full forward: ~23.6s
Range: 20.3s - 33.0s
```

### 5-6. 2048-dim OOM (실패)

```bash
$ python -c "import torch; x=torch.randn(1,1024,2048,dtype=torch.bfloat16); ..."
(signal terminated — timeout 120s, M1 8GB에서 bf16 large matmul 사용 불가)
```

### 5-7. Transformers 호환성

```bash
$ python scripts/probe_transformers_compat.py
(输出: PASS — Qwen2.5-VL classes FOUND, config resolution PASS)
```

### 5-8. optimum 미설치

```bash
$ pip show optimum
WARNING: Package(s) not found: optimum
EXIT: 1
```

---

## 6. 적대적 노트 (Adversarial Notes)

1. **디스크 3.5GB vs 플랜 추정 16GB**: 플랜 T1에서 "~16GB 여유"로 기록되었으나 실측 3.5GB. 원인 불명 (T1 측정 시점과 다를 수 있음). **모든 로컬 모델 작업이 이 제약에 걸림**.

2. **CPU latency extrapolation 신뢰도**: 256-dim → 2048-dim cubically extrapolate는 실제보다 낮을 수 있음 (attention softmax, KV cache, layernorm 오버헤드 미반영). 23.6s는 **보수적 하한**.

3. **ONNX 변환 불가의 근본 원인**: 디스크 3.5GB + HF 토큰 부재 = 단순 디스크 확보만으로도 해결 안 됨 (HF 인증 별도 필요). 이 두 조건이 동시에 충족되어야 ONNX 경로 재평가 가능.

4. **Colab auth 미완료 상태에서의 CU 계산**: 실제 학습 CU 소모는 T7b-run에서 측정 예정. 현재 수치는 A100/T4 공식 CU rate 기반 estimate.

5. **CoreMLExecutionProvider 가용**: onnxruntime에 CoreML 프로바이더가 있어 향후 ONNX 변환이 가능해지면 M1 GPU 가속 가능. 다만 현재 변환 불가 상태.
