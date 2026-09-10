# Task 11 — 모바일/브라우저 배포 연구 (실측 벤치마크 문서)

## 1. 환경 실측

| 항목 | 실측값 | 근거 |
|------|--------|------|
| 디스크 여유 (`/`) | **3.5 GB** (228 GB 중 12 GB 사용) | `df -h /` 실측 |
| 칩 | Apple M1 | `sysctl -n machdep.cpu.brand_string` |
| RAM | 8 GB (plan 기준) | plan T1 고지 |
| MPS availability | **UNKNOWN** — `torch` 미설치 (global python3) | `pip3 show torch` → not found |
| onnxruntime | **미설치** (global python3) | `pip3 show onnxruntime` → not found |
| mlx / mlx-lm | **미설치** | `pip3 show mlx mlx-lm` → not found |
| numpy | 2.4.6 | `pip3 show numpy` |
| Matmul GFLOP/s (M1 CPU, float32) | **~138 GFLOP/s** | numpy 1024×1024 matmul ×5 = 0.078s → (2·1024³·5)/0.078/1e9 |
| T9 앱 정적 파일 | vanilla JS (`app.js` 206행, `index.html`, `styles.css`) — 프레임워크·빌드 없음 | `app/static/` 디렉토리 확인 |

> **핵심 제약**: 디스크 3.5 GB — plan 추정 16 GB 대비 **78% 부족**. 모델 다운로드 시 디스크 초과 고위험.

## 2. 경로별 벤치마크 표

| 경로 | 모델 크기 | 로드 시간 | Latency 추정 | 모바일/데스크톱 적합성 |
|------|-----------|-----------|-------------|----------------------|
| **(a) WebGPU ONNX Runtime Web / Transformers.js** | Nano-460M Q4: **289 MB** | 브라우저 fetch + WASM 초기화: **est. 3-8초** (네트워크 의존) | VisionPsy-Nano-460M 기준: est. **0.5-2초/image** (WebGPU); LFM2.5-VL-1.6B ONNX WebGPU 데모 존재 (HF Space) — 데스크톱 WebGPU 지원 시 **실증** | **★★★ 데스크톱: O / 모바일: △** (WebGPU 지원 브라우저 한정, iOS Safari 18+ 제한적) |
| **(b) GGUF Q4_K_M + llama.cpp WASM (wllama)** | 3B Q4_K_M: **~1.8 GB** / Nano-460M Q4: **289 MB** | wllama WASM 초기화 + GGUF 스트리밍 로드: **est. 5-15초** (3B), **2-5초** (Nano) | Nano-460M Q4: est. **0.3-1초/token** (WASM CPU); 3B Q4: est. **1-3초/token** (WASM CPU, 느림) | **★★☆ 데스크톱: △ (Nano 택할 시) / 모바일: △** (wasm Threads 제한, 배터리 소모) |
| **(c) MLX (M1 전용)** | mlx 미설치; M1 8GB RAM 제약 | 미측정 (설치 불가 — mlx 미설치 + disk 3.5GB로 모델 로드 불가) | 8GB RAM 한계: 3B bf16(6GB) → **적재 불가**; 3B Q4(1.8GB) → 시스템+모델 합계 ~5GB → **가능하나 여유 3GB 미만, 스왑 고위험** | **★☆☆ 데스크톱 M1: △ (Nano만 현실적) / 모바일: X** (M1 전용, 브라우저 불가) |

### 로드 시간·Latency 근거

- **WebGPU/ORT-Web**: 실제 WebGPU 런타임은 브라우저에서만 동작 — 로컬 측정 불가. desk research 기반:
  - VisionPsy-Nano-460M (460M params, SigLIP2+SmolLM2, Apache-2.0): Q4_K_M 289 MB → 브라우저 다운로드可行
  - LFM2.5-VL-1.6B ONNX WebGPU 데모: HuggingFace Space에서 공개 데모 존재 (desk research 확인)
  - WebGPU compute: M1 Mac Safari 18+ / Chrome 113+ 지원
- **wllama/GGUF**: llama.cpp WASM은 SIMD + pthreads 기반. 3B 모델은 WASM에서 단일 스레드 시 **1-3초/token** (CPU GFLOP/s ~138 기반 추정: 3B FLOP/token ≈ 6B → 6/138 ≈ 43ms/token理想, but WASM 오버헤드 10-50x → 실질 0.5-3초)
- **MLX**: mlx 미설치 상태. M1 8GB RAM에서 3B bf16(6GB) 로드 시 시스템 메모리 소진 → OOM 예상. 3B Q4(1.8GB)는 이론적이나 실측 불가 (모델 파일 미존재, 디스크 3.5GB로 다운로드 불가)

### GFLOP/s 기반 Latency 추정 공식

```
M1 CPU measured: ~138 GFLOP/s (numpy float32)
模型 3B: ~6 GFLOP/token (forward pass)
이deal CPU latency: 6/138 ≈ 43ms/token
WASM overhead (10-50x): 430ms - 2.15초/token
WebGPU (GPU 가속, 5-10x CPU): 4.3 - 8.6ms/token理想 → 실질 20-50ms/token
```

## 3. 추천 1선

### **WebGPU ONNX Runtime Web + GGUF Q4_K_M (Nano-460M, 289 MB)**

**근거:**

1. **T9 앱이 이미 vanilla JS + 정적 파일** → 수정 최소화 (index.html/app.js/styles.css에 ORT-Web 스크립트 추가만으로 WebGPU 추론 가능)
2. **M1 8GB + 디스크 3.5GB 제약** → 로컬 MLX 대용량 모델(3B+) 적재·변환 **현실적 불가** (disk 78% 부족, RAM 스왑 고위험)
3. **네이티브 앱 금지 (plan Must NOT)** → MLX M1 전용 경로는 폐쇄적, 브라우저 데모 불가
4. **Nano-460M Q4 289MB** → 브라우저 캐시 + CDN 전제 시 모바일/데스크톱 모두 **1회 로드 후 캐시 재사용 가능**
5. **LFM2.5-VL-1.6B WebGPU 데모가 HF Space에 존재** → 기술 실증 완료 (desk research)

### 차선 (secondary): MLX + Nano-460M (M1 데스크톱 한정)

- mlx 설치 + Nano-460M GGUF 변환 시 M1에서 **로컬 실행 가능** (모델 크기 289MB, RAM 여유 충분)
- 단점: M1 전용, 브라우저 데모 불가, 피치 데모(어디서든)와 불일치

## 4. 명령 로그

```bash
# 환경 실측
$ df -h /
Filesystem        Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s3s1   228Gi    12Gi   3.5Gi    78%    459k    37M    1%   /

$ sysctl -n machdep.cpu.brand_string
Apple M1

$ python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# ModuleNotFoundError: No module named 'torch'

$ pip3 show mlx mlx-lm
# WARNING: Package(s) not found: mlx, mlx-lm

$ pip3 show torch
# WARNING: Package(s) not found: torch

# Matmul benchmark (M1 CPU float32)
$ python3 -c "import numpy as np; import time; N=1024; a=np.random.randn(N,N).astype('float32'); b=np.random.randn(N,N).astype('float32'); s=time.time(); [exec('c=a@b') for _ in range(5)]; print(f'{(time.time()-s):.3f}s'); print(f'{2*N**3*5/(time.time()-s)/1e9:.1f} GFLOP/s')"
# 0.078s, ~138.3 GFLOP/s (actual measured)

# T9 앱 정적 파일 확인
$ ls app/static/
app.js  index.html  styles.css
# vanilla JS, no build step, no framework → web-deployable
```

## 5. QA Failure 시나리오

- **disk 3.5GB (plan 16GB 대비 78% 부족)**: 모델 다운로드 시 디스크 초과 → WebGPU 브라우저 캐시 경로로 **유일화**
- **torch 미설치 / MPS 미확인**: 로컬 MLX 경로 실측 불가 → desk research + GFLOP/s 추정에 의존
- **mlx 미설치**: MLX 경로 완전 비실측 → 표에서 "미측정" 명시

> **결론: 디스크 3.5GB + RAM 8GB + 네이티브 금지 = WebGPU 브라우저 경로가 유일한 실증 가능 경로.**
