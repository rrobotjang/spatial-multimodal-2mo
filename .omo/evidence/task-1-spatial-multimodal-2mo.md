# Task 1 — 환경 베이스라인 증거 (spatial-multimodal-2mo)

> 태스크: T1 (Wave 0) — 환경 베이스라인 (conda venv / Colab CLI / HF 토큰 / 로컬 데이터 / 리포 스캐폴드 / QA 도구)
> 작성일: 2026-09-10 | 채택 플랜: `/Users/robotjang/.omo/plans/spatial-multimodal-2mo.md`
> 실행자: T1 executor (오케스트레이터 사전 검증 항목 + 미완 작업 병합)

---

## 1. 컴퓨트 환경

### (a) ML 가상환경 (오케스트레이터 사전 검증 — 재실행 안 함, 사실로 사용)

- 경로: `/opt/anaconda3/envs/venv` (Python 3.14)
- 명령 (증명 커맨드):
  ```
  KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/venv/bin/python -c "import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__)"
  ```
- 버전 (사전 검증 결과):
  - **torch: 2.12.1**
  - **torchvision: 0.27.1**
  - **transformers: 5.12.1**
- 임포트 smoke 통과 (KMP_DUPLICATE_LIB_OK=TRUE 워크어라운드 적용).

## 2. Colab CLI 상태

- 설치는 본 태스크에서 수행 (사전 미설치 확인: `pip show google-colab-cli` → exit 1, "Package(s) not found").
- 설치 명령:
  ```
  /opt/anaconda3/envs/venv/bin/pip install google-colab-cli
  ```
- 설치 결과: **google-colab-cli 0.6.0** 설치 완료 (Apache-2.0, `Successfully installed google-colab-cli-0.6.0 ...`).
  - 의존성: google-auth 2.58.0, google-auth-oauthlib 1.4.1, jupyter-kernel-client 1.0.2, oauthlib 3.3.1, request-oauthlib 2.0.0 등.
- `colab status` 실행 (stdin `/dev/null` — 인터랙티브 차단 없이) 결과:
  - **exit code: 1**
  - 출력 요약: OAuth 브라우저 인증 요청 메시지 — `https://accounts.google.com/o/oauth2/auth?...` 인증 URL 표시 후 "Enter the authorization code:" 대기 → 입력 없이 Aborted.
- **Auth 필요 여부: YES** (컬럼: 구글 계정 OAuth 브라우저 로그인 — `colab auth` → 브라우저 인증 후 코드 입력 필요).
- 마무리: 인터랙티브 로그인 대기 없이 "auth required"로 문서화만 하고 종료 (MUST NOT 준수).

## 3. HuggingFace 토큰

- 확인 방법 (내용은 절대 출력/기록하지 않음 — 존재 여부만):
  ```
  test -e ~/.huggingface/token && echo EXISTS || echo ABSENT
  ```
- 결과: **ABSENT** (`~/.huggingface/token` 파일 없음).
- 영향: gated 모델 `Qwen2.5-VL-3B-Instruct`(T7) 접근은 사용자 HF 토큰 확보 후 진행 필요. 비-gated 자원(T2–T5 공개 데이터/가중치 로드)은 블록 없음.

## 4. 노트북 / 가중치 / 원본 데이터

### 4-1. 플랜 지정 4개 노트북 (`ls ~/Downloads/*.ipynb` → 사전 검증 71개 파일 중)

| 노트북 | 존재 | 크기(파일 정보) |
| --- | --- | --- |
| `KITTI_RetinaNet_Autonomous_Driving_Submission-3.ipynb` | O | ~/Downloads 최상위 |
| `segmentation_project_with_unetpp_overlay.ipynb` | O | ~/Downloads 최상위 |
| `pose_estimation_submission_final_v2_transferChatgpt5.6.ipynb` | O | ~/Downloads 최상위 |
| `Seq2seq와_Attention_프로젝트__Seq2seq으로_번역기_만들기.ipynb` | O | ~/Downloads 최상위 |

### 4-2. 가중치 (`ls -l ~/Downloads/` 사전 검증)

| 파일 | 존재 | 크기 |
| --- | --- | --- |
| `best_model.pt` (POSE, Hourglass/ResNet50) | O | **42,310,517 bytes** (~40.4 MiB) |
| `tut1-model.pt` (LANG, Seq2Seq 튜터 모델) | O | **442,333,429 bytes** (~421.9 MiB) |

### 4-3. KITTI / MPII 원본 데이터 (본 태스크에서 재조사)

- 조사 명령:
  ```
  find ~/Downloads -maxdepth 3 -type d | grep -iE "kitti|mpii|training|testing"
  find ~/Downloads -maxdepth 3 \( -iname "image_2" -o -iname "label_2" -o -iname "odometry" -o -iname "annot*" \)
  find ~/Downloads -maxdepth 3 -iname "*kitti*" -o -iname "*mpii*"
  ls ~/Downloads/poseEstimate ~/Downloads/segmentation ~/Downloads/go_stop ~/Downloads/files* ~/Downloads/drive-download-* ~/Downloads/aiffel-*
  ```
- 결과:
  - **KITTI 원본 데이터셋: 부재 (NOT FOUND)** — `image_2` / `label_2` / `odometry` 디렉토리 구조 없음. 매치는 노트북 파일명(`KITTI_*.ipynb`)뿐, 이미지/라벨 데이터 아님.
  - **MPII 원본 데이터셋: 부재 (NOT FOUND)** — `mpii`/`annot` 디렉토리 없음. `poseEstimate/` 폴더는 노트북+가중치(`model-epoch-2-loss-1.2050.pt`)만 포함.
  - 관련 로컬 데이터는 `go_stop/`(go_*/stop_* 10장), `segmentation/`(seg_model_unet.pth), 방대한 DKTC CSV류(`drive-download-*`, `aiffel-d-lthon-dktc-online-19`) — KITTI/MPII와 무관.
- 판정: 원본 데이터 부재는 **문서화된 발견 사항**. 플랜 폴백 적용 → **Colab 경유 KITTI 공식 다운로드 계획** + 라이선스 고지를 T2/T3/T6/T9에 참조 전달 (리뷰 이슈 #3). MPII도 동일한 폴백(Colab 공식 다운로드, T4-POSE에서).

## 5. 리포 스캐폴드

- 생성 명령:
  ```
  mkdir -p /Users/robotjang/spatial-multimodal-2mo/.omo/evidence
  git -C /Users/robotjang/spatial-multimodal-2mo init
  ```
- `git init` 결과: `Initialized empty Git repository in /Users/robotjang/spatial-multimodal-2mo/.git/` (브랜치: main)
- `.gitignore` (리포 루트 `/Users/robotjang/spatial-multimodal-2mo/.gitignore`) — 플랜 지정 내용 그대로 (8줄):
  ```
  *.bin
  *.pt
  *.pth
  *.safetensors
  *.ckpt
  .ipynb_checkpoints/
  __pycache__/
  *.egg-info/
  node_modules/
  .venv/
  ```
  - 주의: `.pth` 포함 — `improved_lstm_model.pth`, `stanford_dogs_resnet50_cam.pth`류 및 학습 가중치가 커밋에서 제외됨.
- `git status` 확인 (본 파일 작성 후 최종 검증):
  - exit code: 0
  - 출력: `On branch main / No commits yet / Untracked files: .gitignore, .omo/`
  - 커밋 없음 (첫 커밋은 T2 담당 — MUST NOT 준수).
  - `git remote` 추가/푸시 없음 (MUST NOT 준수).
- `/Users/robotjang/spatial-multimodal-2mo/.omo/evidence/` 디렉토리 생성 확인.

## 6. QA 도구 설치 상태

| 도구 | 상태 | 버전 |
| --- | --- | --- |
| fastapi | 사전 검증 (설치됨) | 0.138.0 |
| uvicorn | 사전 검증 (설치됨) | 0.49.0 |
| playwright | **본 태스크 설치** | **1.62.0** |
| playwright chromium | **본 태스크 설치** | cache: `chromium-1234`, `chromium_headless_shell-1234`, `ffmpeg-1011` |

- playwright 설치 명령:
  ```
  /opt/anaconda3/envs/venv/bin/pip install playwright
  /opt/anaconda3/envs/venv/bin/playwright install chromium
  ```
- 결과: `Successfully installed greenlet-3.5.5 playwright-1.62.0 pyee-13.0.1` / `playwright install chromium` **exit 0** (브라우저 캐시 디렉토리에 `chromium-1234`, `chromium_headless_shell-1234` 확인 — T9 Playwright E2E 사용 준비 완료).

---

## 최종 검증 (VERIFY 수트)

```
$ git -C /Users/robotjang/spatial-multimodal-2mo status ; echo $?
On branch main
No commits yet
Untracked files:  .gitignore, .omo/
exit 0

$ ls /Users/robotjang/spatial-multimodal-2mo/.omo/evidence/
task-1-spatial-multimodal-2mo.md

$ pip show google-colab-cli
Name: google-colab-cli  /  Version: 0.6.0  (Apache-2.0)

$ pip show playwright
Name: playwright  /  Version: 1.62.0  (Apache-2.0)
```

## 미완 / 차단 사항

1. **HF 토큰 부재 (ABSENT)** — gated `Qwen2.5-VL-3B-Instruct`(T7 LoRA 파인튜닝 주 주제) 접근 전 사용자 액션 필요: `huggingface-cli login` 또는 `~/.huggingface/token` 생성. T2–T5(공개 데이터/노트북)는 진행 가능.
2. **Colab CLI OAuth 인증 필요 (YES)** — 사용자 액션 필요: `colab auth`(브라우저 OAuth 로그인 + 코드 입력). 550 CU 접근성은 T7(Colab A100/T4 실행)에서 실체화 예정.
3. **KITTI 원본 데이터 부재** — 플랜 폴백 채택: Colab 경유 KITTI 공식 다운로드(라이선스 고지 포함). T2(DET)·T3(SEG)·T6(데이터 설계)·T9(데모)에 참조 전달.
4. **MPII 원본 데이터 부재** — 동일 폴백: Colab 경유 MPII 공식 다운로드. T4(POSE)·T13(보너스 요가)에 참조 전달.
5. (참고) Python 3.14 + transformers 5.12.1의 Qwen2.5-VL 호환성은 T7에서 Colab 검증 후 pin (플랜 리뷰 #4 — 로컬 제약 아님).