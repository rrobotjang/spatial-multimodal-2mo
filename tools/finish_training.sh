#!/usr/bin/env bash
# ============================================================================
# spatial-multimodal-2mo — QLoRA(text-only, Qwen2.5-VL-3B, T4) 완주용 스크립트
# ----------------------------------------------------------------------------
# 진단 확정 (38차/39세션 실증):
#   1) 학습은 정상 가동 (loss 1.71→1.16, GPU 100%, T4 13.9GB, QLoRA text-only)
#   2) Colab 무료 세션은 ~62-64분 하드캡(17+연속, keep_alive 무효) → 66스텝
#      (≈2시간)을 단일 세션으로 완주 불가
#   3) 단일 근본원인: collect_training.sh 未, train_lora.py cfg 기본값
#      "save_steps": 100 > 총 66스텝 → 체크포인트 저장 0건 = 세션마다 전손
#   4) 업로드 채널 500(11+회), 백그라운드 위임 사망(3회) → 작은 exec만 신뢰
# ----------------------------------------------------------------------------
# 사용법 (호스트에서, 1회):
#   bash tools/finish_training.sh [--fresh] [--patch-only]
#
#   이 스크립트는 아래 순서만 자동 수행:
#     A. colab CLI 경로 확보 (anaconda3/venv 등 후보 탐색)
#     B. 세션 목록에서 T4 세션명 자동 캡처 (없으면 기동)
#     C. VM 로컬 train_lora.py 를 exec 채널의 작은(2KB 이하) 파일로 탐색
#     D. VM 로컬에서 sed in-place: "save_steps":100→5, warmup_ratio→warmup_steps
#     E. detached 기동, 2분 후 checkpoint-5 실존 검증
#     F. 세션 사망 후엔 새 세션에서 resume_from_checkpoint 로 재개 (루프)
# ============================================================================
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

find_colab() {
  for c in \
    "/opt/anaconda3/envs/venv/bin/colab" \
    "/opt/anaconda3/bin/colab" \
    "/opt/anaconda3/envs/venv/bin/colab" \
    "$(command -v colab 2>/dev/null)"; do
    [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}
COLAB="$(find_colab)" || COLAB="colab"
echo "colab: $COLAB"
echo "로컬 준비 완료 — 다음 단계는 VM에서 수행 (exec 채널)."
