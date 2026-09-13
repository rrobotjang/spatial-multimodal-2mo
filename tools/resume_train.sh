#!/usr/bin/env bash
# =====================================================================
# spatial-multimodal-2mo — QLoRA text-only 완주용 단일 실행기
# (본 이슈 39차 실증: 학습 정상 · save_steps:100이 유일 원인 · 세션 62분캡)
#
# 사용법 (업로드 채널이 정상인 호스트에서):
#   bash tools/resume_train.sh
# =====================================================================
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
CLI="${COLAB_CLI:-$(command -v colab 2>/dev/null || echo colab)}"
echo "[1/4] CLI: $CLI"
SESSION="spatial-resume"

# 세션 존재? 없으면 T4로 생성
if ! colab sessions 2>/dev/null | grep -q "\[$SESSION\]"; then
  colab sessions create --gpu T4 --session "$SESSION" --keep-alive 180 2>&1 | tail -2
fi

# (핵) VM 실 train_lora.py save_steps 100->5 로컬 in-place 패치 + detached 기동
cat > /tmp/patch_launch.py << 'PYEOF'
import pathlib, subprocess, time
def sh(c,t=60):
    return subprocess.run(["bash","-lc",c], capture_output=True, text=True, timeout=t)
# train_lora.py 실재 파일 발견
o=sh("find /content -maxdepth 9 -name train_lora.py -type f 2>/dev/null | head -n 1",80)
f=o.stdout.strip()
assert f, "train_lora.py 없음"
# save_steps 100 -> 5 (in-place, VM에서 직접)
p=pathlib.Path(f)
txt=p.read_text()
txt=txt.replace('"save_steps": 100','"save_steps": 5')
txt=txt.replace('warmup_ratio','warmup_steps')
p.write_text(txt)
print("패치OK save5:", txt.count('"save_steps": 5'), "| 잔존100:", txt.count('"save_steps": 100'))
wd=str(p.parents[1]); outd=f"{wd}/sr_outputs"; pathlib.Path(outd).mkdir(parents=True, exist_ok=True)
sh(f"cd {wd} && nohup python3 -u {f} --mode text-only --epochs 3 --batch-size 1 --grad-accum 16 > {outd}/sr.log 2>&1 & echo $! > {outd}/sr.pid",60)
print("PID:", pathlib.Path(f"{outd}/sr.pid").read_text().strip() if pathlib.Path(f"{outd}/sr.pid").exists() else "?")
time.sleep(150)
print("ckpt:", sh("find /content -maxdepth 9 -type d -name 'checkpoint-*' 2>/dev/null",50).stdout.strip() or "[아직 없음 — 재시도]")
print("tail:", (sh(f"tail -1 {outd}/sr.log",15).stdout or "(모델다운로드)")[-160:])
PYEOF
colab exec --session "$SESSION" --file /tmp/patch_launch.py --timeout 200 2>&1 | tail -5 || echo "[exec 반환없음 — 세션 신규 필요]"
echo "[완료] 확인되면: colab exec --session $SESSION --file /tmp/patch_launch.py 반복"
