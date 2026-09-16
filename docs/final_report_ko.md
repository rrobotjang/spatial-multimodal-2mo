# 최종 보고서 — Spatial-Multimodal-2Mo: 장면 그래프 기반 Ego 전방 안전 판정 파이프라인

## 1. 과제 요약
- 목표: KITTI 검증(60장면)에 대해 `CANNOT`(진행 불가) / `SAFE`(진행 가능) 판정을 내리고, 벤치마크 정합률 ≥85% 게이트 달성.
- 현재 결과: **18.9% (10/53 정합)** — 게이트 미달(FAIL < 85%).

## 2. 시스템 아키텍처
- 스택: NEO4j(장면 그래프) + MLLM(캡션/추론) + FastAPI/uvicorn + Cloudflare 터널.
- 판정 파이프라인(`pipeline/pipeline.py`):
  1) 장면 캡션 생성
  2) 오브젝트/관계 추출 → 장면 그래프 빌드
  3) 그래프에서 ego 전방 취약 도로 사용자(vulnerable) / 중차량 블로킹 탐지
  4) 최종 판정: 취약 개체·차단이 전방에 있으면 `CANNOT`, 아니면 `SAFE`

## 3. 벤치마크 절차
- 벤치 드라이버: `bench_v49.py` — 60장면, 터널 경유 `/api/demo/{scene_id}`.
- GOLD 추출: 각 장면의 staged_reasoning/answer에서 `CANNOT|SAFE` 토큰.
- 판정 추출: 서버 응답 본문에서 `"CANNOT"|"SAFE"` 토큰.
- GATE: 정합률 ≥85% → 통과.

## 4. 결과 분석 (실측)
- GOLD 분포(60장면): CANNOT=47, SAFE=13, NONE=0 (v49 힌트 규칙).
- 정합: CANNOT=10, 불일치=43.
- 비정합 원인 (실측):
  - `HTTP 500`: kitti-scene-70, 372, 164, 336 → 서버 오류로 판정 부재.
  - `gold=CANNOT pred=SAFE`: kitti-scene-345, 190, 209, 106 → 취약 개체/중차량이 전방에 있음에도 판정 분기가 SAFE로 귀결.
- 근본 원인: 파이프라인 step-3의 취약 개체 전방 탐지(`vulnerable_ahead`)가 일부 관계(예: `near_crosswalk` 경유 취약 개체)를 판정으로 연결하지 못함 → step-4가 항상 SAFE 벡도.

## 5. 게이트 판정
- **FAIL**: rate 18.9% < 85%.
- CANNOT 장면 대부분이 SAFE로 오판 — 취약 전방 판정 분기 패치가 선행되어야 재벤치 가능.

## 6. 향후 계획
1. `pipeline.py` step-3/4 패치: 취약 개체가 crosswalk 점유·jaywalk·stop-line 위반 시 `CANNOT`을 내보내도록 분기 보강.
2. HTTP 500 원인 제거(서버 안정화).
3. 터널 홈 서버 재시작 후 bench_v49 재실행 → ≥85% 게이트 달성 시 재보고.

## 7. 실행/재현 명령
```bash
ROOT=/Users/robotjang/spatial-multimodal-2mo
PY=/Users/robotjang/Downloads/transformer/.venv/bin/python
TUN=$(tr -d ' \n' </tmp/opencode/tunnel_v14.txt)
$PY $ROOT/pipeline/pipeline.py # 서버 (uvicorn)
$PY /tmp/opencode/bench_v49.py "$TUN" "$ROOT/data/kitti_scene_val.jsonl"
```

## 8. 정합 실패 단계 진단 (archaeology-style 원인 발굴)
- 벤치 실측(60장면, tunnel 경유)에서 CANNOT 장면 47개 중 대부분이 서버에서 SAFE로 판정됨.
- 근본 원인 지점(pipeline.py):
  - step-3 취약 개체 전방 판정이 `is_ahead_of(to ego)` 관계나 **vulnerable near_crosswalk** 조합에만 반응.
  - `crosswalk 점유`(jaywalking)·`stop line 위반`·`중차량(Tram/Truck/Van) 전방 점유`가 3중 관계 그래프 표현에서 삼중 매칭 실패 → SAFE로 귀결.
- 이는 코드 버그(취약 분기 조건 누락)로 확인 — 판정 로직이 아니라 파이프라인 분기 패치 문제.

## 9. 최종 결론
- 게이트: **FAIL (18.9% < 85%)**.
- 파이프라인 판정부 패치 후 재벤치가 선행되어야 게이트를 통과할 수 있음.
- 본 보고서는 벤치 실측·원인 분석·재현 명령을 담은 결정판이며, 게이트 통과까지는 추가 작업이 필요함.
