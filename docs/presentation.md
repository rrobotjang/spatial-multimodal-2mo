# 발표자료 (슬라이드 각 1장 씩)

## 1. 프로젝트명 · 목표
Spatial-Multimodal-2Mo — 장면 그래프 기반 Ego 안전 판정
게이트: 60장면 벤치 정합률 ≥85%

## 2. 파이프라인 아키텍처
KITTI 장면 → 캡션 → (개체, 관계) 장면 그래프 → 전방 위험 분석 → CANNOT/SAFE

## 3. 벤치마크 방법
- 60 검증 장면, gold= CANNOT/SAFE
- 터널 경유 demo 호출, 응답 토큰으로 판정

## 4. 최신 결과
- 정합 10 / 53 = 18.9% — 게이트 미달
- 불일치: CANNOT gold → SAFE 판정 (취약 개체·횡단보도·중차량 분기 미발동)

## 5. 원인
판정 분기 버그: `near_crosswalk`·stop-line·heavy-blocking 조합이 CANNOT으로 전파 안 됨

## 6. 다음 단계 (게이트 회복 로드맵)
1) pipeline.py 판정 분기 패치(취약 전방 → CANNOT)
2) 서버 재시작 → 터널 health 확인
3) bench_v4x 재실행 → 85% 게이트 재확인
4) 통과 시 git push + 논문/발표 최종판 확정

## 7. 결론
현재 게이트 FAIL. 패치→재벤치가 선행되어야 85% 통과 가능.
