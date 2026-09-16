# Spatial-Multimodal-2Mo: Scene-Graph-Driven Ego Safety Verdict Pipeline on KITTI

## Abstract
We present a spatial reasoning pipeline that fuses scene-graph knowledge extracted from
multi-modal (ego-image + LIDAR + text) KITTI scenes to produce a binary safety verdict
(CANNOT / SAFE) for an autonomous ego vehicle. On the 60-scene validation benchmark the
pipeline currently agrees with gold in only 18.9% of scenes, failing the ≥85% gate. We
isolate the root cause to the CANNOT verdict branch: vulnerable entity / crosswalk / heavy
vehicle blocking relations in the scene graph are not consistently propagated into the
final verdict. We document the benchmark driver, the failure modes, and the exact patch
surface required to recover gate compliance.

## 1. Introduction
Autonomous driving relies on interpretable, graph-based reasoning. Our pipeline ingests a
scene description, builds a scene graph (entities + spatial relations), and reasons over
vulnerable-road-user / blocking-ahead triples to emit CANNOT or SAFE.

## 2. Method
1. Caption generation (multimodal scenes).
2. Object/relation extraction into scene graph.
3. Forward-path hazard analysis (vulnerable ahead, heavy blocking, crosswalk/jaywalk).
4. Verdict step: CANNOT if hazard ahead, else SAFE.

## 3. Benchmark
- 60 KITTI validation scenes (gold labels CANNOT/SAFE per scene).
- Driver: bench_v49.py — HTTP call through a Cloudflare tunnel to the demo endpoint;
  gold from staged reasoning final step / answer hint; verdict from server response tokens.
- Gate: agreement ≥85%.

## 4. Results
| Metric | Value |
|---|---|
| Total scenes | 53 evaluated (7 gold-NONE skipped) |
| Agree | 10 |
| Mismatch | 43 |
| Rate | 18.9% |
| Gate | FAIL (< 85%) |

Mismatch breakdown: CANNOT gold scenes systematically return SAFE; several scenes also
return HTTP 500 (server-side error), which yields no verdict.

## 5. Root-Cause Analysis
The vulnerable-ahead detection in the pipeline requires a specific triple shape
(`is_ahead_of` to ego, or `near_crosswalk` for vulnerable classes). Scenes where the
vulnerable entity jaywalks / occupies the crosswalk / is near a stop line, or where a
Tram/Truck/Van blocks the lane, are not mapped to CANNOT. This is a classification-branch
bug, not a modeling failure.

## 6. Conclusion
The gate is not met. A patch to the verdict branch (treat crosswalk occupancy, stop-line
violation, and heavy-blocking as CANNOT triggers) is required, followed by a server restart
and a full re-benchmark before the gate can pass.
