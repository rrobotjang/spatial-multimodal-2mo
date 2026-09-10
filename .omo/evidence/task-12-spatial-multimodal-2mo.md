# Task 12 Evidence — US Investor Pitch Package

> 태스크: T12 (Wave 5) — US 피칭 자산 (데모 스토리 + 라이선스 고지 + 실행 데모)
> 작성일: 2026-09-10 | 채택 플랜: `/Users/robotjang/.omo/plans/spatial-multimodal-2mo.md`
> 실행자: T12 executor
> 상태: DONE — 4섹션 완료, 3씬 실측 verdict, 라이선스 원문 검증 완료
> 제약 준수: 라이선스 오인 기재 없음, 미검증 성능 수치 없음, T7 미완료 상태 그대로 서술

---

## 1. 90-second demo script (ENGLISH)

> Presenter reads this exactly while the T9 app (`uvicorn main:app --port 8011`) runs.
> Scene order: `kitti-scene-70` → `kitti-scene-380` → `kitti-scene-336`.
> Stage count: **4 stages × 3 scenes = 12 stage transitions** (Grounding → Scene Graph → Staged Reasoning → Final Answer, per scene).
> Screen cues in [brackets]. Verdict texts below are the pipeline's literal `stage4_answer` output (measured 2026-09-10).

---

**[Screen: app header, 4-stage flow bar visible: 1·Grounding → 2·Scene Graph → 3·Staged Reasoning → 4·Final Answer. Select kitti-scene-70.]**

"Every frame of a drive is a hard question: can my car go, or must it stop? Today I'll show you a four-stage reasoning engine that answers it, on real KITTI street scenes.

[Click Run Pipeline. Stage 1 lights up.]

**Stage one is entity grounding.** The engine parses the scene into objects — pedestrians, cars, cyclists — with positions, not just pixels. This frame, scene 70, grounds a pedestrian and a car.

[Stage 2 lights up.]

**Stage two builds a scene graph.** It connects those entities into spatial relationships — who is near whom, what is ahead of the ego vehicle — four relational links in this scene.

[Stage 3 lights up.]

**Stage three runs staged reasoning.** A safety check fires: a vulnerable road user is ahead of the ego vehicle. The rule engine reasons the way a driver would.

[Stage 4 lights up.]

**Stage four reaches the verdict.** The app returns, literally: *"CANNOT proceed with payment — unsafe. Ego vehicle must stop and yield to vulnerable road user(s)."* The car stops.

[Select kitti-scene-380.]

Now the same pipeline on a calm scene — a car waiting behind the stop line. Single entity, single relation. Stage four reads: *"SAFE — payment may proceed. Road is clear."*

[Select kitti-scene-336.]

Finally, the hard case: three pedestrians crossing as a group — four entities, seventeen relational links. Stage three sees every one of them. The verdict: *"CANNOT proceed with payment — unsafe."* That is how decisions get made — grounding, graph, reasoning, answer — every time, on every frame.

[Pause for handoff.]

The engine behind this is upgrading from deterministic rules to a fine-tuned spatial-aware language model — three billion parameters — so it learns judgment, not just lookups. That's the product. Thank you."

---

## 2. Product story & one-liner

### One-liner (18 words)

> **"Turns single-domain notebook models into one spatial multimodal engine that decides when a self-driving car may safely proceed."**

### Narrative: notebook 4-domain knowledge → spatial multimodal

The product fuses four independent, notebook-level computer-vision knowledge domains into a single spatial reasoning engine (the **DRScaffold 4-stage pipeline**: grounding → scene graph → staged reasoning → final verdict). Each domain contributes one capability — none alone can make a safety decision, together they can:

| Domain | Knowledge extracted (T2–T5, real notebook/smoke metrics only) | Contribution to spatial reasoning |
|---|---|---|
| **DET** — detection (T2) | RetinaNet-ResNet50-FPN (COCO-pretrained), head swapped to **8 KITTI classes** (Car/Van/Truck/Pedestrian/Person_sitting/Cyclist/Tram/Misc). Class imbalance real (obs. Car=3943 vs Cyc.=188). Eval @IoU≥0.5: precision 0.614, recall 0.500, F1 0.551; GO/STOP decision accuracy 80% (8/10). Driving rules: person height >150px → Stop; vehicle max(w,h) >150px → Stop; score threshold 0.35 (eval) / 0.25–0.50 (driving) | Grounding stage: **what** is present and **where** (normalized bboxes, class taxonomy) |
| **SEG** — segmentation (T3) | Binary road segmentation (KITTI semantic class 7). UNet 31.0M / UNet++ 36.6M params. Mean IoU (full test set): **UNet 0.761, UNet++ 0.763**; sample-1 IoU 0.895 / 0.921. Augmentation + RandomSizedCrop; threshold 0.5 | Drivable-area prior: whether an entity sits on the drivable road or off-road |
| **POSE** — pose (T4) | Stacked Hourglass, **MPII 16 keypoints**, 16.25M params, heatmap 64×64×16, keypoints normalized to (x,y) in [0,1]; upper-body joints more reliable than lower-body | **Body state** of road users: standing/walking pose vs. sitting (Person_sitting risk assessment) |
| **LANG** — language (T5) | Ko→En seq2seq with attention (GRU encoder + Bahdanau), 3/3 exact translation matches in smoke; architecture = encoder-decoder + attention | Produces the **instruction-following verdict sentence** (SAFE / CANNOT), human-readable |

Four notebooks of perception + one notebook of language → **one spatial multimodal pipeline** that answers a driving-safety question in English, in four auditable stages. Because the reasoning is staged and exposed (not a black box), an investor or an engineer can see exactly *why* a verdict was reached — the product differentiator versus end-to-end VLM baselines.

> **(평가 메트릭: T7 학습 로그 실측 확정 — 현재 대기)**

> No test-set benchmark numbers are claimed for the fine-tuned adapter or the fused pipeline. The numbers above are T2–T5 notebook/smoke measurements only. End-to-end eval accuracy will be published from the T7 training log when it completes — none are asserted here.

### Runs anywhere (deployment reality, T10/T11 — measured, not claimed)

- Local inline demo: the T9 app is static-served, offline, no external API (MockBackend) — works on a single laptop.
- Measured M1 constraints (T10/T11): 3.5 GB free disk, 8 GB RAM → full 3B BF16 weights (~6 GB) cannot load locally; pure-CPU forward extrapolated at ~20–33 s/query (unsafe for live demo).
- Recommended serving path (T10): **Colab API** (A100 1–4 s / T4 3–8 s per query, pre-cached scenes <1 s), 550 CU/month budget covering T7 training (est. 26–45 CU) + demo serving (est. 1.5–6 CU) ≈ 10–20% of quota. An honest claim: *the demo runs anywhere you can open a browser to a Colab-backed endpoint.*
- T11 secondary path: browser WebGPU/ONNX (Nano-class, 289 MB Q4) keeps on-device deployment viable for the future.

---

## 3. Qwen license notice

### Verified license of Qwen2.5-VL-3B-Instruct

**Qwen RESEARCH LICENSE AGREEMENT** (Release Date: **September 19, 2024**) — **non-commercial / research & evaluation only.**

- **Not** Apache-2.0. The Apache-2.0 variants in the Qwen2.5-VL family are the **7B and 32B** weight sizes. The **3B-Instruct variant used in this project carries the Qwen RESEARCH LICENSE**, as verified directly from:
  - Official model card on Hugging Face: `Qwen/Qwen2.5-VL-3B-Instruct` → `license_name: qwen-research`, `license_link: …/blob/main/LICENSE` (verified 2026-09-10).
  - Full license text: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE
  - Third-party confirmation (DeepLearning.AI "The Batch", 2025-02-12): "Qwen2.5-VL-3B is free for non-commercial uses."
  - Repo README (QwenLM/Qwen2.5-VL, 19.9k★, License field = Apache-2.0 for the *repository*, and the Qwen README's "License Agreement" notes: "All our open-source models, except for the 3B and 72B variants, are licensed under Apache 2.0.")

### Key clauses (verbatim from the RESEARCH LICENSE, and what they mean)

| Clause | Text (verbatim) | Meaning |
|---|---|---|
| §1.i — Non-Commercial | "Non-Commercial" shall mean **for research or evaluation purposes only**. | The 3B weights may be used for R&D and demos — not for a commercial product without a separate license. |
| §2.a — Grant | …under Alibaba Cloud's intellectual property… **FOR NON-COMMERCIAL PURPOSES ONLY**. | Same as above, operative grant. |
| §2.b + §9.a — Commercial path | "If you are commercially using the Materials, you shall request a license from us." / "You shall request a separate license from us, if you use the Materials in ways not expressly agreed to in this Agreement." | Commercialization requires a separate commercial license from Alibaba Cloud. |
| §3 — Redistribution | Must pass along this Agreement + "Notice" file attribution: *"Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved."* + "Built with Qwen" (or "Improved using Qwen") displayed in product docs. | Any derivative distribution must credit Qwen and carry the notice. |
| §7.b — Termination | "We may terminate this Agreement if **you breach any of the terms or conditions** of this Agreement." | Revocation is breach-triggered, not a standalone "competition" clause. |
| §5.c — Termination (patent) | Licenses terminate automatically "if you commence a lawsuit … alleging that the Materials … infringe any intellectual property…". | Patent-assertion against Alibaba kills the license. |

**Accuracy note — two claims the plan brief floated are NOT in the license text, and are therefore not asserted:**
1. There is **no standalone "competition" / "revoke-for-competing" clause** in the Qwen RESEARCH LICENSE (nor in the standard Qwen LICENSE for 72B). Termination triggers in the actual text are breach (§7.b) and patent litigation (§5.c). The strongest practical reservation is the **non-commercial (§1.i / §2.a) restriction itself**.
2. **MIT / Apache-2.0 for the 3B is wrong.** Only the 7B/32B variants are Apache-2.0.

### KITTI dataset license (verified)

**KITTI Vision Benchmark Suite** (Karlsruhe Institute of Technology + Toyota Technological Institute at Chicago) is released under **CC BY-NC-SA 3.0** — **non-commercial research use only**, attribution required, share-alike. Sources verified 2026-09-10: official benchmark site (cvlibs.net/datasets/kitti), Ultralytics dataset docs (`license: CC-BY-NC-SA-3.0`), Kaggle/Dataset Ninja/LanceDB mirrors. This is consistent with the repo's T2/T3 evidence notes.

### Copy-paste license notice block (for pitch decks / appendix)

> **Third-party licenses & notices**
>
> - **Qwen2.5-VL-3B-Instruct** — Qwen RESEARCH LICENSE AGREEMENT (Release Date: September 19, 2024). Non-commercial, research/evaluation use only. *"Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved."* Commercial deployment requires a separate license from Alibaba Cloud. Model documentation references "Built with Qwen" per §4.b. APACHE-2.0 DOES NOT APPLY to this 3B variant (the 7B and 32B variants are Apache-2.0).
> - **KITTI dataset** — CC BY-NC-SA 3.0 (Karlsruhe Institute of Technology; non-commercial, attribution, share-alike).
> - **torchvision RetinaNet-ResNet50-FPN** — weights are BSD-3-Clause (PyTorch); model subject to respective third-party limitations.
> - **T7 training status**: the LoRA adapter on Qwen2.5-VL-3B is **not yet trained** (blocked). All demo verdicts shown in this package were produced by the deterministic `MockBackend` scaffold. End-to-end evaluation metrics will be published from the T7 training log.

---

## 4. 3-scene demo flow

> All verdicts below are **measured outputs** of `/Users/robotjang/spatial-multimodal-2mo/pipeline/pipeline.py` (SpatialPipeline + MockBackend) run at runtime on **2026-09-10** for the three scene IDs below, loaded from `data/kitti_scene_val.jsonl`. Verdict strings are the literal `stage4_answer` values. Not guessed, not from the app's comments.

### Scene 1 — `kitti-scene-70` (the braking case) — verdict: **CANNOT**
| Field | Value (real) |
|---|---|
| Image | `/data/kitti/image_2/005306.png` (375×1242) |
| Question | "Is the pedestrian crossing legally at a crosswalk or jaywalking?" |
| Stage 1 | 2 entities grounded: `pedestrian_1` (Pedestrian), `car_1` (Car) — pedestrian carries full 16-keypoint pose |
| Stage 2 | 4 relation triples (pedestrian ↔ car ↔ ego_vehicle) |
| Stage 4 verdict (literal) | **"CANNOT proceed with payment — unsafe. Ego vehicle must stop and yield to vulnerable road user(s)."** |
| Presenter says | How the demo opener goes. The car must stop — this is the moment the pipeline earns its safety claim. |

> Demo notes: run once; the app caches results (`_result_cache`) so re-runs are instant. This is the scene used in the 90-second script opening.

### Scene 2 — `kitti-scene-380` (the calm case) — verdict: **SAFE**
| Field | Value (real) |
|---|---|
| Image | `/data/kitti/image_2/006635.png` (375×1242) |
| Question | "Is the car positioned behind the stop line at this intersection?" |
| Stage 1 | 1 entity: `car_1` (Car) |
| Stage 2 | 1 relation triple |
| Stage 4 verdict (literal) | **"SAFE — payment may proceed. Road is clear."** |
| Presenter says | Contrast with scene 1 — no vulnerable users, road clear, the engine does not over-stop. Verifies the verdict is state-dependent, not a fixed answer. |

### Scene 3 — `kitti-scene-336` (the dense case) — verdict: **CANNOT**
| Field | Value (real) |
|---|---|
| Image | `/data/kitti/image_2/004368.png` (375×1242) |
| Question | "How many pedestrians are crossing the road, and in which lane are they?" |
| Stage 1 | 4 entities: `pedestrian_1`, `pedestrian_2`, `pedestrian_3` (all with pose) + `car_1` (Car) |
| Stage 2 | 17 relation triples (the densest graph of the three) |
| Stage 4 verdict (literal) | **"CANNOT proceed with payment — unsafe. Ego vehicle must stop and yield to vulnerable road user(s)."** |
| Presenter says | The hard case closes the pitch: three crossing pedestrians, all with full body pose, seventeen spatial links — the reasoning engine sees each of them and stops. Stage-transition count across the whole demo: **12 transitions** (4 stages × 3 scenes). |

### Demo run sequence (T9 app, `uvicorn main:app --port 8011`)
1. `GET /api/scenes` → pick `kitti-scene-70` from the 12 curated scenes.
2. `POST /api/demo/kitti-scene-70` → watch Stage 1→4 cards populate.
3. `POST /api/demo/kitti-scene-380` (SAFE contrast).
4. `POST /api/demo/kitti-scene-336` (dense CANNOT finale).
5. (Optional, instrumental) `POST /api/upload` with a valid image → returns structured `adapter_not_ready` HTTP 503 — an honest, on-screen demonstration that the T7 Qwen adapter is the pending integration, not yet live. Non-image files → HTTP 400. This keeps the pitch credible: the scaffold today, the fine-tuned multimodal adapter is the next, already-allocated step (T7, 550 CU budget, est. 26–45 CU training).

**Status note (must be read to the investor if asked):** the adapter (Qwen2.5-VL-3B LoRA, T7) is **not yet trained**; all demo verdicts above come from the deterministic scaffold. The pitch presents the *architecture and the live 4-stage behavior* as real today, and the *adapter-driven accuracy* as the explicit next step with a fixed measurement plan — never as an existing benchmark.

---

## Adversarial notes (what I caught myself about to fabricate)

1. **Qwen2.5-VL-3B license**: the task brief asserted "Apache-2.0 as of July 2025" and "rewrite, don't say research-only". Direct verification of the official HF LICENSE file proved the opposite: the **3B-Instruct is Qwen RESEARCH LICENSE (non-commercial)**. Writing "Apache-2.0" would have been a license misstatement — the exact failure the plan's Must-NOT list forbids. I state the verified reality and record this discrepancy openly.
2. **"Competition revocation clause"**: both the brief and I assumed a standard "Qwen may revoke for competition" clause. It is **not present** in either the RESEARCH LICENSE or the standard Qwen LICENSE. I almost paraphrased it; instead the notice states the actual termination triggers (§7.b breach, §5.c patent suit) and the stronger real reservation (non-commercial restriction).
3. **Eval metrics**: I was tempted to quote a task-accuracy figure for the pipeline. T7 is not trained; the only honest line is the mandated placeholder `(평가 메트릭: T7 학습 로그 실측 확정 — 현재 대기)`, plus the T2–T5 notebook numbers (which are real measurements) kept strictly labeled as notebook/smoke results.
4. **Scene verdicts**: I did not trust the app comments (scene 70 → "CANNOT", 380 → "SAFE"); I ran `pipeline.pipeline` at runtime and recorded the literal outputs. Scene 336's dense 17-triple graph and 3-pedestrian grounding are measured, not inferred.
5. **KITTI license**: confirmed CC BY-NC-SA 3.0 from the benchmark site + Ultralytics + mirrors; it is non-commercial, so the demo scenes and any generated scene data inherit NC constraints — noted in the notice block.

## Files / cleanup
- NT (new task): `.omo/evidence/task-12-spatial-multimodal-2mo.md` (this file).
- No other files touched/created (no app/, pipeline/, data/, scripts/, plan edits). Nothing staged beyond this file for commit.