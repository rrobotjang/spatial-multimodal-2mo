# Task 6 Evidence — Spatial Instruction Dataset (Spatial Understanding)

Status: **DONE** — 400 synthesized samples, train/val split, HF `datasets`-loadable
Date: 2026-09-10
Executor: T6 (generate → validate → commit)

Deliverables:
- `data/kitti_scene_train.jsonl` (340 samples)
- `data/kitti_scene_val.jsonl` (60 samples)
- `data/schema.json` (JSON schema)
- `scripts/generate_spatial_dataset.py` (generator)
- `scripts/validate_spatial_dataset.py` (validation)

---

## 1. Schema description

One JSON object per line (JSONL). Every sample contains all required keys:

| Key | Type | Description |
|---|---|---|
| `id` | string | `kitti-scene-<N>` (unique) |
| `image_path` | string | **Placeholder** local path to KITTI image (e.g. `/data/kitti/image_2/006282.png`) — NO images copied/embedded/uploaded (KITTI license) |
| `image_size` | [int, int] | `[H, W]` = `[375, 1242]` (KITTI typical) |
| `entities` | array | `{name, class, bbox:[x1,y1,x2,y2] normalized 0-1, pose?}` — KITTI 8 classes (T2), optional MPII 16-joint pose for persons (T4) |
| `scene_graph` | array | Triplets `{subject, relation, object}` (e.g. `car_1 is_left_of car_2`, `pedestrian_1 near_crosswalk crosswalk_1`, `car_3 approaching ego_vehicle`) |
| `question` | string | English reasoning question (safety / payment / spatial) |
| `answer` | string | Ground-truth English answer |
| `staged_reasoning` | array | 2-4 steps `{step, text}` matching DRScaffold 4 stages: grounding → graph → reasoning → answer |

Full formal schema: `data/schema.json` (JSON Schema draft-07).

## 2. Synthesis method (IMPORTANT disclaimer)

**This is synthesized instruction data, NOT real KITTI label annotations.**

KITTI labels are not local (T1: raw data absent locally; Colab-only download planned in T2/T3). Per plan constraint, NO KITTI images were copied, embedded, or redistributed — `image_path` is a plain string placeholder pointing to the planned Colab download path `/data/kitti/image_2/<frame>.png`.

Synthesis approach:
- Knowledge drawn from T2 (KITTI 8 classes + RetinaNet detections + class distribution), T3 (road/crosswalk semantics), T4 (MPII 16 joints, upper-body reliability high / lower-body low).
- 24 scenario templates encode plausible KITTI urban scenes: crosswalk crossings, jaywalking, traffic-light compliance, tram right-of-way, platoon ordering, occlusion hazards, emergency-vehicle absence, person sitting, misc obstacles, wet-road braking, lane-change gating, etc.
- Bounding boxes are **designed coordinates**: generated to be geometrically consistent with the scene-graph triplets (e.g. if `car_1 is_left_of truck_1`, car_1's bbox center x < truck's). Poses are anatomically plausible standing/crossing MPII keypoint sets.
- ~8% hard negatives: questions with false premises ("what is the cyclist doing" when no cyclist exists; "describe the pedestrian" when none present) that the model must reject.

## 3. Row counts

| Split | File | Rows |
|---|---|---|
| Train | `data/kitti_scene_train.jsonl` | **340** |
| Val | `data/kitti_scene_val.jsonl` | **60** |
| Total | — | **400** |

Train ≥240 ✓ | Val ≥60 ✓ (target ≥300 total exceeded)

## 4. Validation command + output

```
$ /opt/anaconda3/envs/venv/bin/python scripts/validate_spatial_dataset.py
Generating train split: 340 examples [00:00, 24xxx examples/s]
[load_dataset] train: 340 rows, features=['id', 'image_path', 'image_size', 'entities', 'scene_graph', 'question', 'answer', 'staged_reasoning']
  rows missing any required key: 0
  missing by key: {'id': 0, 'image_path': 0, 'image_size': 0, 'entities': 0, 'scene_graph': 0, 'question': 0, 'answer': 0, 'staged_reasoning': 0}
  rows with no entities: 0
[load_dataset] val: 60 rows, features=['id', 'image_path', 'image_size', 'entities', 'scene_graph', 'question', 'answer', 'staged_reasoning']
  rows missing any required key: 0
  missing by key: {'id': 0, 'image_path': 0, 'image_size': 0, 'entities': 0, 'scene_graph': 0, 'question': 0, 'answer': 0, 'staged_reasoning': 0}
  rows with no entities: 0
VALIDATION PASSED
```

- `datasets.load_dataset("json", data_files=...)` opens **both** files; `ds.num_rows` prints 340 / 60.
- Assertions: every row contains all 8 required keys; missing-by-key counts all 0; no entity-less rows; all scene-graph objects resolve to an entity or the `crosswalk_1` / `traffic_light_1` / `stop_line_1` / `ego_vehicle` anchors.

### QA scenario notes — happy path (3 samples print)
`datasets` opens train/val and a representative sample prints (see §5 first-10 ids below). Happy-path rows verified: `kitti-scene-4` (payment decision), `kitti-scene-10` (4-vehicle platoon ordering), `kitti-scene-49` (crosswalk stop) — all fully render `id/image_path/entities/scene_graph/question/answer/staged_reasoning`.

### QA scenario notes — failure path (JSONL parse error → fixed)
Initial generation produced empty `scene_graph: []` for single-entity scenes (traffic-light and stop-line scenarios had no cross-entity triplets → invalid as "scene graph" samples and a potential downstream-trainer parse/feature gap). **Fixed** by adding an `ego_vehicle` anchor: every entity now receives a `near_ego_vehicle` / `is_ahead_of` triplet, so all rows have ≥1 triple. Regression-checked: 0 empty-graph rows across both splits after the fix. Malformed-JSONL guard: generator emits `json.dumps` per line; validation re-parses every line via `load_dataset` (would raise on any malformed line).

## 5. Evidence samples (first 10, in `.omo/evidence/task-6-spatial-multimodal-2mo.jsonl`)

`kitti-scene-1` handed to `kitti-scene-10`, one JSON object per line — schema-identical to the dataset rows (IDs, entities, scene_graph, staged_reasoning included).

## 6. File manifest

| Path | Type |
|---|---|
| `data/kitti_scene_train.jsonl` | dataset (340) |
| `data/kitti_scene_val.jsonl` | dataset (60) |
| `data/schema.json` | schema |
| `scripts/generate_spatial_dataset.py` | generator |
| `scripts/validate_spatial_dataset.py` | validator |
| `.omo/evidence/task-6-spatial-multimodal-2mo.jsonl` | evidence (10 samples) |
| `.omo/evidence/task-6-spatial-multimodal-2mo-report.md` | this report |

## 7. Cleanup / status

- No `.pt`/images committed; `data/` + evidence only.
- T2-T5 evidence/scripts untouched. Plan/gitignore untouched.
- No training run performed (T7 is separate).
- License compliance: KITTI images never copied/embedded — `image_path` placeholders only.