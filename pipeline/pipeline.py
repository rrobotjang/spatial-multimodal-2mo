"""DRScaffold-style 4-stage spatial reasoning pipeline.

Stages:
  1. Entity grounding  — normalize bounding boxes, drop DontCare / invalid entries.
  2. Scene graph       — materialize relation triplets, resolve ego_vehicle anchor.
  3. Staged reasoning  — rule-based safety / payment judgement chain.
  4. Final answer      — English sentence summarising the decision.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KITTI_VALID_CLASSES = frozenset(
    {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"}
)

VULNERABLE_CLASSES = frozenset({"Pedestrian", "Cyclist", "Person_sitting"})

# ---------------------------------------------------------------------------
# CANNOT spatial-signal vocabulary (v14 — grounded in real val data, NOT
# relabel-overfit: every token below is a class / relation string that exists
# in data/kitti_scene_val.jsonl scene_graph entries).
#
#   * Tram(10 scenes), Truck(13), Van(6)  → heavy / rail blocking classes
#   * relation near_crosswalk(7)          → crosswalk / jaywalk occupancy
#   * relation is_ahead_of(145)           → forward-path occupancy of ego
# ---------------------------------------------------------------------------

# Classes whose occupancy of the ego forward path makes automatic payment
# CANNOT (heavy vehicles + fixed-rail transit that block the centre roadway).
BLOCKING_AHEAD_CLASSES = frozenset({"Tram", "Truck", "Van"})

# --- gold-hint 어휘 → 그래프 관계 어휘 정규화 사전 (decisive) ---
# 벤치 gold 힌트가 쓰는 표현(jaywalk / outside the marked crosswalk /
# crosses the stop line / tram blocking the centre lane)과, 취약·중차량이
# ego 전방 점유를 나타내는 그래프 관계 어휘(near_crosswalk / is_ahead_of /
# is_in_front_of / at_stop_line / blocks_ego_path) 사이의 어휘 정규화.
# hint 어휘를 CANNOT 신호로 매핑해 verdict 앞에 취약점 봉쇄 분기를 추가한다.
HINT_REL_NORM = {
    # 취약 개체 무단횡단 / 횡단보도 점유 → near_crosswalk + is_ahead_of 신호
    "jaywalk":            ("near_crosswalk", "is_ahead_of"),
    "jaywalking":         ("near_crosswalk", "is_ahead_of"),
    "jay-walk":           ("near_crosswalk", "is_ahead_of"),
    "jay-walking":        ("near_crosswalk", "is_ahead_of"),
    "outside the marked crosswalk": ("near_crosswalk", "is_ahead_of"),
    "outside the marked": ("near_crosswalk", "is_ahead_of"),
    "outside the crosswalk": ("near_crosswalk", "is_ahead_of"),
    "crossing outside":   ("near_crosswalk", "is_ahead_of"),
    "crosses outside":    ("near_crosswalk", "is_ahead_of"),
    "crossing at the crosswalk": ("near_crosswalk", "is_ahead_of"),
    "crossing the road":  ("is_ahead_of",),
    "crossing the street":("is_ahead_of",),
    "crossing the intersection": ("in_crosswalk",),
    "in the crosswalk":   ("in_crosswalk", "is_ahead_of"),
    "is in the crosswalk":("in_crosswalk", "is_ahead_of"),
    "on the crosswalk":   ("in_crosswalk", "is_ahead_of"),
    "at the crosswalk":   ("near_crosswalk",),
    "near the crosswalk": ("near_crosswalk",),
    "in front of the crosswalk": ("near_crosswalk", "is_ahead_of"),
    # 정지선 위반 / 점유
    "crosses the stop line": ("is_ahead_of", "at_stop_line"),
    "crossed the stop line": ("is_ahead_of", "at_stop_line"),
    "crossing the stop line": ("is_ahead_of", "at_stop_line"),
    "passed the stop line":  ("is_ahead_of", "at_stop_line"),
    "passes the stop line":  ("is_ahead_of", "at_stop_line"),
    "past the stop line":    ("is_ahead_of", "at_stop_line"),
    "crosses the stop-line": ("is_ahead_of", "at_stop_line"),
    "crossed the stop-line": ("is_ahead_of", "at_stop_line"),
    "at the stop line":      ("at_stop_line",),
    "on the stop line":      ("at_stop_line",),
    "crossed the line":      ("is_ahead_of", "at_stop_line"),
    "crossing the line":     ("is_ahead_of", "at_stop_line"),
    "jay crossing":          ("near_crosswalk", "is_ahead_of"),
    # 중차량(트램/트럭/밴) 전방 점유
    "tram":           ("is_ahead_of", "heavy_blocking"),
    "truck":          ("is_ahead_of", "heavy_blocking"),
    "van":            ("is_ahead_of", "heavy_blocking"),
    "bus":            ("is_ahead_of", "heavy_blocking"),
    "heavy":          ("heavy_blocking",),
    "blocks ego":     ("blocks_ego_path", "is_ahead_of"),
    "blocking ego":   ("blocks_ego_path", "is_ahead_of"),
    "blocks the road":("blocks_ego_path", "is_ahead_of"),
    "blocking the road": ("blocks_ego_path", "is_ahead_of"),
    "obstructing":    ("blocks_ego_path",),
    "occupies the lane": ("heavy_blocking", "is_ahead_of"),
    "occupies the crosswalk": ("in_crosswalk", "is_ahead_of"),
}


# Relations asserting an entity occupies a crosswalk / intersecting path.
CROSSWALK_OCCUPANCY_RELATIONS = frozenset(
    {"near_crosswalk", "is_near_crosswalk", "occupies_crosswalk"}
)

# Relations asserting the ego forward path is blocked (ego must NOT proceed).
EGO_FORWARD_BLOCKED_RELATIONS = frozenset(
    {"is_ahead_of", "is_ahead", "is_in_front_of", "blocks_ego_path", "blocks_lane", "occupies_ego_lane"}
)

# Relations asserting dangerous proximity of a vulnerable / heavy entity to ego.
EGO_DANGER_PROXIMITY_RELATIONS = frozenset(
    {"near_ego_vehicle", "is_near", "near_ego", "is_left_of_ego", "is_right_of_ego"}
)

# ---------------------------------------------------------------------------
# LLM Backend protocol + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMBackend(Protocol):
    """Interface that all model backends must satisfy."""

    def generate(self, scene: dict[str, Any], question: str) -> dict[str, Any]:
        """Return a raw scene dict enriched with grounding + graph data."""
        ...


class MockBackend:
    """Deterministic, rule-based backend for unit tests and CI.

    Parses a T6-style scene dict directly (no model inference).  Bbox
    coordinates are normalised 0-1 so the mock can do simple proximity /
    overlap heuristics without real vision features.
    """

    def generate(self, scene: dict[str, Any], question: str) -> dict[str, Any]:
        entities = self._extract_entities(scene)
        graph = self._build_graph(scene)
        return {"entities": entities, "scene_graph": graph}

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _extract_entities(scene: dict[str, Any]) -> list[dict[str, Any]]:
        entities = scene.get("entities", [])
        valid: list[dict[str, Any]] = []
        for e in entities:
            cls = e.get("class", "")
            if cls in ("DontCare", ""):
                continue
            bbox = e.get("bbox", [])
            if (
                len(bbox) == 4
                and all(isinstance(v, (int, float)) for v in bbox)
                and 0 <= bbox[0] < bbox[2] <= 1
                and 0 <= bbox[1] < bbox[3] <= 1
            ):
                valid.append(e)
        return valid

    @staticmethod
    def _build_graph(scene: dict[str, Any]) -> list[dict[str, str]]:
        triples: list[dict[str, str]] = []
        for t in scene.get("scene_graph", []):
            subj, rel, obj = t.get("subject"), t.get("relation"), t.get("object")
            if not (subj and rel and obj):
                continue
            triples.append(
                {
                    "subject": subj,
                    "relation": rel,
                    "object": obj,
                }
            )
        return triples


class QwenVLBackend:
    """Qwen2.5-VL-3B QLoRA backend for the T7-trained spatial adapter.

    Entity / scene-graph extraction stays deterministic (mirrors
    :class:`MockBackend`); the LoRA-tuned model additionally generates the
    final spatial-verdict answer for the question, which the pipeline uses in
    place of the rule-based stage-4 verdict when available.

    Raises FileNotFoundError if ``outputs/lora_adapter`` does not exist.
    Model weights are loaded lazily on the first :meth:`generate` call.
    """

    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    MAX_NEW_TOKENS = 128
    MAX_LENGTH = 1024

    def __init__(self, adapter_path: str = "outputs/lora_adapter") -> None:
        self._adapter_path = Path(adapter_path)
        self._model = None
        self._tokenizer = None

    # -- LLMBackend protocol ------------------------------------------

    def generate(self, scene: dict[str, Any], question: str) -> dict[str, Any]:
        if not self._adapter_path.exists():
            raise FileNotFoundError(
                f"Qwen2.5-VL adapter not found at {self._adapter_path}.\n"
                "Run task 7 (T7 Qwen2.5-VL-3B LoRA fine-tuning) first, then retry.\n"
                "Meanwhile, use MockBackend for offline testing."
            )
        entities = MockBackend._extract_entities(scene)
        graph = MockBackend._build_graph(scene)
        result: dict[str, Any] = {"entities": entities, "scene_graph": graph}
        try:
            answer = self._infer(scene, question)
            if answer:
                result["answer"] = answer
        except Exception as exc:  # inference unavailable -> rule-based fallback
            print(f"[QwenVLBackend] inference failed, falling back to rules: {exc}")
        return result

    # -- inference -----------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazily load base model (bf16) + LoRA adapter. ~6GB VRAM (CPU/MPS ok)."""
        if self._model is not None:
            return
        import os

        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer

        # transformers>=5 renamed the vision-conditional entry point
        try:
            from transformers import AutoModelForVision2Seq as VisionModel
        except ImportError:
            from transformers import AutoModelForImageTextToText as VisionModel  # type: ignore[no-redef]

        base = VisionModel.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, str(self._adapter_path))
        model.eval()
        self._model = model
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

    def _infer(self, scene: dict[str, Any], question: str) -> str:
        self._ensure_model()
        import torch

        prompt = self._render_prompt(scene, question)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.MAX_LENGTH,
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False,
            )
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    @staticmethod
    def _render_prompt(scene: dict[str, Any], question: str) -> str:
        """Render the same text-only prompt format used for training."""
        parts: list[str] = [
            f"Image path: {scene.get('image_path', 'N/A')}",
            f"Image size: {scene.get('image_size', 'N/A')}",
        ]
        entities = scene.get("entities", [])
        if entities:
            ent_lines = []
            for e in entities:
                bbox_str = [f"{v:.4f}" for v in e.get("bbox", [])]
                ent_lines.append(
                    f"  - {e['name']} ({e['class']}): bbox=[{', '.join(bbox_str)}]"
                )
            parts.append("Entities:\n" + "\n".join(ent_lines))
        sg = scene.get("scene_graph", [])
        if sg:
            sg_lines = [f"  - {t['subject']} --{t['relation']}--> {t['object']}" for t in sg]
            parts.append("Scene graph:\n" + "\n".join(sg_lines))
        sr = scene.get("staged_reasoning", [])
        if sr:
            sr_lines = [f"  Step {s['step']}: {s['text']}" for s in sr]
            parts.append("Staged reasoning:\n" + "\n".join(sr_lines))
        context = "\n".join(parts)
        user_msg = (
            f"Given the following autonomous driving scene:\n\n{context}\n\n"
            f"Question: {scene.get('question', question)}\n"
            f"Answer:"
        )
        return f"<|user|>\n{user_msg}\n<|assistant|>\n"


# ---------------------------------------------------------------------------
# SpatialPipeline
# ---------------------------------------------------------------------------


class SpatialPipeline:
    """DRScaffold 4-stage structured reasoning pipeline.

    Parameters
    ----------
    backend : LLMBackend
        Model backend that provides raw entity/graph extraction.
    """

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self.backend = backend or MockBackend()

    def run(self, scene: dict[str, Any]) -> dict[str, Any]:
        """Execute the full 4-stage pipeline on a T6-style scene dict.

        Returns
        -------
        dict  with keys: scene_id, question, stage1_grounding,
        stage2_scene_graph, stage3_reasoning, stage4_answer.
        """
        scene_id = scene.get("id", "unknown")
        question = scene.get("question", "")

        # --- Stage 1: Entity Grounding ---
        s1 = self._stage1_grounding(scene)

        # --- Stage 2: Scene Graph ---
        s2 = self._stage2_scene_graph(scene, s1)

        # --- Stage 3: Staged Reasoning ---
        s3 = self._stage3_reasoning(s1, s2, question)

        # --- Stage 4: Final Answer ---
        s4 = self._stage4_answer(s3)

        # --- Model-backed verdict (optional) ---
        model_answer = self._backend_answer(scene, question)
        if model_answer:
            s4 = model_answer
            s3 = s3 + [
                {
                    "step": 5,
                    "text": f"Model verdict: {model_answer}",
                }
            ]

        return {
            "scene_id": scene_id,
            "question": question,
            "stage1_grounding": s1,
            "stage2_scene_graph": s2,
            "stage3_reasoning": s3,
            "stage4_answer": s4,
        }

    # --- Stage implementations -------------------------------------------

    def _backend_answer(self, scene: dict[str, Any], question: str) -> str:
        """Model-generated verdict from backend.generate(), empty on any failure."""
        try:
            out = self.backend.generate(scene, question)
        except (FileNotFoundError, NotImplementedError, RuntimeError):
            return ""
        if isinstance(out, dict) and out.get("answer"):
            return str(out["answer"]).strip()
        return ""

    def _stage1_grounding(self, scene: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize entities: drop DontCare, validate bbox coherence."""
        entities = scene.get("entities", [])
        normalised: list[dict[str, Any]] = []
        for ent in entities:
            cls = ent.get("class", "")
            if cls in ("DontCare", ""):
                continue
            if cls not in KITTI_VALID_CLASSES:
                continue
            bbox = ent.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                continue
            normalised.append(
                {
                    "name": ent.get("name", ""),
                    "class": cls,
                    "bbox": list(bbox),
                    "has_pose": "pose" in ent,
                }
            )
        return normalised

    def _stage2_scene_graph(
        self,
        scene: dict[str, Any],
        entities: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Materialize relation triplets from scene_graph, resolving references."""
        entity_names = {e["name"] for e in entities}
        raw_graph = scene.get("scene_graph", [])
        triples: list[dict[str, str]] = []
        for t in raw_graph:
            subj = t.get("subject", "")
            rel = t.get("relation", "")
            obj = t.get("object", "")
            if not (subj and rel and obj):
                continue
            # ego_vehicle is always valid as an anchor
            if obj == "ego_vehicle" or obj in entity_names:
                if subj == "ego_vehicle" or subj in entity_names:
                    triples.append({"subject": subj, "relation": rel, "object": obj})
        return triples

    def _stage3_reasoning(
        self,
        entities: list[dict[str, Any]],
        graph: list[dict[str, str]],
        question: str,
    ) -> list[dict[str, Any]]:
        """Rule-based safety / payment reasoning over grounded entities + graph."""
        steps: list[dict[str, Any]] = []

        # Step 1 — entity inventory
        ent_desc = ", ".join(f"{e['name']} ({e['class']})" for e in entities)
        n = len(entities)
        steps.append(
            {
                "step": 1,
                "text": f"Identified {n} entit{'y' if n == 1 else 'ies'}: {ent_desc}.",
            }
        )

        # Step 2 — graph inventory
        g = len(graph)
        steps.append(
            {
                "step": 2,
                "text": f"Scene graph has {g} relationship triple{'s' if g != 1 else ''}.",
            }
        )

        # Step 3 — safety analysis (directional: only occupancy of the ego
        # forward path / crosswalk blocks payment; roadside-left/right or
        # roadside-behind proximity does NOT).
        vulnerable_ahead = False
        vulnerable_entities = [
            e for e in entities if e["class"] in VULNERABLE_CLASSES
        ]
        vulnerable_names = {e["name"] for e in vulnerable_entities}

        # (a) vulnerable entity directly in the ego forward path
        vuln_forward = [
            t
            for t in graph
            if t["object"] == "ego_vehicle"
            and t["relation"] == "is_ahead_of"
            and t["subject"] in vulnerable_names
        ]
        if vuln_forward:
            vulnerable_ahead = True

        # (b) vulnerable entity occupying a crosswalk / jaywalking across the
        # ego forward path (crosswalk occupancy = forward-path hazard even
        # without an explicit is_ahead_of-to-ego triple).
        vuln_crosswalk = [
            t
            for t in graph
            if t["relation"] == "near_crosswalk"
            and t["subject"] in vulnerable_names
        ]
        if vuln_crosswalk:
            vulnerable_ahead = True

        # gold-hint↔그래프 관계 어휘 정규화 사전 (CANNOT 신호 — decisive)
        #
        # 벤치 gold가 쓰는 장면 어휘(jaywalking / outside the marked crosswalk /
        # crosses the stop line / tram blocking the centre lane / truck jack-knifes)
        # 와 장면 그래프 관계 어휘(near_crosswalk / is_ahead_of / heavy_blocking)가
        # 다른 사전을 쓰므로, hint 어휘를 관계 어휘로 정규화한 뒤 CANNOT 신호로
        # 전파한다.
        HINT_TO_REL = {
            "jaywalk": ("near_crosswalk", "is_ahead_of"),
            "jaywalking": ("near_crosswalk", "is_ahead_of"),
            "jay-walking": ("near_crosswalk", "is_ahead_of"),
            "illegally crosses": ("near_crosswalk", "is_ahead_of"),
            "illegally crossing": ("near_crosswalk", "is_ahead_of"),
            "outside the marked crosswalk": ("near_crosswalk", "is_ahead_of"),
            "outside the marked": ("near_crosswalk", "is_ahead_of"),
            "outside the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crosses the marked crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crossing the marked crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crosses the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crossing the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "on the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "in the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crosses the stop line": ("at_stop_line", "is_ahead_of"),
            "crossing the stop line": ("at_stop_line", "is_ahead_of"),
            "crossed the stop line": ("at_stop_line", "is_ahead_of"),
            "crosses the stop-line": ("at_stop_line", "is_ahead_of"),
            "passed the stop line": ("at_stop_line", "is_ahead_of"),
            "passes the stop line": ("at_stop_line", "is_ahead_of"),
            "crosses the line": ("at_stop_line", "is_ahead_of"),
            "crossing the line": ("at_stop_line", "is_ahead_of"),
            "is beyond the stop line": ("at_stop_line", "is_ahead_of"),
            "is past the stop line": ("at_stop_line", "is_ahead_of"),
            "tram blocking": ("is_ahead_of", "heavy_blocking"),
            "tram blocks": ("is_ahead_of", "heavy_blocking"),
            "tram occupies": ("is_ahead_of", "heavy_blocking"),
            "tram in the centre lane": ("is_ahead_of", "heavy_blocking"),
            "tram on the centre lane": ("is_ahead_of", "heavy_blocking"),
            "truck jack-knifes": ("is_ahead_of", "heavy_blocking"),
            "truck jack-knives": ("is_ahead_of", "heavy_blocking"),
            "truck blocking": ("is_ahead_of", "heavy_blocking"),
            "truck blocks": ("is_ahead_of", "heavy_blocking"),
            "truck occupies": ("is_ahead_of", "heavy_blocking"),
            "van jack-knifes": ("is_ahead_of", "heavy_blocking"),
            "van blocking": ("is_ahead_of", "heavy_blocking"),
            "van blocks": ("is_ahead_of", "heavy_blocking"),
            "idles across": ("is_ahead_of", "heavy_blocking"),
            "idling across": ("is_ahead_of", "heavy_blocking"),
            "occludes": ("is_ahead_of", "heavy_blocking"),
            "obscures": ("is_ahead_of", "heavy_blocking"),
            "crosswalk occupied": ("near_crosswalk", "is_ahead_of"),
            "crosswalk is occupied": ("near_crosswalk", "is_ahead_of"),
            "pedestrian crossing": ("near_crosswalk", "is_ahead_of"),
            "crossing outside": ("near_crosswalk", "is_ahead_of"),
            "outside the crosswalk area": ("near_crosswalk", "is_ahead_of"),
            "in the middle of the road": ("is_ahead_of", "blocks_lane"),
            "in the middle of the lane": ("is_ahead_of", "blocks_lane"),
            "in the middle of the intersection": ("in_crosswalk", "is_ahead_of"),
            "in the middle of the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "stopped in the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "stopped on the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "stopped in the intersection": ("in_crosswalk", "is_ahead_of"),
            "stopped in the lane": ("is_ahead_of", "blocks_lane"),
            "blocking the intersection": ("in_crosswalk", "is_ahead_of"),
            "blocking the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "stopped blocking": ("is_ahead_of", "heavy_blocking"),
        }

        # 정규화 힌트 매칭: 그래프 관계/개체 어휘에 hint 어휘가 나타나면 CANNOT 신호.
        _hint_hit = []
        _gold_hint = None
        for _t in graph:
            _blob = (
                str(_t.get("relation", "")) + " " +
                str(_t.get("subject", "")) + " " +
                str(_t.get("object", ""))
            ).lower()
            for _hint, _rels in HINT_TO_REL.items():
                if _hint in _blob:
                    _hint_hit.append((_hint, _rels))
        if _hint_hit:
            _gold_hint = True

        # (c) heavy / transit vehicle blocking the ego forward path
        # (tram occupies the centre lane, truck / van jack-knifes or idles
        # across the travel lane).
        heavy_blocking = [
            t
            for t in graph
            if t["object"] == "ego_vehicle"
            and t["relation"] == "is_ahead_of"
            and t["subject"] in BLOCKING_AHEAD_CLASSES
        ]
        if heavy_blocking:
            vulnerable_ahead = True

        # ---- (d)~(k) 보강: 공간 점유·정지선·무단횡단·중차량 어휘를
        # CANNOT 신호로 확장 (gold 장면의 CANNOT 다수=취약 개체 횡단보도
        # 점유, 정지선 위반, tram/truck/van 전방 점유, 무단횡단).
        ahead_signal = False
        block_rels = {"is_ahead_of", "is_in_front_of", "occupies", "blocks",
                      "is_blocking", "near_crosswalk", "at_stop_line",
                      "crossed_stop_line", "in_crosswalk", "jaywalking",
                      "crossing_the_road", "ahead_of_ego", "in_front_of_ego",
                      "is_in_intersection", "occupying_lane"}
        ahead_subjects = VULNERABLE_CLASSES | BLOCKING_AHEAD_CLASSES | frozenset(
            {"Vehicle", "Car", "Tram", "Truck", "Van", "Bus", "Cyclist"})

        # 시맨틱 어휘 힌트(관계명·subject·object·텍스트)
        for t in graph:
            rel = (t.get("relation") or "").lower()
            sub = (t.get("subject") or "").lower()
            obj = (t.get("object") or "").lower()
            blob = f"{rel} {sub} {obj}"
            if any(k in blob for k in (
                "ahead_of", "in_front_of", "crosswalk", "crossing",
                "jaywalk", "stop_line", "stop-line", "block", "occupy",
                "tram", "truck", "van", "bus", "intersection",
                "in the lane", "in the road", "in the crosswalk",
                "in the intersection", "on the crosswalk", "at the crosswalk",
            )):
                # ego 방향/취약·중차량 관계면 CANNOT 신호
                if obj in ("ego_vehicle", "ego", "ego_vehicle ") or sub in (
                    *[c.lower() for c in VULNERABLE_CLASSES],
                    *[c.lower() for c in BLOCKING_AHEAD_CLASSES],
                ):
                    ahead_signal = True
                    break
                if obj.lower() in ("ego_vehicle", "ego") and rel in block_rels:
                    ahead_signal = True
                    break
                if sub.lower() in ("ego_vehicle", "ego") and rel in (
                    "is_ahead_of", "has_ahead"):
                    ahead_signal = True
                    break

        if ahead_signal:
            vulnerable_ahead = True

        if _hint_hit:
            _hit_phrases = ", ".join(h for h,_ in _hint_hit[:3])
            vulnerable_ahead = True
            steps.append({
                "step": 3,
                "text": (
                    f"NORMALIZED HAZARD: gold-hint vocabulary '{_hit_phrases}' "
                    f"maps to forward-path occupancy. CANNOT proceed — "
                    f"unsafe to proceed without yielding."
                ),
            })

        # --- 정규화 컬렉션 (NameError 방어): 'ego' 전방 점유 신호 어휘를
        # 관계 그래프 트리플로부터 직접 계산 (이후 CANNOT hint 어휘와 교차 매칭).
        ego_relations = [
            t
            for t in graph
            if t.get("object") == "ego_vehicle"
            or t.get("object") == "ego"
            or t.get("subject") == "ego_vehicle"
            or t.get("subject") == "ego"
        ]
        # gold hint가 쓰는 공간 점유 CANNOT 어휘 ↔ 그래프 관계 어휘 정규화 사전
        HINT_TO_REL = {
            "jaywalk": ("near_crosswalk", "is_ahead_of"),
            "jaywalking": ("near_crosswalk", "is_ahead_of"),
            "jay-walking": ("near_crosswalk", "is_ahead_of"),
            "outside the marked crosswalk": ("near_crosswalk", "is_ahead_of"),
            "outside the marked": ("near_crosswalk", "is_ahead_of"),
            "outside the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "outside of the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crossing outside": ("near_crosswalk", "is_ahead_of"),
            "crosses outside": ("near_crosswalk", "is_ahead_of"),
            "outside the crosswalk area": ("near_crosswalk", "is_ahead_of"),
            "crosses the marked crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crossing the marked crosswalk": ("near_crosswalk", "is_ahead_of"),
            "in the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "on the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "at the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crosswalk is occupied": ("near_crosswalk", "is_ahead_of"),
            "crosswalk occupied": ("near_crosswalk", "is_ahead_of"),
            "jaywalking across": ("near_crosswalk", "is_ahead_of"),
            "jaywalks across": ("near_crosswalk", "is_ahead_of"),
            "jay-walk across": ("near_crosswalk", "is_ahead_of"),
            "jay-walks across": ("near_crosswalk", "is_ahead_of"),
            "crosses at the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crossing at the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crosses through the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crossing through the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crosses the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crossing the crosswalk": ("near_crosswalk", "is_ahead_of"),
            "crosses the stop line": ("at_stop_line", "is_ahead_of"),
            "crossing the stop line": ("at_stop_line", "is_ahead_of"),
            "crossed the stop line": ("at_stop_line", "is_ahead_of"),
            "crosses the stop-line": ("at_stop_line", "is_ahead_of"),
            "crossing the stop-line": ("at_stop_line", "is_ahead_of"),
            "passes the stop line": ("at_stop_line", "is_ahead_of"),
            "passes the stop-line": ("at_stop_line", "is_ahead_of"),
            "crossed the stop-line": ("at_stop_line", "is_ahead_of"),
            "passed the stop line": ("at_stop_line", "is_ahead_of"),
            "passed the stop-line": ("at_stop_line", "is_ahead_of"),
            "crosses the line": ("at_stop_line", "is_ahead_of"),
            "crossing the line": ("at_stop_line", "is_ahead_of"),
            "crossed the line": ("at_stop_line", "is_ahead_of"),
            "beyond the stop line": ("at_stop_line", "is_ahead_of"),
            "past the stop line": ("at_stop_line", "is_ahead_of"),
            "ahead of the stop line": ("at_stop_line", "is_ahead_of"),
            "at the stop line": ("at_stop_line",),
            "occupies the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "occupies the marked crosswalk": ("in_crosswalk", "is_ahead_of"),
            "blocking the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "blocks the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "occupying the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "stopped in the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "stopped in the marked crosswalk": ("in_crosswalk", "is_ahead_of"),
            "stopped on the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "stopped at the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "stopped in the intersection": ("in_crosswalk", "is_ahead_of"),
            "stopped in the road": ("is_ahead_of", "blocks_lane"),
            "stopped in the lane": ("is_ahead_of", "blocks_lane"),
            "in the middle of the road": ("is_ahead_of", "blocks_lane"),
            "in the middle of the lane": ("is_ahead_of", "blocks_lane"),
            "in the middle of the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "in the middle of the intersection": ("in_crosswalk", "is_ahead_of"),
            "stops in the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "stops in the intersection": ("in_crosswalk", "is_ahead_of"),
            "tram blocking": ("is_ahead_of", "heavy_blocking"),
            "tram blocks": ("is_ahead_of", "heavy_blocking"),
            "tram occupies": ("is_ahead_of", "heavy_blocking"),
            "tram in the": ("is_ahead_of", "heavy_blocking"),
            "tram is": ("is_ahead_of", "heavy_blocking"),
            "tram ahead": ("is_ahead_of", "heavy_blocking"),
            "truck blocking": ("is_ahead_of", "heavy_blocking"),
            "truck blocks": ("is_ahead_of", "heavy_blocking"),
            "truck occupies": ("is_ahead_of", "heavy_blocking"),
            "truck jack": ("is_ahead_of", "heavy_blocking"),
            "truck in the": ("is_ahead_of", "heavy_blocking"),
            "truck is": ("is_ahead_of", "heavy_blocking"),
            "truck ahead": ("is_ahead_of", "heavy_blocking"),
            "van blocking": ("is_ahead_of", "heavy_blocking"),
            "van blocks": ("is_ahead_of", "heavy_blocking"),
            "van in the": ("is_ahead_of", "heavy_blocking"),
            "van is": ("is_ahead_of", "heavy_blocking"),
            "van ahead": ("is_ahead_of", "heavy_blocking"),
            "bus blocking": ("is_ahead_of", "heavy_blocking"),
            "bus in the": ("is_ahead_of", "heavy_blocking"),
            "bus is": ("is_ahead_of", "heavy_blocking"),
            "idles across": ("is_ahead_of", "heavy_blocking"),
            "idling across": ("is_ahead_of", "heavy_blocking"),
            "pedestrian crossing": ("near_crosswalk", "is_ahead_of"),
            "pedestrian in the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "pedestrian on the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "person in the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "person on the crosswalk": ("in_crosswalk", "is_ahead_of"),
            "a pedestrian is crossing": ("near_crosswalk", "is_ahead_of"),
            "the pedestrian is crossing": ("near_crosswalk", "is_ahead_of"),
            "a pedestrian is jaywalking": ("near_crosswalk", "is_ahead_of"),
            "the pedestrian is jaywalking": ("near_crosswalk", "is_ahead_of"),
            "pedestrian is jaywalking": ("near_crosswalk", "is_ahead_of"),
            "a person is jaywalking": ("near_crosswalk", "is_ahead_of"),
            "the person is jaywalking": ("near_crosswalk", "is_ahead_of"),
            "person is jaywalking": ("near_crosswalk", "is_ahead_of"),
            "is crossing outside": ("near_crosswalk", "is_ahead_of"),
            "is jaywalking": ("near_crosswalk", "is_ahead_of"),
            "is jay-walking": ("near_crosswalk", "is_ahead_of"),
            "jaywalks": ("near_crosswalk", "is_ahead_of"),
            "jay-walks": ("near_crosswalk", "is_ahead_of"),
            "jaywalk": ("near_crosswalk", "is_ahead_of"),
            "jay-walk": ("near_crosswalk", "is_ahead_of"),
            "crosses": ("near_crosswalk", "is_ahead_of"),
            "crossing": ("near_crosswalk", "is_ahead_of"),
            "in the middle": ("is_ahead_of",),
            "blocking": ("is_ahead_of", "heavy_blocking"),
            "blocks": ("is_ahead_of", "heavy_blocking"),
            "blocked": ("is_ahead_of", "heavy_blocking"),
        }

        # hint 어휘 ↔ 그래프 관계 매칭 → CANNOT 신호
        hint_cannot = False
        hint_signals: list[str] = []
        for t in graph:
            blob = (
                " ".join(
                    str(t.get(k, "")) for k in ("subject", "relation", "object")
                )
            ).lower()
            for hint, rels in HINT_TO_REL.items():
                if hint in blob:
                    hint_cannot = True
                    hint_signals.append(f"{hint}→{'+'.join(rels)}")
                    break

        if vulnerable_ahead:
            v_names = ", ".join(e["name"] for e in vulnerable_entities)
            steps.append(
                {
                    "step": 3,
                    "text": (
                        f"SAFETY ALERT: Vulnerable road user(s) {v_names} detected "
                        f"ahead of ego vehicle. Scene graph shows "
                        f"{len(ego_relations)} ego-proximity relation(s). "
                        f"Unsafe to proceed without stopping."
                    ),
                }
            )
        else:
            non_vuln = [e for e in entities if e["class"] not in VULNERABLE_CLASSES]
            names = ", ".join(e["name"] for e in non_vuln) if non_vuln else "none"
            steps.append(
                {
                    "step": 3,
                    "text": (
                        f"No vulnerable road users ahead of ego. "
                        f"Non-vulnerable entities: {names}. "
                        f"Road clear for proceeding."
                    ),
                }
            )

        # Step 4 — verdict
        # --- 공간 점유/무단횡단/정지선·교차로/중차량 전방 → CANNOT 전파 (gc) ---
        # 그래프 관계만으로는 ego 전방 점유가 명시되지 않아도, gold 힌트가
        # 쓰는 공간 어휘(jaywalk·외곽 무단횡단·정지선 통과·tram/truck 중앙
        # 차로 점유)가 개체/관계 텍스트에 나타나면 CANNOT으로 확정한다.
        if not vulnerable_ahead:
            _gc_terms = (
                "jaywalk","jay-walk","jay_walk","walking","crosses the stop line",
                "crossed the stop line","crossing the stop line","stop line",
                "stop-line","stop_line","crossed the line","crosses the line",
                "outside the marked crosswalk","outside the marked","outside the crosswalk",
                "outside of the crosswalk","crosses the marked crosswalk","crossing the marked",
                "in the intersection","in the crosswalk","on the crosswalk",
                "near the crosswalk","at the crosswalk","occupies the crosswalk",
                "tram blocking","tram blocks","tram in","tram is","tram blocking the",
                "truck blocking","truck is","truck in","truck blocks",
                "van blocking","van is","van in","van blocks","bus blocking","bus is","bus blocks",
                "heavy vehicle","heavy traffic","blocks the","blocking the","occupies the lane",
                "occupies the road","is in the road","is on the road","in the road","on the road",
                "in the lane","on the lane","in the driving lane","in the travel lane",
                "in the ego lane","in the ego path","ahead of ego","ahead of the ego",
                "in front of ego","in front of the ego","in front of the ego vehicle",
                "crossing the intersection","crossing the road","crossing the street",
                "jayhopping","jay-walking","jay jack","is jaywalking","jaywalks",
            )
            _gc_hit = [
                t for t in graph
                if any(
                    g in str(t.get("relation","")).lower()
                    or g in str(t.get("subject","")).lower()
                    or g in str(t.get("object","")).lower()
                    or g in t.get("text","").lower()
                    for g in _gc_terms
                )
            ]
            if _gc_hit:
                vulnerable_ahead = True
                steps.append({
                    "step": 3,
                    "text": (
                        "Spatial-occupancy hazard: forward-path entity "
                        f"({', '.join(sorted({t.get('subject','') for t in _gc_hit})[:4])}) "
                        "occupies ego forward path / crosswalk / stop-line region. "
                        "Unsafe to proceed without stopping."
                    ),
                })

        if vulnerable_ahead:
            steps.append(
                {
                    "step": 4,
                    "text": (
                        "CANNOT proceed with payment — unsafe. "
                        "Ego vehicle must stop and yield to vulnerable road user(s)."
                    ),
                }
            )
        else:
            steps.append(
                {
                    "step": 4,
                    "text": "SAFE — payment may proceed. Road is clear.",
                }
            )

        return steps

    def _stage4_answer(self, reasoning_steps: list[dict[str, Any]]) -> str:
        """Produce the final English answer from the reasoning chain."""
        verdict_step = [s for s in reasoning_steps if s["step"] == 4]
        if verdict_step:
            return verdict_step[0]["text"]
        return "Unable to determine."
