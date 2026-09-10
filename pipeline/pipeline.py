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
    """Stub for the T7 Qwen2.5-VL adapter.

    Raises FileNotFoundError if ``outputs/lora_adapter`` does not exist,
    clearly instructing the operator to finish task 7 first.
    """

    def __init__(self, adapter_path: str = "outputs/lora_adapter") -> None:
        self._adapter_path = Path(adapter_path)

    def generate(self, scene: dict[str, Any], question: str) -> dict[str, Any]:
        if not self._adapter_path.exists():
            raise FileNotFoundError(
                f"Qwen2.5-VL adapter not found at {self._adapter_path}.\n"
                "Run task 7 (T7 Qwen2.5-VL-3B LoRA fine-tuning) first, then retry.\n"
                "Meanwhile, use MockBackend for offline testing."
            )
        raise NotImplementedError("QwenVLBackend.generate() not yet implemented — T7 pending.")


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

        return {
            "scene_id": scene_id,
            "question": question,
            "stage1_grounding": s1,
            "stage2_scene_graph": s2,
            "stage3_reasoning": s3,
            "stage4_answer": s4,
        }

    # --- Stage implementations -------------------------------------------

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

        # Step 3 — safety analysis
        vulnerable_ahead = False
        vulnerable_entities = [
            e for e in entities if e["class"] in VULNERABLE_CLASSES
        ]
        ego_relations = [
            t
            for t in graph
            if t["object"] == "ego_vehicle"
            and t["subject"] in {e["name"] for e in vulnerable_entities}
        ]

        if ego_relations:
            vulnerable_ahead = True

        # Also check if a vulnerable entity is "near" or "is_near" ego
        near_ego_vulnerable = [
            t
            for t in graph
            if t["relation"] in ("near_ego_vehicle", "is_near")
            and t["subject"] in {e["name"] for e in vulnerable_entities}
        ]
        if near_ego_vulnerable:
            vulnerable_ahead = True

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
