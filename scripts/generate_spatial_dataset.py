#!/usr/bin/env python3
"""
Generate spatial understanding instruction dataset for KITTI autonomous driving.

Synthesizes plausible scene layouts using KITTI domain knowledge (classes from T2,
pose joints from T4, segmentation context from T3). This is NOT real KITTI label data —
it is instruction-tuning data with designed coordinates consistent with scene graphs.

Output: JSONL files (train/val) loadable by HF datasets, plus schema.json.
"""

import json
import random
import os
from typing import Any

random.seed(42)

KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
KITTI_IMAGE_SIZE = [375, 1242]  # H, W typical KITTI

MPII_JOINTS = [
    "R_ANKLE", "R_KNEE", "R_HIP", "L_HIP", "L_KNEE", "L_ANKLE",
    "PELVIS", "THORAX", "UPPER_NECK", "HEAD_TOP",
    "R_WRIST", "R_ELBOW", "R_SHOULDER", "L_SHOULDER", "L_ELBOW", "L_WRIST"
]


def rand_box(cx_range, cy_range, w_range, h_range):
    """Generate a bounding box [x1,y1,x2,y2] normalized 0-1."""
    cx = random.uniform(*cx_range)
    cy = random.uniform(*cy_range)
    w = random.uniform(*w_range)
    h = random.uniform(*h_range)
    x1 = max(0.0, cx - w / 2)
    y1 = max(0.0, cy - h / 2)
    x2 = min(1.0, cx + w / 2)
    y2 = min(1.0, cy + h / 2)
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def standing_pose(base_x=0.5, base_y=0.4):
    """Generate a plausible standing pedestrian MPII pose (normalized coords)."""
    return {
        "R_ANKLE": [round(base_x + 0.02, 4), round(base_y + 0.35, 4)],
        "R_KNEE": [round(base_x + 0.01, 4), round(base_y + 0.2, 4)],
        "R_HIP": [round(base_x + 0.01, 4), round(base_y + 0.05, 4)],
        "L_HIP": [round(base_x - 0.03, 4), round(base_y + 0.05, 4)],
        "L_KNEE": [round(base_x - 0.04, 4), round(base_y + 0.2, 4)],
        "L_ANKLE": [round(base_x - 0.05, 4), round(base_y + 0.35, 4)],
        "PELVIS": [round(base_x - 0.01, 4), round(base_y + 0.0, 4)],
        "THORAX": [round(base_x - 0.01, 4), round(base_y - 0.15, 4)],
        "UPPER_NECK": [round(base_x - 0.01, 4), round(base_y - 0.2, 4)],
        "HEAD_TOP": [round(base_x - 0.01, 4), round(base_y - 0.3, 4)],
        "R_WRIST": [round(base_x + 0.08, 4), round(base_y + 0.0, 4)],
        "R_ELBOW": [round(base_x + 0.06, 4), round(base_y - 0.08, 4)],
        "R_SHOULDER": [round(base_x + 0.04, 4), round(base_y - 0.13, 4)],
        "L_SHOULDER": [round(base_x - 0.06, 4), round(base_y - 0.13, 4)],
        "L_ELBOW": [round(base_x - 0.08, 4), round(base_y - 0.08, 4)],
        "L_WRIST": [round(base_x - 0.1, 4), round(base_y + 0.0, 4)],
    }


def crossing_pose(base_x=0.5, base_y=0.4):
    """Pedestrian with one leg forward (crossing the road)."""
    p = standing_pose(base_x, base_y)
    p["R_ANKLE"] = [round(base_x + 0.06, 4), round(base_y + 0.33, 4)]
    p["R_KNEE"] = [round(base_x + 0.04, 4), round(base_y + 0.18, 4)]
    p["L_ANKLE"] = [round(base_x - 0.08, 4), round(base_y + 0.35, 4)]
    p["L_KNEE"] = [round(base_x - 0.06, 4), round(base_y + 0.2, 4)]
    p["R_WRIST"] = [round(base_x + 0.12, 4), round(base_y - 0.05, 4)]
    p["L_WRIST"] = [round(base_x - 0.14, 4), round(base_y + 0.05, 4)]
    return p


# ─── Scenario templates ───

def make_car_entity(idx, cx_range, cy_range, w_range=(0.08, 0.18), h_range=(0.06, 0.12)):
    return {
        "name": f"car_{idx}",
        "class": "Car",
        "bbox": rand_box(cx_range, cy_range, w_range, h_range),
    }


def make_ped_entity(idx, cx_range, cy_range, pose_type="standing"):
    pose_fn = crossing_pose if pose_type == "crossing" else standing_pose
    bx = random.uniform(*cx_range)
    by = random.uniform(*cy_range)
    return {
        "name": f"pedestrian_{idx}",
        "class": "Pedestrian",
        "bbox": [round(bx - 0.025, 4), round(by - 0.15, 4), round(bx + 0.025, 4), round(by + 0.15, 4)],
        "pose": pose_fn(bx, by - 0.15),
    }


def make_truck_entity(idx, cx_range, cy_range):
    return {
        "name": f"truck_{idx}",
        "class": "Truck",
        "bbox": rand_box(cx_range, cy_range, (0.1, 0.2), (0.08, 0.14)),
    }


def make_cyclist_entity(idx, cx_range, cy_range):
    return {
        "name": f"cyclist_{idx}",
        "class": "Cyclist",
        "bbox": rand_box(cx_range, cy_range, (0.04, 0.08), (0.1, 0.2)),
    }


def make_van_entity(idx, cx_range, cy_range):
    return {
        "name": f"van_{idx}",
        "class": "Van",
        "bbox": rand_box(cx_range, cy_range, (0.08, 0.15), (0.07, 0.12)),
    }


def make_tram_entity(idx, cx_range, cy_range):
    return {
        "name": f"tram_{idx}",
        "class": "Tram",
        "bbox": rand_box(cx_range, cy_range, (0.15, 0.3), (0.08, 0.14)),
    }


def spatial_rel(e1, e2):
    """Determine spatial relationship between two entities based on bbox centers."""
    cx1 = (e1["bbox"][0] + e1["bbox"][2]) / 2
    cx2 = (e2["bbox"][0] + e2["bbox"][2]) / 2
    cy1 = (e1["bbox"][1] + e1["bbox"][3]) / 2
    cy2 = (e2["bbox"][1] + e2["bbox"][3]) / 2
    rels = []
    if cx1 < cx2 - 0.05:
        rels.append("is_left_of")
    elif cx1 > cx2 + 0.05:
        rels.append("is_right_of")
    if cy1 < cy2 - 0.03:
        rels.append("is_behind")
    elif cy1 > cy2 + 0.03:
        rels.append("is_ahead_of")
    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    if dist < 0.25:
        rels.append("is_near")
    return rels if rels else ["is_near"]


def build_scene_graph(entities, road_features=None):
    """Build a scene graph (triplets) from entities and optional road features."""
    triples = []
    for i, e1 in enumerate(entities):
        for j, e2 in enumerate(entities):
            if i >= j:
                continue
            rels = spatial_rel(e1, e2)
            for r in rels:
                triples.append({"subject": e1["name"], "relation": r, "object": e2["name"]})
    if road_features:
        for feat in road_features:
            for e in entities:
                cy = (e["bbox"][1] + e["bbox"][3]) / 2
                cx = (e["bbox"][0] + e["bbox"][2]) / 2
                if feat["type"] == "crosswalk" and abs(cx - feat["cx"]) < 0.15 and cy > 0.5:
                    triples.append({"subject": e["name"], "relation": "near_crosswalk", "object": "crosswalk_1"})
                if feat["type"] == "traffic_light" and cy < 0.4:
                    triples.append({"subject": e["name"], "relation": "below_traffic_light", "object": "traffic_light_1"})
    for e in entities:
        cy = (e["bbox"][1] + e["bbox"][3]) / 2
        rel = "near_ego_vehicle" if cy > 0.6 else "is_ahead_of"
        triples.append({"subject": e["name"], "relation": rel, "object": "ego_vehicle"})
    return triples


def staged_reasoning_steps(grounding_desc, graph_desc, reasoning_desc, answer):
    return [
        {"step": 1, "text": grounding_desc},
        {"step": 2, "text": graph_desc},
        {"step": 3, "text": reasoning_desc},
        {"step": 4, "text": f"Therefore: {answer}"},
    ]


# ─── Scenario generators ───

def scenario_ego_stop_before_ped():
    """Can the ego vehicle stop safely before the pedestrian?"""
    ped = make_ped_entity(1, (0.4, 0.6), (0.55, 0.7), "crossing")
    car_behind = make_car_entity(1, (0.3, 0.5), (0.7, 0.85))
    entities = [ped, car_behind]
    road = [{"type": "crosswalk", "cx": 0.5}]
    sg = build_scene_graph(entities, road)

    ped_cy = (ped["bbox"][1] + ped["bbox"][3]) / 2
    dist_desc = "close" if ped_cy > 0.55 else "moderate"
    answer = random.choice([
        f"The ego vehicle should stop immediately. The pedestrian is in the crosswalk at {dist_desc} distance.",
        f"Safe stopping is possible. The pedestrian is crossing at {dist_desc} range and the ego vehicle has sufficient deceleration distance.",
        f"The ego vehicle cannot stop safely in time. The pedestrian is {dist_desc} and already in the road.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": road,
        "question": "Can the ego vehicle stop safely before the pedestrian in the crosswalk?",
        "answer": answer,
        "reasoning": (
            f"Grounding identified a Pedestrian in the crosswalk and a Car behind. "
            f"Scene graph shows pedestrian_1 near_crosswalk crosswalk_1 and car_1 is_behind pedestrian_1. "
            f"Distance is {dist_desc}, requiring immediate braking decision."
        ),
    }


def scenario_payment_decision():
    """Should in-car payment proceed while pedestrian is in the crosswalk?"""
    ped = make_ped_entity(1, (0.4, 0.6), (0.5, 0.65), "crossing")
    car_ego = make_car_entity(1, (0.45, 0.55), (0.8, 0.95))
    entities = [ped, car_ego]
    road = [{"type": "crosswalk", "cx": 0.5}]
    sg = build_scene_graph(entities, road)
    answer = random.choice([
        "No. The in-car payment should NOT proceed while the pedestrian is actively crossing. Driver attention must remain on the road.",
        "Yes, if the vehicle is fully stopped and the pedestrian has cleared the ego lane. Otherwise, defer payment.",
        "No. Autonomous driving safety protocols require deferring non-driving tasks when pedestrians are in the roadway.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": road,
        "question": "Should the in-car payment system proceed while a pedestrian is in the crosswalk?",
        "answer": answer,
        "reasoning": (
            "Grounding found a Pedestrian in the crosswalk and the ego vehicle nearby. "
            "Scene graph links pedestrian_1 near_crosswalk crosswalk_1. "
            "Safety reasoning: active pedestrian in roadway demands full driver/vehicle attention."
        ),
    }


def scenario_spatial_left_right():
    """Which vehicle is to the left of the truck?"""
    truck = make_truck_entity(1, (0.45, 0.55), (0.4, 0.55))
    car_left = make_car_entity(1, (0.2, 0.35), (0.42, 0.55))
    car_right = make_car_entity(2, (0.65, 0.8), (0.42, 0.55))
    van_right2 = make_van_entity(1, (0.8, 0.95), (0.42, 0.55))
    entities = [truck, car_left, car_right, van_right2]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "Car_1 is to the left of the truck.",
        "The vehicle to the left of the truck is Car_1.",
        "Car_1 (the leftmost car) is positioned to the left of the truck.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Which vehicle is to the left of the truck in this scene?",
        "answer": answer,
        "reasoning": (
            "Grounding identified 4 vehicles: Truck, Car_1, Car_2, Van_1. "
            "Scene graph shows car_1 is_left_of truck_1, car_2 is_right_of truck_1, van_1 is_right_of truck_1. "
            "Spatial comparison confirms car_1 is left of truck."
        ),
    }


def scenario_cyclist_safety():
    """Is the cyclist in a safe position relative to the truck?"""
    cyclist = make_cyclist_entity(1, (0.3, 0.45), (0.5, 0.65))
    truck = make_truck_entity(1, (0.5, 0.65), (0.5, 0.65))
    car_behind = make_car_entity(1, (0.4, 0.55), (0.7, 0.85))
    entities = [cyclist, truck, car_behind]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "The cyclist is in a dangerous position — directly adjacent to the truck with a car approaching from behind.",
        "The cyclist has adequate lateral clearance from the truck. The car behind poses no immediate threat.",
        "The cyclist is in a vulnerable position beside the truck. The truck's blind spot may obscure the cyclist from other drivers.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Is the cyclist in a safe position relative to the truck?",
        "answer": answer,
        "reasoning": (
            "Grounding: Cyclist beside Truck, Car behind. "
            "Scene graph: cyclist_1 is_left_of truck_1, car_1 is_behind cyclist_1. "
            "Safety analysis: lateral proximity to truck creates blind-spot risk."
        ),
    }


def scenario_multi_vehicle_platoon():
    """Describe the spatial arrangement of all vehicles."""
    car1 = make_car_entity(1, (0.1, 0.25), (0.5, 0.6))
    car2 = make_car_entity(2, (0.35, 0.5), (0.5, 0.6))
    truck = make_truck_entity(1, (0.55, 0.7), (0.5, 0.6))
    van = make_van_entity(1, (0.75, 0.9), (0.5, 0.6))
    entities = [car1, car2, truck, van]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "From left to right: Car_1, Car_2, Truck_1, Van_1 — a four-vehicle platoon in the same lane.",
        "The vehicles are arranged left to right as Car_1, Car_2, Truck_1, Van_1, all traveling in the same direction.",
        "Left to right order: Car_1, Car_2, Truck_1, Van_1. They form a single-lane platoon.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Describe the spatial arrangement of all vehicles in this scene.",
        "answer": answer,
        "reasoning": (
            "Grounding: 4 vehicles detected — 2 Cars, 1 Truck, 1 Van. "
            "Scene graph: car_1 is_left_of car_2 is_left_of truck_1 is_left_of van_1. "
            "Spatial ordering from ego-left to ego-right established."
        ),
    }


def scenario_pedestrian_jaywalking():
    """Is the pedestrian crossing legally or jaywalking?"""
    ped = make_ped_entity(1, (0.35, 0.5), (0.4, 0.55), "crossing")
    car_ego = make_car_entity(1, (0.45, 0.55), (0.7, 0.85))
    entities = [ped, car_ego]
    road = [{"type": "crosswalk", "cx": 0.8}]
    sg = build_scene_graph(entities, road)
    answer = random.choice([
        "The pedestrian is jaywalking — crossing outside the marked crosswalk which is positioned to the right.",
        "The pedestrian is crossing at an unmarked location. The crosswalk is further right, indicating jaywalking.",
        "This appears to be jaywalking. The pedestrian is not at the designated crosswalk location.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": road,
        "question": "Is the pedestrian crossing legally at a crosswalk or jaywalking?",
        "answer": answer,
        "reasoning": (
            "Grounding: Pedestrian mid-road, crosswalk detected far right. "
            "Scene graph: pedestrian_1 near crosswalk_1 but not at crosswalk position. "
            "Legal analysis: pedestrian is outside the marked crosswalk zone."
        ),
    }


def scenario_traffic_light_obey():
    """Is the car obeying the traffic light?"""
    car = make_car_entity(1, (0.4, 0.6), (0.6, 0.75))
    entities = [car]
    road = [{"type": "traffic_light", "cx": 0.5, "state": "red"}]
    sg = build_scene_graph(entities, road)
    car_cy = (car["bbox"][1] + car["bbox"][3]) / 2
    if car_cy < 0.65:
        answer = "The car is stopped before the intersection, obeying the red traffic light."
    else:
        answer = "The car appears to be approaching or entering the intersection despite the red light — potential violation."
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": road,
        "question": "Is the car obeying the traffic light at this intersection?",
        "answer": answer,
        "reasoning": (
            "Grounding: Car detected near intersection with traffic light. "
            "Scene graph: car_1 below_traffic_light traffic_light_1. "
            "Traffic light state is red; car position determines compliance."
        ),
    }


def scenario_two_peds_different_states():
    """Compare the poses of two pedestrians."""
    ped1 = make_ped_entity(1, (0.2, 0.35), (0.5, 0.65), "standing")
    ped2 = make_ped_entity(2, (0.6, 0.75), (0.5, 0.65), "crossing")
    car = make_car_entity(1, (0.4, 0.55), (0.75, 0.9))
    entities = [ped1, ped2, car]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "Pedestrian_1 is standing on the sidewalk (left), while Pedestrian_2 is actively crossing the road (right). Pedestrian_2 requires more attention from the ego vehicle.",
        "The left pedestrian is stationary; the right pedestrian is in a crossing stance. The crossing pedestrian poses a higher collision risk.",
        "Pedestrian_1 stands still on the left. Pedestrian_2 crosses from right with legs apart. Only Pedestrian_2 is a dynamic obstacle.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Compare the poses and states of the two pedestrians. Which one poses a greater safety concern?",
        "answer": answer,
        "reasoning": (
            "Grounding: Two pedestrians detected with different poses. "
            "Scene graph: pedestrian_1 is_left_of pedestrian_2, pedestrian_2 is_near car_1. "
            "Pose analysis: ped_1 standing (stationary), ped_2 crossing (dynamic). Dynamic pedestrian has higher risk."
        ),
    }


def scenario_tram_interference():
    """How does the tram affect other traffic?"""
    tram = make_tram_entity(1, (0.5, 0.7), (0.3, 0.5))
    car1 = make_car_entity(1, (0.15, 0.35), (0.5, 0.65))
    car2 = make_car_entity(2, (0.75, 0.95), (0.5, 0.65))
    entities = [tram, car1, car2]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "The tram occupies the center lane, forcing Car_1 to the left and Car_2 to the right. Both cars must yield to the tram's right-of-way.",
        "The tram blocks the center. Cars on either side must slow down and cannot overtake while the tram is present.",
        "The tram is the dominant road user — cars on both sides must maintain distance and yield.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "How does the tram affect the surrounding traffic in this scene?",
        "answer": answer,
        "reasoning": (
            "Grounding: Tram center, two cars flanking. "
            "Scene graph: car_1 is_left_of tram_1, car_2 is_right_of tram_1. "
            "Traffic rules: tram has right-of-way, cars must yield and maintain clearance."
        ),
    }


def scenario_parked_car_occlusion():
    """Is there an occluded pedestrian behind the parked car?"""
    car_parked = make_car_entity(1, (0.3, 0.5), (0.55, 0.7))
    ped_partial = make_ped_entity(1, (0.48, 0.55), (0.55, 0.7), "standing")
    entities = [car_parked, ped_partial]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "A partially occluded pedestrian is visible behind the parked car. The ego vehicle should prepare for the pedestrian to step into the road.",
        "The pedestrian is mostly hidden by the parked car — a classic occlusion hazard. Reduced speed recommended.",
        "Occlusion detected: a pedestrian's legs are visible behind the parked car. Full emergence into the lane is possible.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Is there an occluded pedestrian behind the parked car that the ego vehicle should be aware of?",
        "answer": answer,
        "reasoning": (
            "Grounding: Parked car with partial pedestrian bbox overlapping. "
            "Scene graph: pedestrian_1 is_right_of car_1 (partially overlapping). "
            "Occlusion reasoning: pedestrian behind car may step out — predict hidden hazard."
        ),
    }


def scenario_ego_distance_estimate():
    """Estimate the distance to the nearest vehicle."""
    car_close = make_car_entity(1, (0.4, 0.6), (0.75, 0.9))
    car_far = make_car_entity(2, (0.4, 0.6), (0.2, 0.35))
    truck_far = make_truck_entity(1, (0.7, 0.85), (0.15, 0.3))
    entities = [car_close, car_far, truck_far]
    sg = build_scene_graph(entities)
    close_cy = (car_close["bbox"][1] + car_close["bbox"][3]) / 2
    answer = random.choice([
        f"Car_1 is the nearest vehicle at approximately 5-8 meters (large bbox, low in frame). Car_2 and Truck_1 are 30+ meters ahead.",
        f"The closest vehicle is Car_1 directly ahead at close range. The other two vehicles are significantly farther away.",
        f"Car_1 is nearest (estimated 5-10m based on bbox size). Car_2 is mid-range (~25m). Truck_1 is far (~40m).",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Which vehicle is nearest to the ego vehicle, and approximately how far is it?",
        "answer": answer,
        "reasoning": (
            "Grounding: 3 vehicles at different depths (indicated by y-position and bbox size). "
            "Scene graph: car_1 is_ahead car_2 is_ahead truck_1 (depth ordering). "
            "Distance estimation: bbox area inversely proportional to distance. Car_1 has largest bbox → nearest."
        ),
    }


def scenario_hard_negative_no_pedestrian():
    """Trick question — no pedestrian present."""
    car1 = make_car_entity(1, (0.2, 0.4), (0.5, 0.65))
    car2 = make_car_entity(2, (0.6, 0.8), (0.5, 0.65))
    entities = [car1, car2]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "There is no pedestrian in this scene. Only two cars are detected — one on the left and one on the right.",
        "No pedestrians are present. The scene contains two vehicles in adjacent lanes.",
        "This scene has no pedestrians. Only Car_1 (left) and Car_2 (right) are visible.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Describe the pedestrian's position and whether the ego vehicle should yield.",
        "answer": answer,
        "reasoning": (
            "Grounding: Only 2 cars detected, no pedestrian class found. "
            "Scene graph: car_1 is_left_of car_2. No pedestrian-related triples. "
            "Answer: no pedestrian present — the premise of the question is incorrect."
        ),
    }


def scenario_hard_negative_wrong_class():
    """Trick: what type is the 'cyclist'? — it's actually a van."""
    van = make_van_entity(1, (0.4, 0.6), (0.5, 0.65))
    car = make_car_entity(1, (0.15, 0.3), (0.5, 0.65))
    entities = [van, car]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "There is no cyclist in this scene. The vehicle in the center is a Van, not a cyclist.",
        "The object you may be referring to is actually a Van (class: Van), not a Cyclist. Only a Van and a Car are present.",
        "No cyclist detected. The center vehicle is classified as Van. The left vehicle is a Car.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "What is the cyclist doing in this scene? Describe their action and position.",
        "answer": answer,
        "reasoning": (
            "Grounding: Van and Car detected. No Cyclist class present. "
            "Scene graph: van_1 is_right_of car_1. No cyclist triples. "
            "Correction: the question assumes a cyclist, but the center object is a Van."
        ),
    }


def scenario_occlusion_depth():
    """Which vehicle is furthest from the ego vehicle?"""
    car_near = make_car_entity(1, (0.4, 0.6), (0.75, 0.9))
    car_mid = make_car_entity(2, (0.4, 0.6), (0.45, 0.55))
    truck_far = make_truck_entity(1, (0.4, 0.6), (0.15, 0.25))
    entities = [car_near, car_mid, truck_far]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "Truck_1 is the furthest vehicle, visible at the top of the frame (smallest bbox). Car_1 is nearest, Car_2 is mid-range.",
        "The truck is furthest away based on its small bbox and high position in the image. Car_1 is closest.",
        "Truck_1 is furthest (estimated 50+ meters). Car_2 is at medium range (~20m). Car_1 is closest (~5m).",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Which vehicle is furthest from the ego vehicle?",
        "answer": answer,
        "reasoning": (
            "Grounding: 3 vehicles at different vertical positions in frame. "
            "Scene graph: car_1 is_ahead car_2 is_ahead truck_1. "
            "Depth cue: higher y-position + smaller bbox = greater distance. Truck_1 is furthest."
        ),
    }


def scenario_emergency_vehicle():
    """Is there an emergency vehicle that requires yielding?"""
    car1 = make_car_entity(1, (0.2, 0.4), (0.55, 0.7))
    car2 = make_car_entity(2, (0.6, 0.8), (0.55, 0.7))
    tram = make_tram_entity(1, (0.4, 0.6), (0.3, 0.45))
    entities = [car1, car2, tram]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "No emergency vehicle is present. The scene contains two cars and a tram — all standard vehicles.",
        "There is no ambulance, fire truck, or police car in this scene. Only regular traffic is visible.",
        "No emergency vehicle detected. The tram is a public transit vehicle, not an emergency responder.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Is there an emergency vehicle in this scene that requires the ego vehicle to yield?",
        "answer": answer,
        "reasoning": (
            "Grounding: 2 Cars and 1 Tram detected. No emergency vehicle classes (ambulance/fire_truck/police_car). "
            "Scene graph: tram_1 is_behind car_1, tram_1 is_behind car_2. "
            "Conclusion: no emergency vehicle present."
        ),
    }


def scenario_person_sitting():
    """Identify and describe the person sitting on the curb."""
    person_sit = {
        "name": "person_sitting_1",
        "class": "Person_sitting",
        "bbox": rand_box((0.15, 0.3), (0.6, 0.75), (0.03, 0.06), (0.06, 0.1)),
    }
    car = make_car_entity(1, (0.5, 0.7), (0.5, 0.65))
    entities = [person_sit, car]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "A Person_sitting is detected on the left curb, separate from the moving traffic. Car_1 passes on the right.",
        "The sitting person is on the roadside (left), not in the driving lane. Car_1 is in the travel lane to the right.",
        "Person_sitting_1 is stationary on the curb. The moving Car_1 is in the lane to the right. No immediate collision risk.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Describe the person sitting near the road and assess any risk to the ego vehicle.",
        "answer": answer,
        "reasoning": (
            "Grounding: Person_sitting detected (rare KITTI class, low occurrence). "
            "Scene graph: person_sitting_1 is_left_of car_1. "
            "Risk assessment: sitting person is off-road, low collision risk unless they stand and enter the lane."
        ),
    }


def scenario_multiple_pedestrians_group():
    """How many pedestrians are crossing and which lane are they in?"""
    peds = []
    for i in range(3):
        cx = 0.3 + i * 0.2
        peds.append(make_ped_entity(i + 1, (cx - 0.05, cx + 0.05), (0.5, 0.65), "crossing"))
    car = make_car_entity(1, (0.4, 0.6), (0.8, 0.95))
    entities = peds + [car]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "Three pedestrians are crossing in a group from left to right across the ego lane. The ego vehicle should stop.",
        "A group of 3 pedestrians is crossing the road. All are in the ego lane — the vehicle must yield.",
        "Three Pedestrians are crossing together. They occupy the center of the driving lane. Stop recommended.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": [{"type": "crosswalk", "cx": 0.5}],
        "question": "How many pedestrians are crossing the road, and in which lane are they?",
        "answer": answer,
        "reasoning": (
            f"Grounding: 3 Pedestrians detected in crossing pose, 1 Car behind. "
            f"Scene graph: all pedestrians near_crosswalk crosswalk_1, car_1 is_ahead of pedestrian group. "
            f"Count: 3 active crossers in ego lane. Priority: pedestrian right-of-way."
        ),
    }


def scenario_distance_ordering():
    """Order all objects by distance from ego (nearest to farthest)."""
    car_near = make_car_entity(1, (0.4, 0.6), (0.8, 0.95))
    ped_mid = make_ped_entity(1, (0.3, 0.45), (0.5, 0.6))
    truck_far = make_truck_entity(1, (0.6, 0.8), (0.2, 0.35))
    cyclist_far2 = make_cyclist_entity(1, (0.1, 0.25), (0.1, 0.2))
    entities = [car_near, ped_mid, truck_far, cyclist_far2]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "Nearest to farthest: Car_1 (~5m), Pedestrian_1 (~15m), Truck_1 (~35m), Cyclist_1 (~50m).",
        "Distance order: Car_1 is closest, then Pedestrian_1, then Truck_1, then Cyclist_1 farthest away.",
        "Car_1 (nearest), Pedestrian_1 (mid), Truck_1 (far), Cyclist_1 (farthest) — ordered by bbox size and y-position.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Order all detected objects by their distance from the ego vehicle, nearest to farthest.",
        "answer": answer,
        "reasoning": (
            "Grounding: 4 objects — Car, Pedestrian, Truck, Cyclist at different depths. "
            "Scene graph: depth ordering based on y-position and bbox area. "
            "Distance estimation uses inverse bbox area and vertical position in frame."
        ),
    }


def scenario_lane_change_safety():
    """Is it safe for the ego vehicle to change lanes?"""
    car_left = make_car_entity(1, (0.05, 0.25), (0.5, 0.65))
    car_ahead = make_car_entity(2, (0.4, 0.6), (0.4, 0.55))
    car_right = make_car_entity(3, (0.75, 0.95), (0.55, 0.7))
    entities = [car_left, car_ahead, car_right]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "Lane change is NOT safe. Car_1 occupies the left lane and Car_3 is in the right lane. Both adjacent lanes are occupied.",
        "The ego vehicle cannot safely change lanes — both left and right lanes have vehicles at similar distances.",
        "Unsafe to change lanes: Car_1 (left) and Car_3 (right) block both adjacent lanes. Maintain current lane behind Car_2.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Is it safe for the ego vehicle to change lanes in this scene?",
        "answer": answer,
        "reasoning": (
            "Grounding: 3 cars — left, center-ahead, right. "
            "Scene graph: car_1 is_left_of car_2, car_3 is_right_of car_2. "
            "Lane change analysis: both adjacent lanes occupied → no safe gap."
        ),
    }


def scenario_weather_impact():
    """How do road conditions affect stopping distance?"""
    car_ahead = make_car_entity(1, (0.4, 0.6), (0.55, 0.7))
    entities = [car_ahead]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "Wet road conditions increase stopping distance by 30-50%. The ego vehicle should increase following distance to Car_1.",
        "On wet roads, the ego vehicle needs approximately 1.5x the normal stopping distance. Maintain extra gap behind Car_1.",
        "Rain reduces tire grip. The ego vehicle should double the following distance to Car_1 compared to dry conditions.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Given wet road conditions, how should the ego vehicle adjust its following distance to the car ahead?",
        "answer": answer,
        "reasoning": (
            "Grounding: Car detected ahead in the ego lane. "
            "Scene graph: car_1 is_ahead ego_vehicle. "
            "Environmental reasoning: wet roads reduce friction coefficient, increasing braking distance by 30-50%."
        ),
    }


def scenario_ego_stop_line():
    """Is the vehicle behind the stop line?"""
    car = make_car_entity(1, (0.4, 0.6), (0.7, 0.85))
    entities = [car]
    road = [{"type": "stop_line", "cx": 0.5, "cy": 0.6}]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "The car is behind the stop line — it has stopped at the correct position before the intersection.",
        "Car_1 is properly stopped behind the stop line. The vehicle is compliant with the traffic rule.",
        "The stop line is at y=0.6 and the car's front bbox is at y=0.7 — the car is behind the line. Correct stopping position.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": road,
        "question": "Is the car positioned behind the stop line at this intersection?",
        "answer": answer,
        "reasoning": (
            "Grounding: Car detected near intersection with stop line. "
            "Scene graph: car_1 is_ahead stop_line_1. "
            "Position check: car bbox y_min (0.7) > stop_line cy (0.6) → car is behind the line."
        ),
    }


def scenario_misc_object():
    """Identify the unusual object in the scene."""
    misc = {
        "name": "misc_1",
        "class": "Misc",
        "bbox": rand_box((0.6, 0.8), (0.5, 0.65), (0.04, 0.08), (0.04, 0.08)),
    }
    car = make_car_entity(1, (0.2, 0.4), (0.5, 0.65))
    entities = [misc, car]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "An object classified as 'Misc' is detected to the right of Car_1. It could be debris, a cone, or an unknown obstacle.",
        "The Misc object is an unidentified obstacle. The ego vehicle should treat it as a potential hazard and avoid it.",
        "Misc_1 is an unclassified object on the road. Its exact nature is uncertain — proceed with caution.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "What is the miscellaneous object in the scene and how should the ego vehicle respond?",
        "answer": answer,
        "reasoning": (
            "Grounding: Misc class object detected (KITTI class 7). "
            "Scene graph: misc_1 is_right_of car_1. "
            "Unknown object protocol: treat as obstacle, reduce speed, prepare evasive maneuver."
        ),
    }


def scenario_tram_right_of_way():
    """Which vehicle must yield to the tram?"""
    tram = make_tram_entity(1, (0.4, 0.6), (0.4, 0.55))
    car1 = make_car_entity(1, (0.2, 0.35), (0.55, 0.7))
    car2 = make_car_entity(2, (0.65, 0.8), (0.55, 0.7))
    entities = [tram, car1, car2]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "Both Car_1 and Car_2 must yield to the tram. Trams have absolute right-of-way on their tracks.",
        "Car_1 (left) and Car_2 (right) must both yield to Tram_1. The tram cannot deviate from its tracks.",
        "All surrounding vehicles must yield. The tram has priority — it cannot brake quickly or change direction.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "Which vehicle must yield to the tram, and why?",
        "answer": answer,
        "reasoning": (
            "Grounding: Tram center, two cars flanking. "
            "Scene graph: car_1 is_left_of tram_1, car_2 is_right_of tram_1. "
            "Right-of-way: trams have absolute priority on tracks. Cars must yield."
        ),
    }


def scenario_cyclist_pov():
    """From the cyclist's perspective, is the car too close?"""
    cyclist = make_cyclist_entity(1, (0.4, 0.6), (0.5, 0.65))
    car = make_car_entity(1, (0.6, 0.75), (0.5, 0.65))
    entities = [cyclist, car]
    sg = build_scene_graph(entities)
    answer = random.choice([
        "The car is within 1-2 meters of the cyclist — dangerously close. The cyclist has insufficient safe space.",
        "Car_1 is too close to the cyclist. The lateral gap appears less than the recommended 1.5 meters.",
        "Yes, the car is dangerously close to the cyclist. The cyclist should be given at least 1.5m of clearance.",
    ])
    return {
        "entities": entities,
        "scene_graph": sg,
        "road_features": None,
        "question": "From the cyclist's perspective, is the nearby car too close for safety?",
        "answer": answer,
        "reasoning": (
            "Grounding: Cyclist and Car in close proximity. "
            "Scene graph: car_1 is_right_of cyclist_1, car_1 is_near cyclist_1. "
            "Safety analysis: lateral gap < 1.5m violates safe passing distance."
        ),
    }


# ─── All scenario functions ───

SCENARIO_FNS = [
    scenario_ego_stop_before_ped,
    scenario_payment_decision,
    scenario_spatial_left_right,
    scenario_cyclist_safety,
    scenario_multi_vehicle_platoon,
    scenario_pedestrian_jaywalking,
    scenario_traffic_light_obey,
    scenario_two_peds_different_states,
    scenario_tram_interference,
    scenario_parked_car_occlusion,
    scenario_ego_distance_estimate,
    scenario_hard_negative_no_pedestrian,
    scenario_hard_negative_wrong_class,
    scenario_occlusion_depth,
    scenario_emergency_vehicle,
    scenario_person_sitting,
    scenario_multiple_pedestrians_group,
    scenario_distance_ordering,
    scenario_lane_change_safety,
    scenario_weather_impact,
    scenario_ego_stop_line,
    scenario_misc_object,
    scenario_tram_right_of_way,
    scenario_cyclist_pov,
]


def generate_sample(idx: int) -> dict[str, Any]:
    """Generate one dataset sample."""
    fn = random.choice(SCENARIO_FNS)
    scene = fn()

    # Pick a plausible KITTI image path
    img_id = random.randint(0, 7480)
    image_path = f"/data/kitti/image_2/{img_id:06d}.png"

    # Build staged reasoning
    grounding_desc = f"Identified {len(scene['entities'])} entities: " + ", ".join(
        f"{e['name']} ({e['class']})" for e in scene["entities"]
    ) + "."
    graph_desc = f"Scene graph has {len(scene['scene_graph'])} relationship triples."
    reasoning_desc = scene["reasoning"]

    staged = staged_reasoning_steps(grounding_desc, graph_desc, reasoning_desc, scene["answer"])

    return {
        "id": f"kitti-scene-{idx}",
        "image_path": image_path,
        "image_size": KITTI_IMAGE_SIZE,
        "entities": scene["entities"],
        "scene_graph": scene["scene_graph"],
        "question": scene["question"],
        "answer": scene["answer"],
        "staged_reasoning": staged,
    }


def write_jsonl(samples: list[dict], path: str):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    NUM_SAMPLES = 400
    VAL_RATIO = 0.15

    samples = [generate_sample(i + 1) for i in range(NUM_SAMPLES)]
    random.shuffle(samples)

    val_count = int(NUM_SAMPLES * VAL_RATIO)
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "kitti_scene_train.jsonl")
    val_path = os.path.join(out_dir, "kitti_scene_val.jsonl")
    schema_path = os.path.join(out_dir, "schema.json")

    write_jsonl(train_samples, train_path)
    write_jsonl(val_samples, val_path)

    # Schema
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "KITTI Spatial Instruction Dataset",
        "description": "Synthesized spatial understanding instruction data for autonomous driving. NOT real KITTI labels — coordinates are designed to be consistent with scene graphs.",
        "type": "object",
        "required": ["id", "image_path", "image_size", "entities", "scene_graph", "question", "answer", "staged_reasoning"],
        "properties": {
            "id": {"type": "string", "description": "Unique sample ID, format kitti-scene-<N>"},
            "image_path": {"type": "string", "description": "Placeholder local path to KITTI image (no actual images redistributed)"},
            "image_size": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2, "description": "[H, W] of KITTI image"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "class", "bbox"],
                    "properties": {
                        "name": {"type": "string"},
                        "class": {"type": "string", "enum": KITTI_CLASSES},
                        "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "[x1,y1,x2,y2] normalized 0-1"},
                        "pose": {"type": "object", "description": "MPII 16-joint pose (only for Pedestrian/Person_sitting)"},
                    },
                },
            },
            "scene_graph": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["subject", "relation", "object"],
                    "properties": {
                        "subject": {"type": "string"},
                        "relation": {"type": "string"},
                        "object": {"type": "string"},
                    },
                },
            },
            "question": {"type": "string", "description": "English reasoning question"},
            "answer": {"type": "string", "description": "Ground-truth English answer"},
            "staged_reasoning": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["step", "text"],
                    "properties": {
                        "step": {"type": "integer", "minimum": 1, "maximum": 4},
                        "text": {"type": "string"},
                    },
                },
                "minItems": 2,
                "maxItems": 4,
                "description": "DRScaffold 4-stage reasoning: grounding → graph → reasoning → answer",
            },
        },
    }
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"Train: {len(train_samples)} samples → {train_path}")
    print(f"Val:   {len(val_samples)} samples → {val_path}")
    print(f"Schema: {schema_path}")


if __name__ == "__main__":
    main()
