#!/usr/bin/env python3
"""Validate the spatial instruction dataset: loadable by HF datasets + schema key checks."""
import json
import sys

sys.path.insert(0, ".")
from datasets import load_dataset

REQUIRED = ["id", "image_path", "image_size", "entities", "scene_graph", "question", "answer", "staged_reasoning"]

for name, path in [("train", "data/kitti_scene_train.jsonl"), ("val", "data/kitti_scene_val.jsonl")]:
    ds = load_dataset("json", data_files=path, split="train")
    print(f"[load_dataset] {name}: {ds.num_rows} rows, features={list(ds.features.keys())}")

    missing_any = 0
    missing_by_key = {k: 0 for k in REQUIRED}
    bad_relations = 0
    no_entities = 0

    for i in range(ds.num_rows):
        row = ds[i]
        for k in REQUIRED:
            if k not in row or row[k] is None:
                missing_by_key[k] += 1
                missing_any += 1
        if not row.get("entities"):
            no_entities += 1
        ent_names = {e["name"] for e in row.get("entities", [])}
        for triple in row.get("scene_graph", []):
            if triple["subject"] not in ent_names or triple["object"].startswith("crosswalk") or triple["object"].startswith("traffic_light") or triple["object"].startswith("stop_line"):
                continue
            if triple["object"] not in ent_names and not triple["object"].startswith(("crosswalk", "traffic_light", "stop_line", "ego_vehicle")):
                bad_relations += 1

    print(f"  rows missing any required key: {missing_any}")
    print(f"  missing by key: {missing_by_key}")
    print(f"  rows with no entities: {no_entities}")
    if bad_relations:
        print(f"  [WARN] triplets referencing unknown object: {bad_relations}")

# Print one sample row
ds = load_dataset("json", data_files="data/kitti_scene_train.jsonl", split="train")
print("\n=== Sample row (train[0]) ===")
print(json.dumps(ds[0], indent=2, ensure_ascii=False)[:2500])
print("\nVALIDATION PASSED" if missing_any == 0 and no_entities == 0 and bad_relations == 0 else "\nVALIDATION FAILED")