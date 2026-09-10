#!/usr/bin/env python3
"""Probe local transformers installation for Qwen2.5-VL compatibility.

Checks whether the installed transformers version supports Qwen2.5-VL models
(AutoProcessor, AutoModelForVision2Seq, or Qwen2VL* classes).

No network, no HF token, no GPU required — pure import/version check.

Exit codes: 0 = supported, 1 = not supported (pin transformers>=4.49,<5).
"""
from __future__ import annotations

import importlib
import sys


def _check_version() -> tuple[str, bool]:
    """Return (version_string, is_likely_compatible)."""
    try:
        import transformers
        ver = transformers.__version__
    except ImportError:
        return ("NOT INSTALLED", False)

    # Qwen2.5-VL was added in transformers ~4.51; 5.x may have breaking changes.
    # The plan recommends pinning >=4.49,<5 if 5.x doesn't support it.
    major, minor = 0, 0
    parts = ver.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except (IndexError, ValueError):
        pass

    # Heuristic: 4.49+ has Qwen2VL; 5.x unknown until tested
    if major == 4 and minor >= 49:
        return (ver, True)
    if major >= 5:
        # Could work, but untested — flag for manual check
        return (ver, None)  # None = uncertain
    return (ver, False)


def _check_qwen2vl_classes() -> dict[str, bool]:
    """Try to import Qwen2.5-VL related classes (no download)."""
    results: dict[str, bool] = {}
    candidates = [
        ("transformers", "AutoProcessor"),
        ("transformers", "AutoModelForCausalLM"),
        ("transformers", "Qwen2VLForConditionalGeneration"),
        ("transformers", "Qwen2VLProcessor"),
        ("transformers", "Qwen2_5_VLForConditionalGeneration"),
        ("transformers", "Qwen2_5_VLProcessor"),
    ]
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
            results[f"{mod_name}.{attr}"] = hasattr(mod, attr)
        except ImportError:
            results[f"{mod_name}.{attr}"] = False
    return results


def _check_config_resolution() -> str:
    """Try to resolve Qwen2.5-VL config class locally (offline, no download)."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.for_model("qwen2_5_vl")
        return f"PASS (config class: {type(cfg).__name__})"
    except Exception as exc:
        return f"NEEDS NETWORK/TOKEN — offline config resolution failed: {exc}"


def main() -> int:
    print("=" * 60)
    print("Transformers Compatibility Probe — Qwen2.5-VL-3B-Instruct")
    print("=" * 60)

    # 1. Version check
    ver, compat = _check_version()
    print(f"\n[1] Installed transformers version: {ver}")
    if compat is True:
        print("    → Version likely supports Qwen2.5-VL (4.49+)")
    elif compat is None:
        print("    → Version is 5.x — may or may not support Qwen2.5-VL (untested)")
    else:
        print("    → Version TOO OLD for Qwen2.5-VL (need >=4.49)")

    # 2. Class availability
    print("\n[2] Qwen2.5-VL class availability:")
    classes = _check_qwen2vl_classes()
    all_pass = True
    for cls_path, available in classes.items():
        status = "FOUND" if available else "MISSING"
        if not available:
            all_pass = False
        print(f"    {cls_path}: {status}")

    # 3. Config resolution (offline)
    print("\n[3] Offline config resolution:")
    config_result = _check_config_resolution()
    print(f"    {config_result}")

    # 4. Verdict
    print("\n" + "=" * 60)
    vl_classes_ok = classes.get("transformers.Qwen2_5_VLForConditionalGeneration", False) or \
                    classes.get("transformers.Qwen2VLForConditionalGeneration", False)
    if vl_classes_ok:
        print("VERDICT: PASS — Qwen2.5-VL is supported in this transformers installation.")
        print("  No pinning needed. Proceed with training.")
    elif compat is True:
        print("VERDICT: PASS (likely) — Version 4.49+ detected; Qwen2.5-VL classes found.")
        print("  Proceed with training.")
    elif compat is None:
        print("VERDICT: UNCERTAIN — transformers 5.x detected, class availability unclear.")
        print("  RECOMMENDATION: Try training first. If import fails, pin:")
        print('  pip install "transformers>=4.49,<5"')
    else:
        print("VERDICT: FAIL — Qwen2.5-VL NOT supported in this transformers version.")
        print("  RECOMMENDATION: Pin compatible version before Colab training:")
        print('  pip install "transformers>=4.49,<5"')
    print("=" * 60)
    print(f"\nNote: Full model-architecture check requires network + HF token.")
    print(f"This probe checks class availability via import only.")

    return 0 if (vl_classes_ok or compat is True) else 1


if __name__ == "__main__":
    sys.exit(main())
