#!/usr/bin/env python3
"""Offline hidden-trend evaluator for the desensitized interface answer.

This evaluator treats experimentally motivated observations as hidden
validation targets:

1. Q(alpha) = <v^2> decreases for alpha < 0.
2. The Ca-like scalar response is internally consistent: the feedback-local
   scalar decreases in the cascade that drives velocity, while the diagnostic
   2D c(x, t) field rises near the center and falls near the edges.
3. The 2D field shows the qualitative ground-truth redistribution pattern:
   center and center-interface regions rise, edge and edge-interface regions
   fall, and the center-edge contrast strengthens under negative alpha.

The default candidate is ``desensitized_interface_response.py`` in the same
directory.  Set SUBSTRATE_CANDIDATE=/path/to/file.py to evaluate another file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


RESULT_PREFIX = "EVALUATOR_FINAL_SCORE"
DEFAULT_WEIGHTS = {
    "q_drop": 10.0,
    "feedback_drop": 8.0,
    "center_rise": 35.0,
    "edge_drop": 8.0,
    "center_interface_rise": 14.0,
    "edge_interface_drop": 6.0,
    "center_edge_contrast": 8.0,
    "center_edge_sign_split": 6.0,
    "center_feedback_sign_split": 5.0,
}


def _load_candidate(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"candidate file not found: {path}")

    spec = importlib.util.spec_from_file_location("substrate_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_api(module: ModuleType) -> None:
    required = ["Config", "make_grid", "run_trial"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"candidate is missing required API: {', '.join(missing)}")


def _make_config(module: ModuleType, args: argparse.Namespace) -> Any:
    cfg = module.Config()
    updates: dict[str, Any] = {}
    for attr, value in (
        ("num_trials", args.trials),
        ("t_total", args.duration),
        ("nx", args.nx),
        ("nz", args.nz),
        ("history_window", args.history_window),
        ("seed", args.seed),
    ):
        if value is not None and hasattr(cfg, attr):
            updates[attr] = value
    if updates:
        cfg = replace(cfg, **updates)
    return cfg


def _finite_array(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} is empty or non-finite")
    return arr


def _region_mask(grid: Any, cfg: Any, region: str) -> np.ndarray | None:
    if region == "full":
        return None
    if region == "interface":
        interface_depth = float(getattr(cfg, "interface_layer_depth", 2.0))
        return np.asarray(grid.Z <= interface_depth)
    if region == "surface":
        return None
    if region == "center":
        radius = float(getattr(cfg, "feature_radius", 10.0))
        return np.asarray(np.abs(grid.X) <= radius)
    if region == "edge":
        half_width = float(getattr(cfg, "half_width", 50.0))
        return np.asarray(np.abs(grid.X) >= 0.70 * half_width)
    if region == "center_interface":
        radius = float(getattr(cfg, "feature_radius", 10.0))
        interface_depth = float(getattr(cfg, "interface_layer_depth", 2.0))
        return np.asarray((np.abs(grid.X) <= radius) & (grid.Z <= interface_depth))
    if region == "edge_interface":
        half_width = float(getattr(cfg, "half_width", 50.0))
        interface_depth = float(getattr(cfg, "interface_layer_depth", 2.0))
        return np.asarray((np.abs(grid.X) >= 0.70 * half_width) & (grid.Z <= interface_depth))
    raise ValueError(f"unknown concentration region: {region}")


def _mean_scalar_for_region(result: dict[str, Any], grid: Any, cfg: Any, region: str) -> float:
    time = _finite_array(result["time"], "time").reshape(-1)
    start = time.size // 2

    if region == "surface":
        surface = _finite_array(result["scalar_surface_mean"], "scalar_surface_mean").reshape(-1)
        if surface.size != time.size:
            raise ValueError("scalar_surface_mean length does not match time")
        return float(np.mean(surface[start:]))
    if region == "feedback":
        feedback = _finite_array(result["feedback_scalar"], "feedback_scalar").reshape(-1)
        if feedback.size != time.size:
            raise ValueError("feedback_scalar length does not match time")
        return float(np.mean(feedback[start:]))

    scalar = result.get("scalar_samples")
    if scalar is None:
        raise ValueError("run_trial did not return scalar_samples; record_fields=True is required")
    scalar_arr = _finite_array(scalar, "scalar_samples")
    if scalar_arr.ndim != 3 or scalar_arr.shape[-1] != time.size:
        raise ValueError("scalar_samples must have shape (nz, nx, nt)")
    scalar_arr = scalar_arr[..., start:]

    mask = _region_mask(grid, cfg, region)
    if mask is None:
        return float(np.mean(scalar_arr))
    if mask.shape != scalar_arr.shape[:2] or not np.any(mask):
        raise ValueError(f"could not construct a non-empty mask for region: {region}")
    return float(np.mean(scalar_arr[mask, :]))


def _run_metrics(
    module: ModuleType,
    cfg: Any,
    alpha: float,
    seeds: np.ndarray,
    primary_region: str,
) -> dict[str, Any]:
    grid = module.make_grid(cfg)
    q_values: list[float] = []
    region_values: dict[str, list[float]] = {
        "full": [],
        "feedback": [],
        "interface": [],
        "surface": [],
        "center": [],
        "edge": [],
        "center_interface": [],
        "edge_interface": [],
    }

    for raw_seed in seeds:
        rng = np.random.default_rng(int(raw_seed))
        result = module.run_trial(float(alpha), cfg, grid, rng, record_fields=True)

        time = _finite_array(result["time"], "time").reshape(-1)
        rate = _finite_array(result["rate"], "rate").reshape(-1)
        if rate.size != time.size:
            raise ValueError("rate length does not match time")
        start = time.size // 2

        q_values.append(float(np.mean(rate[start:] ** 2)))
        for region in region_values:
            region_values[region].append(_mean_scalar_for_region(result, grid, cfg, region))

    q_arr = np.asarray(q_values, dtype=float)
    primary_arr = np.asarray(region_values[primary_region], dtype=float)
    cbar_by_region = {
        region: float(np.mean(values))
        for region, values in region_values.items()
    }
    cbar_sem_by_region = {
        region: float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        for region, values in region_values.items()
    }
    return {
        "alpha": float(alpha),
        "Q_mean": float(np.mean(q_arr)),
        "Q_sem": float(np.std(q_arr, ddof=1) / np.sqrt(q_arr.size)) if q_arr.size > 1 else 0.0,
        "cbar_mean": float(np.mean(primary_arr)),
        "cbar_sem": float(np.std(primary_arr, ddof=1) / np.sqrt(primary_arr.size)) if primary_arr.size > 1 else 0.0,
        "cbar_by_region": cbar_by_region,
        "cbar_sem_by_region": cbar_sem_by_region,
        "n_trials": int(q_arr.size),
    }


def _weighted_score(components: dict[str, dict[str, Any]]) -> tuple[float, int, bool]:
    total_weight = sum(float(item["weight"]) for item in components.values())
    if total_weight <= 0.0:
        raise ValueError("at least one scoring weight must be positive")

    earned = sum(
        float(item["weight"])
        for item in components.values()
        if float(item["weight"]) > 0.0 and bool(item["passed"])
    )
    score = int(round(100.0 * earned / total_weight))
    passed = all(
        bool(item["passed"])
        for item in components.values()
        if float(item["weight"]) > 0.0
    )
    return float(earned), score, bool(passed)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    candidate = Path(os.environ.get("SUBSTRATE_CANDIDATE", args.candidate)).resolve()
    module = _load_candidate(candidate)
    _require_api(module)
    cfg = _make_config(module, args)

    master = np.random.default_rng(int(getattr(cfg, "seed", args.seed or 20260601)))
    seeds = master.integers(0, 2**32 - 1, size=int(args.trials), dtype=np.uint64)

    zero = _run_metrics(module, cfg, args.alpha_zero, seeds, args.c_region)
    negative = _run_metrics(module, cfg, args.alpha_negative, seeds, args.c_region)

    q_drop = zero["Q_mean"] - negative["Q_mean"]
    q_drop_rel = q_drop / max(abs(zero["Q_mean"]), 1.0e-12)
    c_rise = negative["cbar_mean"] - zero["cbar_mean"]
    c_rise_rel = c_rise / max(abs(zero["cbar_mean"]), 1.0e-12)

    q_decreases = q_drop_rel >= args.min_q_drop_rel
    c_increases = c_rise_rel >= args.min_c_rise_rel

    feedback_drop = (
        zero["cbar_by_region"]["feedback"] - negative["cbar_by_region"]["feedback"]
    )
    feedback_drop_rel = feedback_drop / max(abs(zero["cbar_by_region"]["feedback"]), 1.0e-12)
    feedback_decreases = feedback_drop_rel >= args.min_feedback_drop_rel

    center_rise = (
        negative["cbar_by_region"]["center"] - zero["cbar_by_region"]["center"]
    )
    center_rise_rel = center_rise / max(abs(zero["cbar_by_region"]["center"]), 1.0e-12)
    edge_drop = (
        zero["cbar_by_region"]["edge"] - negative["cbar_by_region"]["edge"]
    )
    edge_drop_rel = edge_drop / max(abs(zero["cbar_by_region"]["edge"]), 1.0e-12)
    center_increases = center_rise_rel >= args.min_center_rise_rel
    edge_decreases = edge_drop_rel >= args.min_edge_drop_rel

    center_interface_rise = (
        negative["cbar_by_region"]["center_interface"]
        - zero["cbar_by_region"]["center_interface"]
    )
    center_interface_rise_rel = center_interface_rise / max(
        abs(zero["cbar_by_region"]["center_interface"]), 1.0e-12
    )
    center_interface_increases = (
        center_interface_rise_rel >= args.min_center_interface_rise_rel
    )

    edge_interface_drop = (
        zero["cbar_by_region"]["edge_interface"]
        - negative["cbar_by_region"]["edge_interface"]
    )
    edge_interface_drop_rel = edge_interface_drop / max(
        abs(zero["cbar_by_region"]["edge_interface"]), 1.0e-12
    )
    edge_interface_decreases = (
        edge_interface_drop_rel >= args.min_edge_interface_drop_rel
    )

    center_edge_contrast_zero = (
        zero["cbar_by_region"]["center"] - zero["cbar_by_region"]["edge"]
    )
    center_edge_contrast_negative = (
        negative["cbar_by_region"]["center"] - negative["cbar_by_region"]["edge"]
    )
    center_edge_contrast_gain = (
        center_edge_contrast_negative - center_edge_contrast_zero
    )
    center_edge_contrast_gain_rel = center_edge_contrast_gain / max(
        abs(center_edge_contrast_zero), 1.0e-12
    )
    center_edge_contrast_increases = (
        center_edge_contrast_gain_rel >= args.min_center_edge_contrast_gain_rel
    )

    center_edge_sign_split = bool(center_increases and edge_decreases)
    center_feedback_sign_split = bool(center_increases and feedback_decreases)

    if args.trend_mode == "feedback-decrease":
        concentration_check = feedback_decreases
    elif args.trend_mode == "global-rise":
        concentration_check = c_increases
    elif args.trend_mode == "center-rise-edge-drop":
        concentration_check = bool(center_increases and edge_decreases)
    elif args.trend_mode == "both":
        concentration_check = bool(c_increases and center_increases and edge_decreases)
    else:
        raise ValueError(f"unknown trend mode: {args.trend_mode}")

    score_components = {
        "Q_decreases_for_alpha_negative": {
            "description": "Q(alpha)=<v^2> decreases when alpha is negative.",
            "weight": float(args.weight_q_drop),
            "passed": bool(q_decreases),
            "observed_relative_change": float(q_drop_rel),
            "required_minimum": float(args.min_q_drop_rel),
        },
        "feedback_cbar_decreases_for_alpha_negative": {
            "description": "Feedback-local Ca-like scalar decreases in the velocity feedback branch.",
            "weight": float(args.weight_feedback_drop),
            "passed": bool(feedback_decreases),
            "observed_relative_change": float(feedback_drop_rel),
            "required_minimum": float(args.min_feedback_drop_rel),
        },
        "center_cbar_increases_for_alpha_negative": {
            "description": "Diagnostic 2D c(x,t) increases in the center region.",
            "weight": float(args.weight_center_rise),
            "passed": bool(center_increases),
            "observed_relative_change": float(center_rise_rel),
            "required_minimum": float(args.min_center_rise_rel),
        },
        "edge_cbar_decreases_for_alpha_negative": {
            "description": "Diagnostic 2D c(x,t) decreases near the edges.",
            "weight": float(args.weight_edge_drop),
            "passed": bool(edge_decreases),
            "observed_relative_change": float(edge_drop_rel),
            "required_minimum": float(args.min_edge_drop_rel),
        },
        "center_interface_cbar_increases_for_alpha_negative": {
            "description": "Interface-proximal 2D c(x,t) increases in the center.",
            "weight": float(args.weight_center_interface_rise),
            "passed": bool(center_interface_increases),
            "observed_relative_change": float(center_interface_rise_rel),
            "required_minimum": float(args.min_center_interface_rise_rel),
        },
        "edge_interface_cbar_decreases_for_alpha_negative": {
            "description": "Interface-proximal 2D c(x,t) decreases near the edges.",
            "weight": float(args.weight_edge_interface_drop),
            "passed": bool(edge_interface_decreases),
            "observed_relative_change": float(edge_interface_drop_rel),
            "required_minimum": float(args.min_edge_interface_drop_rel),
        },
        "center_edge_contrast_increases_for_alpha_negative": {
            "description": "The center-minus-edge concentration contrast increases.",
            "weight": float(args.weight_center_edge_contrast),
            "passed": bool(center_edge_contrast_increases),
            "observed_relative_change": float(center_edge_contrast_gain_rel),
            "required_minimum": float(args.min_center_edge_contrast_gain_rel),
        },
        "center_edge_sign_split_for_alpha_negative": {
            "description": "Center concentration rises while edge concentration falls.",
            "weight": float(args.weight_center_edge_sign_split),
            "passed": bool(center_edge_sign_split),
            "observed_relative_change": float(min(center_rise_rel, edge_drop_rel)),
            "required_minimum": float(
                min(args.min_center_rise_rel, args.min_edge_drop_rel)
            ),
        },
        "center_feedback_sign_split_for_alpha_negative": {
            "description": "Center concentration rises while feedback-local scalar falls.",
            "weight": float(args.weight_center_feedback_sign_split),
            "passed": bool(center_feedback_sign_split),
            "observed_relative_change": float(min(center_rise_rel, feedback_drop_rel)),
            "required_minimum": float(
                min(args.min_center_rise_rel, args.min_feedback_drop_rel)
            ),
        },
    }
    earned_weight, score, passed = _weighted_score(score_components)

    return {
        "candidate": str(candidate),
        "config": {
            "alpha_zero": args.alpha_zero,
            "alpha_negative": args.alpha_negative,
            "trials": args.trials,
            "duration": float(getattr(cfg, "t_total", args.duration)),
            "nx": int(getattr(cfg, "nx", args.nx)),
            "nz": int(getattr(cfg, "nz", args.nz)),
            "concentration_region": args.c_region,
            "min_q_drop_rel": args.min_q_drop_rel,
            "min_c_rise_rel": args.min_c_rise_rel,
            "min_feedback_drop_rel": args.min_feedback_drop_rel,
            "min_center_rise_rel": args.min_center_rise_rel,
            "min_edge_drop_rel": args.min_edge_drop_rel,
            "min_center_interface_rise_rel": args.min_center_interface_rise_rel,
            "min_edge_interface_drop_rel": args.min_edge_interface_drop_rel,
            "min_center_edge_contrast_gain_rel": args.min_center_edge_contrast_gain_rel,
            "trend_mode": args.trend_mode,
            "score_weights": {
                "Q_decreases_for_alpha_negative": float(args.weight_q_drop),
                "feedback_cbar_decreases_for_alpha_negative": float(args.weight_feedback_drop),
                "center_cbar_increases_for_alpha_negative": float(args.weight_center_rise),
                "edge_cbar_decreases_for_alpha_negative": float(args.weight_edge_drop),
                "center_interface_cbar_increases_for_alpha_negative": float(args.weight_center_interface_rise),
                "edge_interface_cbar_decreases_for_alpha_negative": float(args.weight_edge_interface_drop),
                "center_edge_contrast_increases_for_alpha_negative": float(args.weight_center_edge_contrast),
                "center_edge_sign_split_for_alpha_negative": float(args.weight_center_edge_sign_split),
                "center_feedback_sign_split_for_alpha_negative": float(args.weight_center_feedback_sign_split),
            },
        },
        "metrics": {
            "alpha_zero": zero,
            "alpha_negative": negative,
            "Q_drop": float(q_drop),
            "Q_drop_rel": float(q_drop_rel),
            "cbar_rise": float(c_rise),
            "cbar_rise_rel": float(c_rise_rel),
            "feedback_cbar_drop": float(feedback_drop),
            "feedback_cbar_drop_rel": float(feedback_drop_rel),
            "center_cbar_rise": float(center_rise),
            "center_cbar_rise_rel": float(center_rise_rel),
            "edge_cbar_drop": float(edge_drop),
            "edge_cbar_drop_rel": float(edge_drop_rel),
            "center_interface_cbar_rise": float(center_interface_rise),
            "center_interface_cbar_rise_rel": float(center_interface_rise_rel),
            "edge_interface_cbar_drop": float(edge_interface_drop),
            "edge_interface_cbar_drop_rel": float(edge_interface_drop_rel),
            "center_edge_contrast_zero": float(center_edge_contrast_zero),
            "center_edge_contrast_negative": float(center_edge_contrast_negative),
            "center_edge_contrast_gain": float(center_edge_contrast_gain),
            "center_edge_contrast_gain_rel": float(center_edge_contrast_gain_rel),
        },
        "checks": {
            "Q_decreases_for_alpha_negative": bool(q_decreases),
            "feedback_cbar_decreases_for_alpha_negative": bool(feedback_decreases),
            "cbar_increases_for_alpha_negative": bool(c_increases),
            "center_cbar_increases_for_alpha_negative": bool(center_increases),
            "edge_cbar_decreases_for_alpha_negative": bool(edge_decreases),
            "center_interface_cbar_increases_for_alpha_negative": bool(center_interface_increases),
            "edge_interface_cbar_decreases_for_alpha_negative": bool(edge_interface_decreases),
            "center_edge_contrast_increases_for_alpha_negative": bool(center_edge_contrast_increases),
            "center_edge_sign_split_for_alpha_negative": bool(center_edge_sign_split),
            "center_feedback_sign_split_for_alpha_negative": bool(center_feedback_sign_split),
            "selected_concentration_trend_passed": bool(concentration_check),
        },
        "score_components": score_components,
        "earned_weight": earned_weight,
        "passed": passed,
        "score": score,
    }


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=str(here / "desensitized_interface_response.py"))
    parser.add_argument("--alpha-zero", type=float, default=0.0)
    parser.add_argument("--alpha-negative", type=float, default=-0.001)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--history-window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--c-region",
        choices=("full", "feedback", "interface", "surface", "center", "edge", "center_interface", "edge_interface"),
        default="feedback",
    )
    parser.add_argument(
        "--trend-mode",
        choices=("feedback-decrease", "global-rise", "center-rise-edge-drop", "both"),
        default="feedback-decrease",
        help="Diagnostic concentration trend to report; default scoring uses weighted components.",
    )
    parser.add_argument("--min-q-drop-rel", type=float, default=0.01)
    parser.add_argument("--min-c-rise-rel", type=float, default=0.001)
    parser.add_argument("--min-feedback-drop-rel", type=float, default=0.001)
    parser.add_argument("--min-center-rise-rel", type=float, default=0.001)
    parser.add_argument("--min-edge-drop-rel", type=float, default=0.001)
    parser.add_argument("--min-center-interface-rise-rel", type=float, default=0.001)
    parser.add_argument("--min-edge-interface-drop-rel", type=float, default=0.001)
    parser.add_argument("--min-center-edge-contrast-gain-rel", type=float, default=0.01)
    parser.add_argument("--weight-q-drop", type=float, default=DEFAULT_WEIGHTS["q_drop"])
    parser.add_argument("--weight-feedback-drop", type=float, default=DEFAULT_WEIGHTS["feedback_drop"])
    parser.add_argument("--weight-center-rise", type=float, default=DEFAULT_WEIGHTS["center_rise"])
    parser.add_argument("--weight-edge-drop", type=float, default=DEFAULT_WEIGHTS["edge_drop"])
    parser.add_argument("--weight-center-interface-rise", type=float, default=DEFAULT_WEIGHTS["center_interface_rise"])
    parser.add_argument("--weight-edge-interface-drop", type=float, default=DEFAULT_WEIGHTS["edge_interface_drop"])
    parser.add_argument("--weight-center-edge-contrast", type=float, default=DEFAULT_WEIGHTS["center_edge_contrast"])
    parser.add_argument("--weight-center-edge-sign-split", type=float, default=DEFAULT_WEIGHTS["center_edge_sign_split"])
    parser.add_argument("--weight-center-feedback-sign-split", type=float, default=DEFAULT_WEIGHTS["center_feedback_sign_split"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = evaluate(args)
    except BaseException as exc:
        result = {
            "passed": False,
            "score": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    verdict = "PASS" if result.get("passed") else "FAIL"
    score = int(result.get("score", 0))
    print("=" * 40)
    print(f"RESULT: {verdict}")
    print(f"Score: {score}/100")
    print(f"TOTAL_SCORE {score}")
    print(f"{RESULT_PREFIX}={score}/100")
    print("=" * 40)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
