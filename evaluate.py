#!/usr/bin/env python3
"""Binary public evaluator for the coupled interface-response answer.

The public output is intentionally limited to PASS/FAIL in ``score_sum``
format.  Internal continuous diagnostics are written only to a private JSONL
path controlled by the judge backend and are never printed to stdout/stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


DEFAULT_WEIGHTS = {
    "criterion_01": 10.0,
    "criterion_02": 8.0,
    "criterion_03": 12.0,
    "criterion_04": 8.0,
    "criterion_05": 10.0,
    "criterion_06": 6.0,
    "criterion_07": 8.0,
    "criterion_08": 6.0,
    "criterion_09": 5.0,
    "criterion_10": 9.0,
    "criterion_11": 8.0,
    "criterion_12": 5.0,
    "criterion_13": 5.0,
}


_TRUSTED_STDOUT = sys.stdout
_TRUSTED_STDERR = sys.stderr


@contextlib.contextmanager
def _suppress_untrusted_output():
    """Hide stdout/stderr produced while importing or running submissions."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass

    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        sys.stdout = _TRUSTED_STDOUT
        sys.stderr = _TRUSTED_STDERR


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
    if region == "zone_a":
        radius = float(getattr(cfg, "feature_radius", 10.0))
        return np.asarray(np.abs(grid.X) <= radius)
    if region == "zone_b":
        half_width = float(getattr(cfg, "half_width", 50.0))
        return np.asarray(np.abs(grid.X) >= 0.70 * half_width)
    if region == "zone_a_interface":
        radius = float(getattr(cfg, "feature_radius", 10.0))
        interface_depth = float(getattr(cfg, "interface_layer_depth", 2.0))
        return np.asarray((np.abs(grid.X) <= radius) & (grid.Z <= interface_depth))
    if region == "zone_b_interface":
        half_width = float(getattr(cfg, "half_width", 50.0))
        interface_depth = float(getattr(cfg, "interface_layer_depth", 2.0))
        return np.asarray((np.abs(grid.X) >= 0.70 * half_width) & (grid.Z <= interface_depth))
    raise ValueError(f"unknown scalar region: {region}")


def _field_structure_diagnostics(result: dict[str, Any], grid: Any, cfg: Any) -> dict[str, float]:
    time = _finite_array(result["time"], "time").reshape(-1)
    feedback = _finite_array(result["feedback_scalar"], "feedback_scalar").reshape(-1)
    surface = _finite_array(result["scalar_surface_mean"], "scalar_surface_mean").reshape(-1)
    scalar = result.get("scalar_samples")
    if scalar is None:
        raise ValueError("run_trial did not return scalar_samples; record_fields=True is required")

    scalar_arr = _finite_array(scalar, "scalar_samples")
    if scalar_arr.ndim != 3 or scalar_arr.shape[-1] != time.size:
        raise ValueError("scalar_samples must have shape (nz, nx, nt)")
    if feedback.size != time.size or surface.size != time.size:
        raise ValueError("diagnostic time-series lengths must match time")

    start = max(0, time.size // 2)
    segment = scalar_arr[..., start:]
    mean_field = np.mean(segment, axis=-1)
    temporal_std = np.std(segment, axis=-1)
    field_mean = float(np.mean(segment))
    field_scale = max(abs(field_mean), 1.0e-12)
    field_std_rel = float(np.std(segment) / field_scale)
    temporal_activity_rel = float(np.mean(temporal_std) / field_scale)

    if mean_field.shape[1] > 2:
        first_x = np.mean(np.abs(np.diff(mean_field, axis=1)))
        second_x = np.mean(np.abs(np.diff(mean_field, n=2, axis=1)))
        roughness_ratio = float(second_x / max(first_x, 1.0e-12))
    else:
        roughness_ratio = 0.0

    omega = float(getattr(cfg, "omega", 0.12))
    t_segment = time[start:]
    centered = segment - mean_field[..., None]
    if t_segment.size >= 3:
        kernel = np.exp(-1j * omega * t_segment)
        harmonic = np.abs(
            (2.0 / t_segment.size)
            * np.tensordot(centered, kernel, axes=([-1], [0]))
        )
        harmonic_rel = float(np.mean(harmonic) / field_scale)
    else:
        harmonic_rel = 0.0

    interface_depth = float(getattr(cfg, "interface_layer_depth", 2.0))
    depth = float(getattr(cfg, "depth", np.max(grid.Z)))
    interface_mask = np.asarray(grid.Z <= interface_depth)
    deep_mask = np.asarray(grid.Z >= 0.70 * depth)
    if interface_mask.shape != mean_field.shape or deep_mask.shape != mean_field.shape:
        raise ValueError("grid shape does not match scalar_samples")
    interface_deep_contrast_rel = float(
        (np.mean(mean_field[interface_mask]) - np.mean(mean_field[deep_mask]))
        / field_scale
    )

    local_surface_separation_rel = float(
        np.mean(np.abs(feedback[start:] - surface[start:]))
        / max(abs(float(np.mean(surface[start:]))), 1.0e-12)
    )

    return {
        "field_std_rel": field_std_rel,
        "temporal_activity_rel": temporal_activity_rel,
        "roughness_ratio": roughness_ratio,
        "harmonic_rel": harmonic_rel,
        "interface_deep_contrast_rel": interface_deep_contrast_rel,
        "local_surface_separation_rel": local_surface_separation_rel,
    }


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
    with _suppress_untrusted_output():
        grid = module.make_grid(cfg)
    q_values: list[float] = []
    region_values: dict[str, list[float]] = {
        "full": [],
        "feedback": [],
        "interface": [],
        "surface": [],
        "zone_a": [],
        "zone_b": [],
        "zone_a_interface": [],
        "zone_b_interface": [],
    }
    field_diagnostics: dict[str, list[float]] = {
        "field_std_rel": [],
        "temporal_activity_rel": [],
        "roughness_ratio": [],
        "harmonic_rel": [],
        "interface_deep_contrast_rel": [],
        "local_surface_separation_rel": [],
    }

    for raw_seed in seeds:
        rng = np.random.default_rng(int(raw_seed))
        with _suppress_untrusted_output():
            result = module.run_trial(float(alpha), cfg, grid, rng, record_fields=True)

        time = _finite_array(result["time"], "time").reshape(-1)
        rate = _finite_array(result["rate"], "rate").reshape(-1)
        if rate.size != time.size:
            raise ValueError("rate length does not match time")
        start = time.size // 2

        q_values.append(float(np.mean(rate[start:] ** 2)))
        for region in region_values:
            region_values[region].append(_mean_scalar_for_region(result, grid, cfg, region))
        diagnostics = _field_structure_diagnostics(result, grid, cfg)
        for key, value in diagnostics.items():
            field_diagnostics[key].append(float(value))

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
        "field_diagnostics": {
            key: float(np.mean(values))
            for key, values in field_diagnostics.items()
        },
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


def _candidate_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def _scrub_private_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the private record useful without exposing model-specific names."""

    components = result.get("score_components") or {}
    scrubbed_components: dict[str, Any] = {}
    for index, item in enumerate(components.values(), start=1):
        scrubbed_components[f"criterion_{index:02d}"] = {
            "passed": bool(item.get("passed", False)),
            "weight": float(item.get("weight", 0.0)),
            "observed": float(item.get("observed_relative_change", 0.0)),
            "threshold": float(item.get("required_minimum", 0.0)),
        }

    private_metrics = result.get("metrics") or {}
    return {
        "candidate": result.get("candidate"),
        "candidate_sha256": _candidate_digest(Path(str(result.get("candidate", "")))),
        "raw_score": int(result.get("score", 0)),
        "passed": bool(result.get("passed", False)),
        "earned_weight": float(result.get("earned_weight", 0.0)),
        "error": result.get("error"),
        "criteria": scrubbed_components,
        "numeric_snapshot": {
            "metric_01": float(private_metrics.get("metric_01_abs_rel", 0.0)),
            "metric_02": float(private_metrics.get("metric_02_abs_rel", 0.0)),
            "metric_03": float(private_metrics.get("metric_03", 0.0)),
            "metric_04": float(private_metrics.get("metric_04_abs_rel", 0.0)),
            "metric_05": float(private_metrics.get("metric_05", 0.0)),
            "metric_06": float(private_metrics.get("metric_06_abs_rel", 0.0)),
            "metric_07": float(private_metrics.get("metric_07_abs_rel", 0.0)),
            "metric_08": float(private_metrics.get("metric_08", 0.0)),
            "metric_09": float(private_metrics.get("metric_09", 0.0)),
            "metric_10": float(private_metrics.get("metric_10", 0.0)),
            "metric_11": float(private_metrics.get("metric_11", 0.0)),
        },
        "recorded_at": time.time(),
    }


def _write_private_score(result: dict[str, Any]) -> None:
    path_text = os.environ.get("SUBSTRATE_PRIVATE_SCORE_PATH", "/tmp/substrate_private_scores.jsonl")
    if not path_text:
        return
    path = Path(path_text)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_scrub_private_result(result), sort_keys=True) + "\n")
    except OSError:
        # Private telemetry must never perturb visible grading.
        pass


def _public_feedback_lines(result: dict[str, Any]) -> list[str]:
    checks = result.get("checks") or {}
    if not checks:
        return [
            "PUBLIC_DIAGNOSTICS:",
            "  model_execution=needs_work",
            "  note=no scores, thresholds, hidden metrics, or model-specific field names are reported",
        ]

    def status(*names: str) -> str:
        return "ok" if all(bool(checks.get(name, False)) for name in names) else "needs_work"

    return [
        "PUBLIC_DIAGNOSTICS:",
        f"  response_observable={status('criterion_01', 'criterion_02')}",
        f"  spatial_scalar_trend={status('criterion_03', 'criterion_04', 'criterion_05', 'criterion_06', 'criterion_07', 'criterion_08', 'criterion_09')}",
        f"  continuum_field_structure={status('criterion_10', 'criterion_12')}",
        f"  local_diagnostic_consistency={status('criterion_11')}",
        f"  temporal_modulation={status('criterion_13')}",
        "  note=no scores, thresholds, hidden metrics, or model-specific field names are reported",
    ]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    candidate = Path(os.environ.get("SUBSTRATE_CANDIDATE", args.candidate)).resolve()
    with _suppress_untrusted_output():
        module = _load_candidate(candidate)
    _require_api(module)
    cfg = _make_config(module, args)

    master = np.random.default_rng(int(getattr(cfg, "seed", args.seed or 20260601)))
    seeds = master.integers(0, 2**32 - 1, size=int(args.trials), dtype=np.uint64)

    zero = _run_metrics(module, cfg, args.condition_a, seeds, args.c_region)
    negative = _run_metrics(module, cfg, args.condition_b, seeds, args.c_region)

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
        negative["cbar_by_region"]["zone_a"] - zero["cbar_by_region"]["zone_a"]
    )
    center_rise_rel = center_rise / max(abs(zero["cbar_by_region"]["zone_a"]), 1.0e-12)
    edge_drop = (
        zero["cbar_by_region"]["zone_b"] - negative["cbar_by_region"]["zone_b"]
    )
    edge_drop_rel = edge_drop / max(abs(zero["cbar_by_region"]["zone_b"]), 1.0e-12)
    center_increases = center_rise_rel >= args.min_center_rise_rel
    edge_decreases = edge_drop_rel >= args.min_edge_drop_rel

    center_interface_rise = (
        negative["cbar_by_region"]["zone_a_interface"]
        - zero["cbar_by_region"]["zone_a_interface"]
    )
    center_interface_rise_rel = center_interface_rise / max(
        abs(zero["cbar_by_region"]["zone_a_interface"]), 1.0e-12
    )
    center_interface_increases = (
        center_interface_rise_rel >= args.min_center_interface_rise_rel
    )

    edge_interface_drop = (
        zero["cbar_by_region"]["zone_b_interface"]
        - negative["cbar_by_region"]["zone_b_interface"]
    )
    edge_interface_drop_rel = edge_interface_drop / max(
        abs(zero["cbar_by_region"]["zone_b_interface"]), 1.0e-12
    )
    edge_interface_decreases = (
        edge_interface_drop_rel >= args.min_edge_interface_drop_rel
    )

    metric_07_base_a = (
        zero["cbar_by_region"]["zone_a"] - zero["cbar_by_region"]["zone_b"]
    )
    metric_07_base_b = (
        negative["cbar_by_region"]["zone_a"] - negative["cbar_by_region"]["zone_b"]
    )
    metric_07_abs = (
        metric_07_base_b - metric_07_base_a
    )
    metric_07_abs_rel = metric_07_abs / max(
        abs(metric_07_base_a), 1.0e-12
    )
    center_edge_contrast_increases = (
        metric_07_abs_rel >= args.min_metric_07_abs_rel
    )

    center_edge_sign_split = bool(center_increases and edge_decreases)
    center_feedback_sign_split = bool(center_increases and feedback_decreases)
    field_diag_zero = zero["field_diagnostics"]
    field_diag_negative = negative["field_diagnostics"]
    field_std_ok = (
        field_diag_zero["field_std_rel"] >= args.min_field_std_rel
        and field_diag_negative["field_std_rel"] >= args.min_field_std_rel
        and field_diag_zero["roughness_ratio"] <= args.max_field_roughness_ratio
        and field_diag_negative["roughness_ratio"] <= args.max_field_roughness_ratio
    )
    local_diagnostic_ok = (
        field_diag_zero["local_surface_separation_rel"]
        >= args.min_local_surface_separation_rel
        and field_diag_negative["local_surface_separation_rel"]
        >= args.min_local_surface_separation_rel
    )
    interface_structure_ok = (
        field_diag_zero["interface_deep_contrast_rel"]
        >= args.min_interface_depth_contrast_rel
        and field_diag_negative["interface_deep_contrast_rel"]
        >= args.min_interface_depth_contrast_rel
    )
    modulation_structure_ok = (
        field_diag_negative["harmonic_rel"] >= args.min_harmonic_rel
        and field_diag_negative["temporal_activity_rel"]
        >= args.min_temporal_activity_rel
    )

    if args.trend_mode == "local-shift":
        scalar_check = feedback_decreases
    elif args.trend_mode == "global-shift":
        scalar_check = c_increases
    elif args.trend_mode == "zone-a-zone-b":
        scalar_check = bool(center_increases and edge_decreases)
    elif args.trend_mode == "both":
        scalar_check = bool(c_increases and center_increases and edge_decreases)
    else:
        raise ValueError(f"unknown trend mode: {args.trend_mode}")

    score_components = {
        "criterion_01": {
            "description": "internal criterion",
            "weight": float(args.weight_q_drop),
            "passed": bool(q_decreases),
            "observed_relative_change": float(q_drop_rel),
            "required_minimum": float(args.min_q_drop_rel),
        },
        "criterion_02": {
            "description": "internal criterion",
            "weight": float(args.weight_feedback_drop),
            "passed": bool(feedback_decreases),
            "observed_relative_change": float(feedback_drop_rel),
            "required_minimum": float(args.min_feedback_drop_rel),
        },
        "criterion_03": {
            "description": "internal criterion",
            "weight": float(args.weight_center_rise),
            "passed": bool(center_increases),
            "observed_relative_change": float(center_rise_rel),
            "required_minimum": float(args.min_center_rise_rel),
        },
        "criterion_04": {
            "description": "internal criterion",
            "weight": float(args.weight_edge_drop),
            "passed": bool(edge_decreases),
            "observed_relative_change": float(edge_drop_rel),
            "required_minimum": float(args.min_edge_drop_rel),
        },
        "criterion_05": {
            "description": "internal criterion",
            "weight": float(args.weight_center_interface_rise),
            "passed": bool(center_interface_increases),
            "observed_relative_change": float(center_interface_rise_rel),
            "required_minimum": float(args.min_center_interface_rise_rel),
        },
        "criterion_06": {
            "description": "internal criterion",
            "weight": float(args.weight_edge_interface_drop),
            "passed": bool(edge_interface_decreases),
            "observed_relative_change": float(edge_interface_drop_rel),
            "required_minimum": float(args.min_edge_interface_drop_rel),
        },
        "criterion_07": {
            "description": "internal criterion",
            "weight": float(args.weight_center_edge_contrast),
            "passed": bool(center_edge_contrast_increases),
            "observed_relative_change": float(metric_07_abs_rel),
            "required_minimum": float(args.min_metric_07_abs_rel),
        },
        "criterion_08": {
            "description": "internal criterion",
            "weight": float(args.weight_center_edge_sign_split),
            "passed": bool(center_edge_sign_split),
            "observed_relative_change": float(min(center_rise_rel, edge_drop_rel)),
            "required_minimum": float(
                min(args.min_center_rise_rel, args.min_edge_drop_rel)
            ),
        },
        "criterion_09": {
            "description": "internal criterion",
            "weight": float(args.weight_center_feedback_sign_split),
            "passed": bool(center_feedback_sign_split),
            "observed_relative_change": float(min(center_rise_rel, feedback_drop_rel)),
            "required_minimum": float(
                min(args.min_center_rise_rel, args.min_feedback_drop_rel)
            ),
        },
        "criterion_10": {
            "description": "internal criterion",
            "weight": float(args.weight_field_structure),
            "passed": bool(field_std_ok),
            "observed_relative_change": float(
                min(field_diag_zero["field_std_rel"], field_diag_negative["field_std_rel"])
            ),
            "required_minimum": float(args.min_field_std_rel),
        },
        "criterion_11": {
            "description": "internal criterion",
            "weight": float(args.weight_local_diagnostic),
            "passed": bool(local_diagnostic_ok),
            "observed_relative_change": float(
                min(
                    field_diag_zero["local_surface_separation_rel"],
                    field_diag_negative["local_surface_separation_rel"],
                )
            ),
            "required_minimum": float(args.min_local_surface_separation_rel),
        },
        "criterion_12": {
            "description": "internal criterion",
            "weight": float(args.weight_interface_structure),
            "passed": bool(interface_structure_ok),
            "observed_relative_change": float(
                min(
                    field_diag_zero["interface_deep_contrast_rel"],
                    field_diag_negative["interface_deep_contrast_rel"],
                )
            ),
            "required_minimum": float(args.min_interface_depth_contrast_rel),
        },
        "criterion_13": {
            "description": "internal criterion",
            "weight": float(args.weight_modulation_structure),
            "passed": bool(modulation_structure_ok),
            "observed_relative_change": float(field_diag_negative["harmonic_rel"]),
            "required_minimum": float(args.min_harmonic_rel),
        },
    }
    earned_weight, score, passed = _weighted_score(score_components)

    return {
        "candidate": str(candidate),
        "config": {
            "condition_a": args.condition_a,
            "condition_b": args.condition_b,
            "trials": args.trials,
            "duration": float(getattr(cfg, "t_total", args.duration)),
            "nx": int(getattr(cfg, "nx", args.nx)),
            "nz": int(getattr(cfg, "nz", args.nz)),
            "diagnostic_region": args.c_region,
            "min_q_drop_rel": args.min_q_drop_rel,
            "min_c_rise_rel": args.min_c_rise_rel,
            "min_feedback_drop_rel": args.min_feedback_drop_rel,
            "min_criterion_03": args.min_center_rise_rel,
            "min_criterion_04": args.min_edge_drop_rel,
            "min_criterion_05": args.min_center_interface_rise_rel,
            "min_criterion_06": args.min_edge_interface_drop_rel,
            "min_metric_07_abs_rel": args.min_metric_07_abs_rel,
            "trend_mode": args.trend_mode,
            "score_weights": {
                "criterion_01": float(args.weight_q_drop),
                "criterion_02": float(args.weight_feedback_drop),
                "criterion_03": float(args.weight_center_rise),
                "criterion_04": float(args.weight_edge_drop),
                "criterion_05": float(args.weight_center_interface_rise),
                "criterion_06": float(args.weight_edge_interface_drop),
                "criterion_07": float(args.weight_center_edge_contrast),
                "criterion_08": float(args.weight_center_edge_sign_split),
                "criterion_09": float(args.weight_center_feedback_sign_split),
                "criterion_10": float(args.weight_field_structure),
                "criterion_11": float(args.weight_local_diagnostic),
                "criterion_12": float(args.weight_interface_structure),
                "criterion_13": float(args.weight_modulation_structure),
            },
        },
        "metrics": {
            "condition_a": zero,
            "condition_b": negative,
            "metric_01_abs": float(q_drop),
            "metric_01_abs_rel": float(q_drop_rel),
            "metric_aux_abs": float(c_rise),
            "metric_aux_abs_rel": float(c_rise_rel),
            "metric_02_abs": float(feedback_drop),
            "metric_02_abs_rel": float(feedback_drop_rel),
            "metric_03_abs": float(center_rise),
            "metric_03": float(center_rise_rel),
            "metric_04_abs": float(edge_drop),
            "metric_04_abs_rel": float(edge_drop_rel),
            "metric_05_abs": float(center_interface_rise),
            "metric_05": float(center_interface_rise_rel),
            "metric_06_abs": float(edge_interface_drop),
            "metric_06_abs_rel": float(edge_interface_drop_rel),
            "metric_07_base_a": float(metric_07_base_a),
            "metric_07_base_b": float(metric_07_base_b),
            "metric_07_abs": float(metric_07_abs),
            "metric_07_abs_rel": float(metric_07_abs_rel),
            "metric_08": float(
                min(field_diag_zero["field_std_rel"], field_diag_negative["field_std_rel"])
            ),
            "metric_09": float(
                min(
                    field_diag_zero["local_surface_separation_rel"],
                    field_diag_negative["local_surface_separation_rel"],
                )
            ),
            "metric_10": float(
                min(
                    field_diag_zero["interface_deep_contrast_rel"],
                    field_diag_negative["interface_deep_contrast_rel"],
                )
            ),
            "metric_11": float(field_diag_negative["harmonic_rel"]),
            "field_structure": {
                "condition_a": field_diag_zero,
                "condition_b": field_diag_negative,
            },
        },
        "checks": {
            "criterion_01": bool(q_decreases),
            "criterion_02": bool(feedback_decreases),
            "criterion_aux_01": bool(c_increases),
            "criterion_03": bool(center_increases),
            "criterion_04": bool(edge_decreases),
            "criterion_05": bool(center_interface_increases),
            "criterion_06": bool(edge_interface_decreases),
            "criterion_07": bool(center_edge_contrast_increases),
            "criterion_08": bool(center_edge_sign_split),
            "criterion_09": bool(center_feedback_sign_split),
            "criterion_10": bool(field_std_ok),
            "criterion_11": bool(local_diagnostic_ok),
            "criterion_12": bool(interface_structure_ok),
            "criterion_13": bool(modulation_structure_ok),
            "criterion_selected": bool(scalar_check),
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
    parser.add_argument("--alpha-zero", dest="condition_a", type=float, default=0.0)
    parser.add_argument("--alpha-negative", dest="condition_b", type=float, default=-0.001)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--history-window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--c-region",
        choices=("full", "feedback", "interface", "surface", "zone_a", "zone_b", "zone_a_interface", "zone_b_interface"),
        default="feedback",
    )
    parser.add_argument(
        "--trend-mode",
        choices=("local-shift", "global-shift", "zone-a-zone-b", "both"),
        default="local-shift",
        help="Diagnostic scalar trend to report; default scoring uses weighted components.",
    )
    parser.add_argument("--min-q-drop-rel", type=float, default=0.01)
    parser.add_argument("--min-c-rise-rel", type=float, default=0.001)
    parser.add_argument("--min-feedback-drop-rel", type=float, default=0.001)
    parser.add_argument("--min-center-rise-rel", type=float, default=0.001)
    parser.add_argument("--min-edge-drop-rel", type=float, default=0.001)
    parser.add_argument("--min-center-interface-rise-rel", type=float, default=0.001)
    parser.add_argument("--min-edge-interface-drop-rel", type=float, default=0.001)
    parser.add_argument("--min-center-edge-contrast-gain-rel", dest="min_metric_07_abs_rel", type=float, default=0.01)
    parser.add_argument("--min-field-std-rel", type=float, default=0.02)
    parser.add_argument("--max-field-roughness-ratio", type=float, default=0.85)
    parser.add_argument("--min-local-surface-separation-rel", type=float, default=0.05)
    parser.add_argument("--min-interface-depth-contrast-rel", type=float, default=0.10)
    parser.add_argument("--min-harmonic-rel", type=float, default=1.0e-4)
    parser.add_argument("--min-temporal-activity-rel", type=float, default=1.0e-4)
    parser.add_argument("--weight-q-drop", type=float, default=DEFAULT_WEIGHTS["criterion_01"])
    parser.add_argument("--weight-feedback-drop", type=float, default=DEFAULT_WEIGHTS["criterion_02"])
    parser.add_argument("--weight-center-rise", type=float, default=DEFAULT_WEIGHTS["criterion_03"])
    parser.add_argument("--weight-edge-drop", type=float, default=DEFAULT_WEIGHTS["criterion_04"])
    parser.add_argument("--weight-center-interface-rise", type=float, default=DEFAULT_WEIGHTS["criterion_05"])
    parser.add_argument("--weight-edge-interface-drop", type=float, default=DEFAULT_WEIGHTS["criterion_06"])
    parser.add_argument("--weight-center-edge-contrast", type=float, default=DEFAULT_WEIGHTS["criterion_07"])
    parser.add_argument("--weight-center-edge-sign-split", type=float, default=DEFAULT_WEIGHTS["criterion_08"])
    parser.add_argument("--weight-center-feedback-sign-split", type=float, default=DEFAULT_WEIGHTS["criterion_09"])
    parser.add_argument("--weight-field-structure", type=float, default=DEFAULT_WEIGHTS["criterion_10"])
    parser.add_argument("--weight-local-diagnostic", type=float, default=DEFAULT_WEIGHTS["criterion_11"])
    parser.add_argument("--weight-interface-structure", type=float, default=DEFAULT_WEIGHTS["criterion_12"])
    parser.add_argument("--weight-modulation-structure", type=float, default=DEFAULT_WEIGHTS["criterion_13"])
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

    _write_private_score(result)
    passed = bool(result.get("passed"))
    verdict = "PASS" if passed else "FAIL"
    case_status = "OK" if passed else "WA"
    visible_score = 1 if passed else 0
    cases_ok = 1 if passed else 0
    print("=" * 40)
    print(f"RESULT: {verdict}")
    for line in _public_feedback_lines(result):
        print(line)
    print(f"CASE visible {case_status} score={visible_score}")
    print(f"TOTAL_SCORE {visible_score}")
    print(f"CASES_OK {cases_ok}")
    print("CASES_TOTAL 1")
    print("=" * 40)
    return 0 if passed else 1


if __name__ == "__main__":
    exit_code = main()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(exit_code)
