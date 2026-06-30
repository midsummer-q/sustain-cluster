import dataclasses

import numpy as np
import pandas as pd

from utils.low_carbon_window import compute_window_overlap_ratio


DEFAULT_MIN_ACTUAL_LCW_OVERLAP_RATIO = 0.5


def build_lcwra_audit_record(
    task,
    selected_plan,
    all_candidate_plans,
    actual_start_time=None,
    actual_finish_time=None,
    audit_stage="completed",
    min_actual_lcw_overlap_ratio=DEFAULT_MIN_ACTUAL_LCW_OVERLAP_RATIO,
    dest_dc=None,
    config=None,
    unfinished_location=None,
    episode_end_time=None,
) -> dict:
    """Build a flat audit record for LCWRA selected/completed plan analysis."""
    plans = list(all_candidate_plans or [])
    selected = selected_plan
    reachable_count = sum(1 for plan in plans if getattr(plan, "reachable_lcw", False))
    deadline_count = sum(1 for plan in plans if getattr(plan, "deadline_feasible", False))
    sla_safe_count = sum(1 for plan in plans if getattr(plan, "sla_safe", False))
    rejected_by_guard_count = sum(1 for plan in plans if getattr(plan, "rejected_by_sla_guard", False))

    actual_start_time = actual_start_time if actual_start_time is not None else getattr(task, "start_time", None)
    actual_finish_time = actual_finish_time if actual_finish_time is not None else getattr(task, "finish_time", None)

    planned_start = getattr(selected, "planned_start_time", None)
    planned_finish = getattr(selected, "planned_finish_time", None)
    planned_overlap = getattr(selected, "low_carbon_overlap_ratio", 0.0)
    selected_window_overlap = _actual_overlap_with_selected_window_ratio(selected, actual_start_time, actual_finish_time)
    all_windows_overlap = _actual_overlap_ratio(selected, actual_start_time, actual_finish_time, dest_dc, config)
    actual_overlap = all_windows_overlap if all_windows_overlap is not None else selected_window_overlap
    overlap_error = None if actual_overlap is None else actual_overlap - planned_overlap
    actual_missed = None if actual_overlap is None else actual_overlap < min_actual_lcw_overlap_ratio
    actual_hit = None if actual_overlap is None else not actual_missed
    episode_end_time = pd.Timestamp(episode_end_time) if episode_end_time is not None else None
    sla_deadline = getattr(task, "sla_deadline", None)

    return {
        "audit_stage": audit_stage,
        "task_id": getattr(task, "job_name", None),
        "origin_dc_id": getattr(task, "origin_dc_id", None),
        "dest_dc_id": getattr(selected, "dest_dc_id", None),
        "decision_time": getattr(selected, "decision_time", None),
        "arrival_time": getattr(selected, "arrival_time", None),
        "planned_start_time": planned_start,
        "planned_finish_time": planned_finish,
        "actual_start_time": actual_start_time,
        "actual_finish_time": actual_finish_time,
        "sla_deadline": sla_deadline,
        "low_carbon_window_start": getattr(selected, "low_carbon_window_start", None),
        "low_carbon_window_end": getattr(selected, "low_carbon_window_end", None),
        "low_carbon_overlap_ratio": planned_overlap,
        "planned_low_carbon_overlap_ratio": planned_overlap,
        "actual_overlap_with_selected_window_ratio": selected_window_overlap,
        "actual_low_carbon_overlap_ratio": actual_overlap,
        "actual_low_carbon_overlap_source": (
            "all_low_carbon_windows"
            if all_windows_overlap is not None
            else "selected_window_fallback"
            if selected_window_overlap is not None
            else None
        ),
        "low_carbon_overlap_error": overlap_error,
        "reachable_lcw": getattr(selected, "reachable_lcw", False),
        "ci_source_mode": getattr(selected, "ci_source_mode", None),
        "deadline_slack_min": getattr(selected, "deadline_slack_min", None),
        "sla_safe": getattr(selected, "sla_safe", False),
        "sla_risk_score": getattr(selected, "sla_risk_score", None),
        "queue_wait_safety_factor": getattr(selected, "queue_wait_safety_factor", None),
        "max_estimated_queue_wait_min": getattr(selected, "max_estimated_queue_wait_min", None),
        "queue_wait_over_guard_min": getattr(selected, "queue_wait_over_guard_min", None),
        "selected_guard_stage": getattr(selected, "selected_guard_stage", None),
        "rejected_by_sla_guard": getattr(selected, "rejected_by_sla_guard", False),
        "unfinished_location": unfinished_location,
        "sla_deadline_expired_at_episode_end": (
            bool(episode_end_time > pd.Timestamp(sla_deadline))
            if episode_end_time is not None and sla_deadline is not None
            else None
        ),
        "low_carbon_missed": actual_missed,
        "actual_low_carbon_missed": actual_missed,
        "actual_low_carbon_hit": actual_hit,
        "predicted_task_proxy_dc_carbon_kg": getattr(selected, "predicted_task_proxy_dc_carbon_kg", None),
        "predicted_task_proxy_system_carbon_kg": getattr(selected, "predicted_task_proxy_system_carbon_kg", None),
        "predicted_marginal_dc_carbon_kg": getattr(selected, "predicted_marginal_dc_carbon_kg", None),
        "predicted_marginal_system_carbon_kg": getattr(selected, "predicted_marginal_system_carbon_kg", None),
        "predicted_system_carbon_kg": getattr(selected, "predicted_system_carbon_kg", None),
        "candidate_count": len(plans),
        "reachable_candidate_count": reachable_count,
        "deadline_feasible_candidate_count": deadline_count,
        "sla_safe_candidate_count": sla_safe_count,
        "rejected_by_sla_guard_candidate_count": rejected_by_guard_count,
        "any_candidate_rejected_by_sla_guard": rejected_by_guard_count > 0,
        "selected_reason": getattr(selected, "reason", None),
        "plan_start_error_min": _error_minutes(planned_start, actual_start_time),
        "plan_finish_error_min": _error_minutes(planned_finish, actual_finish_time),
        "actual_start_delay_min": _error_minutes(planned_start, actual_start_time),
        "actual_finish_delay_min": _error_minutes(planned_finish, actual_finish_time),
        "all_candidate_plans": [dataclasses.asdict(plan) for plan in plans],
    }


def _actual_overlap_ratio(selected_plan, actual_start_time, actual_finish_time, dest_dc=None, config=None):
    windows = _reconstruct_low_carbon_windows(selected_plan, dest_dc, config)
    if not windows:
        return None
    if actual_start_time is None or actual_finish_time is None:
        return None

    return compute_window_overlap_ratio(actual_start_time, actual_finish_time, windows)


def _actual_overlap_with_selected_window_ratio(selected_plan, actual_start_time, actual_finish_time):
    if selected_plan is None or actual_start_time is None or actual_finish_time is None:
        return None

    window_start = getattr(selected_plan, "low_carbon_window_start", None)
    window_end = getattr(selected_plan, "low_carbon_window_end", None)
    if window_start is None or window_end is None:
        return None

    return compute_window_overlap_ratio(actual_start_time, actual_finish_time, [(window_start, window_end, None)])


def _reconstruct_low_carbon_windows(selected_plan, dest_dc, config):
    if selected_plan is None or dest_dc is None:
        return []

    ci_manager = getattr(dest_dc, "ci_manager", None)
    series = getattr(ci_manager, "carbon_smooth", None)
    if ci_manager is None or series is None:
        return []

    cfg = config or {}
    horizon_steps = int(cfg.get("horizon_steps", 32))
    timestep_minutes = int(cfg.get("timestep_minutes", 15))
    low_carbon_quantile = float(cfg.get("low_carbon_quantile", 0.25))
    min_window_steps = max(1, int(cfg.get("min_window_steps", 1)))
    start_idx = getattr(selected_plan, "decision_time_step", None)
    decision_time = getattr(selected_plan, "decision_time", None)
    if start_idx is None or decision_time is None:
        return []

    start_idx = max(0, int(start_idx))
    end_idx = min(len(series), start_idx + horizon_steps)
    if start_idx >= end_idx:
        return []

    values = [float(value) for value in series[start_idx:end_idx]]
    if not values:
        return []

    percentile = low_carbon_quantile * 100.0 if low_carbon_quantile <= 1.0 else low_carbon_quantile
    threshold = float(np.percentile(values, percentile))
    windows = []
    window_start_idx = None
    for idx, ci in enumerate(values):
        if ci <= threshold:
            if window_start_idx is None:
                window_start_idx = idx
        elif window_start_idx is not None:
            _append_reconstructed_window(windows, decision_time, timestep_minutes, window_start_idx, idx, threshold, min_window_steps)
            window_start_idx = None

    if window_start_idx is not None:
        _append_reconstructed_window(
            windows,
            decision_time,
            timestep_minutes,
            window_start_idx,
            len(values),
            threshold,
            min_window_steps,
        )

    return windows


def _append_reconstructed_window(windows, decision_time, timestep_minutes, start_idx, end_idx, threshold, min_window_steps):
    if end_idx - start_idx < min_window_steps:
        return
    start = pd.Timestamp(decision_time) + pd.Timedelta(minutes=start_idx * timestep_minutes)
    end = pd.Timestamp(decision_time) + pd.Timedelta(minutes=end_idx * timestep_minutes)
    windows.append((start, end, threshold))


def _error_minutes(planned_time, actual_time):
    if planned_time is None or actual_time is None:
        return None
    return (actual_time - planned_time).total_seconds() / 60.0
