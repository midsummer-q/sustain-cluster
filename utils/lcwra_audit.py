import dataclasses

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
) -> dict:
    """Build a flat audit record for LCWRA selected/completed plan analysis."""
    plans = list(all_candidate_plans or [])
    selected = selected_plan
    reachable_count = sum(1 for plan in plans if getattr(plan, "reachable_lcw", False))
    deadline_count = sum(1 for plan in plans if getattr(plan, "deadline_feasible", False))

    actual_start_time = actual_start_time if actual_start_time is not None else getattr(task, "start_time", None)
    actual_finish_time = actual_finish_time if actual_finish_time is not None else getattr(task, "finish_time", None)

    planned_start = getattr(selected, "planned_start_time", None)
    planned_finish = getattr(selected, "planned_finish_time", None)
    planned_overlap = getattr(selected, "low_carbon_overlap_ratio", 0.0)
    actual_overlap = _actual_overlap_ratio(selected, actual_start_time, actual_finish_time)
    overlap_error = None if actual_overlap is None else actual_overlap - planned_overlap
    actual_missed = None if actual_overlap is None else actual_overlap < min_actual_lcw_overlap_ratio
    actual_hit = None if actual_overlap is None else not actual_missed

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
        "sla_deadline": getattr(task, "sla_deadline", None),
        "low_carbon_window_start": getattr(selected, "low_carbon_window_start", None),
        "low_carbon_window_end": getattr(selected, "low_carbon_window_end", None),
        "low_carbon_overlap_ratio": planned_overlap,
        "planned_low_carbon_overlap_ratio": planned_overlap,
        "actual_low_carbon_overlap_ratio": actual_overlap,
        "low_carbon_overlap_error": overlap_error,
        "reachable_lcw": getattr(selected, "reachable_lcw", False),
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
        "selected_reason": getattr(selected, "reason", None),
        "plan_start_error_min": _error_minutes(planned_start, actual_start_time),
        "plan_finish_error_min": _error_minutes(planned_finish, actual_finish_time),
        "actual_start_delay_min": _error_minutes(planned_start, actual_start_time),
        "actual_finish_delay_min": _error_minutes(planned_finish, actual_finish_time),
        "all_candidate_plans": [dataclasses.asdict(plan) for plan in plans],
    }


def _actual_overlap_ratio(selected_plan, actual_start_time, actual_finish_time):
    if selected_plan is None or actual_start_time is None or actual_finish_time is None:
        return None

    window_start = getattr(selected_plan, "low_carbon_window_start", None)
    window_end = getattr(selected_plan, "low_carbon_window_end", None)
    if window_start is None or window_end is None:
        return None

    return compute_window_overlap_ratio(actual_start_time, actual_finish_time, [(window_start, window_end, None)])


def _error_minutes(planned_time, actual_time):
    if planned_time is None or actual_time is None:
        return None
    return (actual_time - planned_time).total_seconds() / 60.0
