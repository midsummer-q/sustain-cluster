import math

import pandas as pd

from data.network_cost.network_delay import get_transmission_delay
from utils.lcwra_types import CandidatePlan
from utils.low_carbon_window import compute_window_overlap_ratio, find_low_carbon_windows


DEFAULT_LCWRA_CONFIG = {
    "horizon_steps": 32,
    "timestep_minutes": 15,
    "low_carbon_quantile": 0.25,
    "min_window_steps": 1,
    "min_lcw_overlap_ratio": 0.5,
    "min_actual_lcw_overlap_ratio": 0.5,
    "queue_wait_mode": "resource_simulation",
    "deadline_feasibility": True,
    "objective": "predicted_marginal_system_carbon",
    "fallback_when_no_reachable_lcw": "lowest_predicted_marginal_system_carbon_feasible",
}

UNREACHABLE_REASONS = {
    "resource_infeasible",
    "resource_wait_unbounded",
    "queue_unbounded_no_release_event",
}


def estimate_candidate_plan(task, dest_dc, dest_dc_name, current_time, cluster_manager, config) -> CandidatePlan:
    """Estimate whether a task can reach a low-carbon execution window at dest_dc."""
    cfg = {**DEFAULT_LCWRA_CONFIG, **(config or {})}
    decision_time = pd.Timestamp(current_time)
    origin_dc_id = getattr(task, "origin_dc_id", None)
    dest_dc_id = getattr(dest_dc, "dc_id", None)
    decision_time_step = getattr(getattr(dest_dc, "ci_manager", None), "time_step", None)

    transmission_delay_min = _estimate_transmission_delay_min(task, dest_dc, cluster_manager)
    arrival_time = decision_time + pd.Timedelta(minutes=transmission_delay_min)

    resource_feasible = _task_fits_dc_capacity(task, dest_dc)
    planned_start_time, planned_finish_time, reason = _simulate_fifo_start_time(
        task=task,
        dest_dc=dest_dc,
        arrival_time=arrival_time,
        current_time=decision_time,
        resource_feasible=resource_feasible,
        cluster_manager=cluster_manager,
        dest_dc_name=dest_dc_name,
    )
    unreachable = reason in UNREACHABLE_REASONS

    if unreachable:
        estimated_queue_wait_min = float("inf")
        deadline_feasible = False
        overlap_ratio = 0.0
        selected_window_start, selected_window_end = None, None
        predicted_transmission_carbon_kg = _predict_transmission_carbon_kg(task, dest_dc, cluster_manager)
        predicted_task_proxy_dc_carbon_kg = float("inf")
        predicted_task_proxy_system_carbon_kg = float("inf")
        predicted_marginal_dc_carbon_kg = float("inf")
        predicted_marginal_system_carbon_kg = float("inf")
    else:
        estimated_queue_wait_min = max(0.0, (planned_start_time - arrival_time).total_seconds() / 60.0)
        deadline_feasible = planned_finish_time <= pd.Timestamp(task.sla_deadline)
        if not cfg.get("deadline_feasibility", True):
            deadline_feasible = True

        windows = find_low_carbon_windows(
            dest_dc,
            decision_time,
            timestep_minutes=int(cfg["timestep_minutes"]),
            horizon_steps=int(cfg["horizon_steps"]),
            low_carbon_quantile=float(cfg["low_carbon_quantile"]),
            min_window_steps=int(cfg.get("min_window_steps", 1)),
        )
        overlap_ratio = compute_window_overlap_ratio(planned_start_time, planned_finish_time, windows)
        selected_window_start, selected_window_end = _select_best_window(planned_start_time, planned_finish_time, windows)

        predicted_transmission_carbon_kg = _predict_transmission_carbon_kg(task, dest_dc, cluster_manager)
        predicted_task_proxy_dc_carbon_kg = _predict_task_proxy_dc_carbon_kg(
            task,
            dest_dc,
            planned_start_time,
            planned_finish_time,
            current_time=decision_time,
            timestep_minutes=int(cfg["timestep_minutes"]),
        )
        predicted_task_proxy_system_carbon_kg = predicted_task_proxy_dc_carbon_kg + predicted_transmission_carbon_kg
        predicted_marginal_dc_carbon_kg = _predict_marginal_facility_dc_carbon_kg(
            task,
            dest_dc,
            planned_start_time,
            planned_finish_time,
            current_time=decision_time,
            timestep_minutes=int(cfg["timestep_minutes"]),
        )
        predicted_marginal_system_carbon_kg = predicted_marginal_dc_carbon_kg + predicted_transmission_carbon_kg

    reachable_lcw = (
        resource_feasible
        and deadline_feasible
        and overlap_ratio >= float(cfg.get("min_lcw_overlap_ratio", 0.5))
        and not unreachable
    )

    if not resource_feasible:
        reason = "resource_infeasible"
    elif unreachable:
        reason = reason
    elif not deadline_feasible:
        reason = "deadline_infeasible"
    elif reachable_lcw:
        reason = "reachable_low_carbon_window"
    elif overlap_ratio > 0:
        reason = "partial_low_carbon_overlap"
    elif selected_window_start is None:
        reason = "no_low_carbon_window_in_horizon"

    return CandidatePlan(
        task_id=getattr(task, "job_name", ""),
        origin_dc_id=origin_dc_id,
        dest_dc_id=dest_dc_id,
        dest_dc_name=dest_dc_name,
        decision_time=decision_time,
        transmission_delay_min=transmission_delay_min,
        arrival_time=arrival_time,
        estimated_queue_wait_min=estimated_queue_wait_min,
        planned_start_time=planned_start_time,
        planned_finish_time=planned_finish_time,
        low_carbon_window_start=selected_window_start,
        low_carbon_window_end=selected_window_end,
        low_carbon_overlap_ratio=overlap_ratio,
        resource_feasible=resource_feasible,
        deadline_feasible=deadline_feasible,
        reachable_lcw=reachable_lcw,
        predicted_task_proxy_dc_carbon_kg=predicted_task_proxy_dc_carbon_kg,
        predicted_task_proxy_system_carbon_kg=predicted_task_proxy_system_carbon_kg,
        predicted_marginal_dc_carbon_kg=predicted_marginal_dc_carbon_kg,
        predicted_dc_carbon_kg=predicted_marginal_dc_carbon_kg,
        predicted_transmission_carbon_kg=predicted_transmission_carbon_kg,
        predicted_marginal_system_carbon_kg=predicted_marginal_system_carbon_kg,
        # Backward-compatible alias. This now points to marginal facility-aware
        # system carbon, not to the legacy task-level carbon proxy.
        predicted_system_carbon_kg=predicted_marginal_system_carbon_kg,
        reason=reason,
        ci_source_mode=cfg.get("ci_source_mode"),
        decision_time_step=decision_time_step,
    )


def _estimate_transmission_delay_min(task, dest_dc, cluster_manager) -> float:
    if getattr(task, "origin_dc_id", None) == getattr(dest_dc, "dc_id", None):
        return 0.0

    origin_loc = cluster_manager.get_dc_location(task.origin_dc_id)
    dest_loc = dest_dc.location
    delay_s = get_transmission_delay(origin_loc, dest_loc, cluster_manager.cloud_provider, task.bandwidth_gb)
    return max(0.0, float(delay_s) / 60.0)


def _task_fits_dc_capacity(task, dest_dc) -> bool:
    return (
        task.cores_req <= getattr(dest_dc, "total_cores", 0)
        and task.gpu_req <= getattr(dest_dc, "total_gpus", 0)
        and task.mem_req <= getattr(dest_dc, "total_mem_GB", 0)
    )


def _simulate_fifo_start_time(task, dest_dc, arrival_time, current_time, resource_feasible, cluster_manager, dest_dc_name):
    if not resource_feasible:
        return pd.Timestamp.max, pd.Timestamp.max, "resource_infeasible"

    available = {
        "cores": float(getattr(dest_dc, "available_cores", 0.0)),
        "gpu": float(getattr(dest_dc, "available_gpus", 0.0)),
        "mem": float(getattr(dest_dc, "available_mem", 0.0)),
    }
    release_events = _running_task_release_events(dest_dc)
    queue_items = _queue_items_with_entry_times(task, dest_dc, arrival_time, current_time, cluster_manager, dest_dc_name)
    event_idx = 0

    def apply_releases_until(timestamp):
        nonlocal event_idx
        while event_idx < len(release_events) and release_events[event_idx][0] <= timestamp:
            _, cores, gpu, mem = release_events[event_idx]
            available["cores"] += cores
            available["gpu"] += gpu
            available["mem"] += mem
            event_idx += 1

    for entry_time, _, queued_task, source in queue_items:
        entry_time = pd.Timestamp(entry_time)
        apply_releases_until(entry_time)

        if not _task_fits_dc_capacity(queued_task, dest_dc):
            if queued_task is task:
                return pd.Timestamp.max, pd.Timestamp.max, "resource_infeasible"
            continue

        candidate_time = entry_time
        while not _has_resources(available, queued_task):
            if event_idx >= len(release_events):
                if queued_task is task:
                    return pd.Timestamp.max, pd.Timestamp.max, "resource_wait_unbounded"
                return pd.Timestamp.max, pd.Timestamp.max, "queue_unbounded_no_release_event"
            candidate_time = release_events[event_idx][0]
            apply_releases_until(candidate_time)

        if queued_task is task:
            finish_time = candidate_time + pd.Timedelta(minutes=float(getattr(task, "duration", 0.0)))
            return candidate_time, finish_time, f"estimated_fifo_start_after_{source}"

        available["cores"] -= float(getattr(queued_task, "cores_req", 0.0))
        available["gpu"] -= float(getattr(queued_task, "gpu_req", 0.0))
        available["mem"] -= float(getattr(queued_task, "mem_req", 0.0))
        release_events.append((
            candidate_time + pd.Timedelta(minutes=float(getattr(queued_task, "duration", 0.0))),
            float(getattr(queued_task, "cores_req", 0.0)),
            float(getattr(queued_task, "gpu_req", 0.0)),
            float(getattr(queued_task, "mem_req", 0.0)),
        ))
        release_events.sort(key=lambda item: item[0])

    return pd.Timestamp.max, pd.Timestamp.max, "queue_unbounded_no_release_event"


def _running_task_release_events(dest_dc):
    release_events = []
    for running_task in list(getattr(dest_dc, "running_tasks", [])):
        finish_time = getattr(running_task, "finish_time", None)
        if finish_time is None:
            continue
        release_events.append((
            pd.Timestamp(finish_time),
            float(getattr(running_task, "cores_req", 0.0)),
            float(getattr(running_task, "gpu_req", 0.0)),
            float(getattr(running_task, "mem_req", 0.0)),
        ))
    release_events.sort(key=lambda item: item[0])
    return release_events


def _queue_items_with_entry_times(task, dest_dc, arrival_time, current_time, cluster_manager, dest_dc_name):
    queue_items = []
    order = 0
    for pending_task in list(getattr(dest_dc, "pending_tasks", [])):
        queue_items.append((pd.Timestamp(current_time), order, pending_task, "pending"))
        order += 1

    for transit_arrival_time, transit_task, transit_dc_name in list(getattr(cluster_manager, "in_transit_tasks", [])):
        if transit_dc_name == dest_dc_name:
            queue_items.append((pd.Timestamp(transit_arrival_time), order, transit_task, "in_transit"))
            order += 1

    queue_items.append((pd.Timestamp(arrival_time), order, task, "candidate"))
    queue_items.sort(key=lambda item: (item[0], item[1]))
    return queue_items


def _has_resources(available, task) -> bool:
    return (
        float(getattr(task, "cores_req", 0.0)) <= available["cores"]
        and float(getattr(task, "gpu_req", 0.0)) <= available["gpu"]
        and float(getattr(task, "mem_req", 0.0)) <= available["mem"]
    )


def _select_best_window(planned_start_time, planned_finish_time, windows):
    best_window = None
    best_overlap_seconds = 0.0
    for window in windows:
        overlap_start = max(pd.Timestamp(planned_start_time), pd.Timestamp(window[0]))
        overlap_end = min(pd.Timestamp(planned_finish_time), pd.Timestamp(window[1]))
        overlap_seconds = max(0.0, (overlap_end - overlap_start).total_seconds())
        if overlap_seconds > best_overlap_seconds:
            best_overlap_seconds = overlap_seconds
            best_window = window

    if best_window is None and windows:
        best_window = windows[0]

    if best_window is None:
        return None, None
    return best_window[0], best_window[1]


def _predict_transmission_carbon_kg(task, dest_dc, cluster_manager) -> float:
    if getattr(task, "origin_dc_id", None) == getattr(dest_dc, "dc_id", None):
        return 0.0

    origin_dc = next(
        (dc for dc in cluster_manager.datacenters.values() if dc.dc_id == getattr(task, "origin_dc_id", None)),
        None,
    )
    if origin_dc is None:
        return 0.0

    energy_kwh = float(getattr(task, "bandwidth_gb", 0.0)) * 0.06
    ci_origin_kg_per_kwh = float(origin_dc.ci_manager.get_current_ci(norm=False)) / 1000.0
    return energy_kwh * ci_origin_kg_per_kwh


def _predict_task_proxy_dc_carbon_kg(task, dest_dc, planned_start_time, planned_finish_time, current_time, timestep_minutes):
    """Legacy task-level proxy retained only for diagnostics; units are kgCO2."""
    task_energy_kwh = float(getattr(task, "cores_req", 0.0)) * float(getattr(task, "duration", 0.0)) / 10000.0
    avg_ci = _average_ci_for_interval(dest_dc, planned_start_time, planned_finish_time, current_time, timestep_minutes)
    return task_energy_kwh * avg_ci / 1000.0


def _predict_marginal_facility_dc_carbon_kg(task, dest_dc, planned_start_time, planned_finish_time, current_time, timestep_minutes):
    """
    Facility-aware marginal approximation, not a legacy task proxy.

    The project does not expose a direct marginal facility-carbon API, so this uses
    incremental IT energy from requested CPU/GPU/MEM resources, multiplies by an
    estimated current PUE, then applies average CI over the planned execution window.
    """
    duration_hours = float(getattr(task, "duration", 0.0)) / 60.0
    incremental_it_kw = (
        0.006 * float(getattr(task, "cores_req", 0.0))
        + 0.5 * float(getattr(task, "gpu_req", 0.0))
        + 0.0025 * float(getattr(task, "mem_req", 0.0))
    )
    marginal_facility_energy_kwh = incremental_it_kw * duration_hours * _estimate_current_pue(dest_dc)
    avg_ci = _average_ci_for_interval(dest_dc, planned_start_time, planned_finish_time, current_time, timestep_minutes)
    return marginal_facility_energy_kwh * avg_ci / 1000.0


def _estimate_current_pue(dest_dc):
    infos = getattr(dest_dc, "infos", {}) or {}
    common = infos.get("__common__", {}) if isinstance(infos, dict) else {}
    agent_dc = infos.get("agent_dc", {}) if isinstance(infos, dict) else {}

    energy_kwh = common.get("energy_consumption_kwh")
    ite_power_kw = agent_dc.get("dc_ITE_total_power_kW")
    if energy_kwh is not None and ite_power_kw:
        ite_energy_kwh = float(ite_power_kw) * 0.25
        if ite_energy_kwh > 0:
            return max(1.0, float(energy_kwh) / ite_energy_kwh)

    pue = common.get("pue") or agent_dc.get("dc_PUE")
    if pue:
        return max(1.0, float(pue))

    return 1.5


def _average_ci_for_interval(dest_dc, planned_start_time, planned_finish_time, current_time, timestep_minutes):
    ci_manager = getattr(dest_dc, "ci_manager", None)
    series = getattr(ci_manager, "carbon_smooth", None)
    time_step = getattr(ci_manager, "time_step", None)
    if ci_manager is None or series is None or time_step is None:
        return 0.0

    start_offset = max(0.0, (pd.Timestamp(planned_start_time) - pd.Timestamp(current_time)).total_seconds() / 60.0)
    finish_offset = max(start_offset + 1e-9, (pd.Timestamp(planned_finish_time) - pd.Timestamp(current_time)).total_seconds() / 60.0)
    start_idx = int(time_step) + int(math.floor(start_offset / timestep_minutes))
    end_idx = int(time_step) + int(math.ceil(finish_offset / timestep_minutes))
    start_idx = max(0, min(len(series), start_idx))
    end_idx = max(start_idx + 1, min(len(series), end_idx))
    values = series[start_idx:end_idx]
    if len(values) == 0:
        return float(ci_manager.get_current_ci(norm=False))
    return float(sum(values) / len(values))
