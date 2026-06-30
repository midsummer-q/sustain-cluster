import pandas as pd

from utils.lcwra_types import CandidatePlan
from utils.task_assignment_strategies import DistributeOracleSLAGuardedLCWRA


def _plan(
    dest_dc_id,
    *,
    reachable_lcw=False,
    resource_feasible=True,
    deadline_feasible=True,
    deadline_slack_min=30.0,
    estimated_queue_wait_min=0.0,
    predicted_marginal_system_carbon_kg=1.0,
):
    decision_time = pd.Timestamp("2026-01-01T00:00:00Z")
    planned_finish_time = decision_time + pd.Timedelta(minutes=30)
    return CandidatePlan(
        task_id="task",
        origin_dc_id=1,
        dest_dc_id=dest_dc_id,
        dest_dc_name=f"DC{dest_dc_id}",
        decision_time=decision_time,
        transmission_delay_min=0.0,
        arrival_time=decision_time,
        estimated_queue_wait_min=estimated_queue_wait_min,
        planned_start_time=decision_time,
        planned_finish_time=planned_finish_time,
        low_carbon_window_start=decision_time,
        low_carbon_window_end=planned_finish_time,
        low_carbon_overlap_ratio=1.0 if reachable_lcw else 0.0,
        resource_feasible=resource_feasible,
        deadline_feasible=deadline_feasible,
        reachable_lcw=reachable_lcw,
        predicted_task_proxy_dc_carbon_kg=predicted_marginal_system_carbon_kg,
        predicted_task_proxy_system_carbon_kg=predicted_marginal_system_carbon_kg,
        predicted_marginal_dc_carbon_kg=predicted_marginal_system_carbon_kg,
        predicted_dc_carbon_kg=predicted_marginal_system_carbon_kg,
        predicted_transmission_carbon_kg=0.0,
        predicted_marginal_system_carbon_kg=predicted_marginal_system_carbon_kg,
        predicted_system_carbon_kg=predicted_marginal_system_carbon_kg,
        reason="test",
        ci_source_mode="oracle",
        deadline_slack_min=deadline_slack_min,
        sla_safe=deadline_slack_min >= 15.0,
        sla_risk_score=max(0.0, 15.0 - deadline_slack_min),
        queue_wait_safety_factor=1.2,
        selected_guard_stage=None,
        rejected_by_sla_guard=reachable_lcw and deadline_slack_min < 15.0,
    )


def test_selects_lowest_carbon_reachable_lcw_candidate_that_is_sla_safe():
    strategy = DistributeOracleSLAGuardedLCWRA(config={"min_deadline_slack_min": 15})
    unsafe_low_carbon = _plan(1, reachable_lcw=True, deadline_slack_min=5.0, predicted_marginal_system_carbon_kg=0.1)
    safe_higher_carbon = _plan(2, reachable_lcw=True, deadline_slack_min=30.0, predicted_marginal_system_carbon_kg=0.5)
    safe_highest_carbon = _plan(3, reachable_lcw=True, deadline_slack_min=45.0, predicted_marginal_system_carbon_kg=2.0)

    selected = strategy._select_plan([unsafe_low_carbon, safe_higher_carbon, safe_highest_carbon])

    assert selected.dest_dc_id == 2
    assert selected.selected_guard_stage == "reachable_lcw_and_sla_safe"
    assert unsafe_low_carbon.rejected_by_sla_guard is True


def test_falls_back_to_lowest_sla_risk_then_carbon_when_no_reachable_safe_candidate():
    strategy = DistributeOracleSLAGuardedLCWRA(config={"min_deadline_slack_min": 15})
    higher_risk = _plan(1, reachable_lcw=True, deadline_slack_min=0.0, predicted_marginal_system_carbon_kg=0.1)
    lower_risk_higher_carbon = _plan(2, reachable_lcw=False, deadline_slack_min=10.0, predicted_marginal_system_carbon_kg=1.0)
    lower_risk_lower_carbon = _plan(3, reachable_lcw=False, deadline_slack_min=10.0, predicted_marginal_system_carbon_kg=0.5)

    selected = strategy._select_plan([higher_risk, lower_risk_higher_carbon, lower_risk_lower_carbon])

    assert selected.dest_dc_id == 3
    assert selected.selected_guard_stage == "deadline_feasible_lowest_sla_risk"


def test_falls_back_to_lowest_queue_wait_when_no_deadline_feasible_candidate():
    strategy = DistributeOracleSLAGuardedLCWRA(config={"min_deadline_slack_min": 15})
    slow = _plan(1, deadline_feasible=False, estimated_queue_wait_min=30.0)
    fast = _plan(2, deadline_feasible=False, estimated_queue_wait_min=5.0)

    selected = strategy._select_plan([slow, fast])

    assert selected.dest_dc_id == 2
    assert selected.selected_guard_stage == "lowest_estimated_queue_wait"
