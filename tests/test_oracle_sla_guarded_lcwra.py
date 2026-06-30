from types import SimpleNamespace

import pandas as pd

from scripts.generate_table1_metrics_parallel import (
    _summarize_lcwra_audit_records,
    collect_unfinished_lcwra_audit_records,
)
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


def test_apply_sla_guard_computes_slack_safe_and_risk():
    strategy = DistributeOracleSLAGuardedLCWRA(
        config={
            "min_deadline_slack_min": 15,
            "queue_wait_safety_factor": 1.2,
            "max_estimated_queue_wait_min": 30,
        }
    )
    plan = _plan(
        1,
        reachable_lcw=True,
        estimated_queue_wait_min=10.0,
        predicted_marginal_system_carbon_kg=1.0,
    )
    task = SimpleNamespace(sla_deadline=plan.planned_finish_time + pd.Timedelta(minutes=20))

    strategy._apply_sla_guard(plan, task)

    assert plan.deadline_slack_min == 20.0
    assert plan.sla_safe is True
    assert abs(plan.sla_risk_score - 2.0) < 1e-9
    assert plan.rejected_by_sla_guard is False
    assert plan.queue_wait_over_guard_min == 0.0


def test_select_plan_respects_configured_fallback_order():
    strategy = DistributeOracleSLAGuardedLCWRA(
        config={
            "fallback_order": [
                "lowest_estimated_queue_wait",
                "reachable_lcw_and_sla_safe",
            ]
        }
    )
    safe_reachable_slow = _plan(
        1,
        reachable_lcw=True,
        deadline_slack_min=30.0,
        estimated_queue_wait_min=30.0,
        predicted_marginal_system_carbon_kg=0.1,
    )
    fast_non_lcw = _plan(
        2,
        reachable_lcw=False,
        deadline_slack_min=30.0,
        estimated_queue_wait_min=2.0,
        predicted_marginal_system_carbon_kg=10.0,
    )

    selected = strategy._select_plan([safe_reachable_slow, fast_non_lcw])

    assert selected.dest_dc_id == 2
    assert selected.selected_guard_stage == "lowest_estimated_queue_wait"


def test_candidate_rejected_rate_summary():
    records = [
        {
            "audit_stage": "selected",
            "rejected_by_sla_guard": False,
            "any_candidate_rejected_by_sla_guard": True,
            "rejected_by_sla_guard_candidate_count": 2,
            "candidate_count": 5,
        },
        {
            "audit_stage": "selected",
            "rejected_by_sla_guard": False,
            "any_candidate_rejected_by_sla_guard": False,
            "rejected_by_sla_guard_candidate_count": 0,
            "candidate_count": 3,
        },
        {
            "audit_stage": "selected",
            "rejected_by_sla_guard": True,
            "any_candidate_rejected_by_sla_guard": True,
            "rejected_by_sla_guard_candidate_count": 1,
            "candidate_count": 2,
        },
    ]

    summary = _summarize_lcwra_audit_records(records)

    assert summary["selected_plan_rejected_by_sla_guard_rate"] == 1 / 3
    assert summary["any_candidate_rejected_by_sla_guard_rate"] == 2 / 3
    assert summary["candidate_rejected_by_sla_guard_rate"] == 3 / 10
    assert summary["rejected_by_sla_guard_rate"] == summary["any_candidate_rejected_by_sla_guard_rate"]


def test_unfinished_lcwra_records_are_collected():
    plan = _plan(1, reachable_lcw=True, deadline_slack_min=20.0)
    task = SimpleNamespace(
        job_name="unfinished-task",
        origin_dc_id=1,
        dest_dc_id=1,
        sla_deadline=pd.Timestamp("2026-01-01T00:20:00Z"),
        start_time=None,
        finish_time=None,
        lcwra_selected_plan=plan,
        lcwra_candidate_plans=[plan],
        lcwra_config={"horizon_steps": 32, "timestep_minutes": 15, "low_carbon_quantile": 0.25},
    )
    dc = SimpleNamespace(pending_tasks=[task], running_tasks=[], ci_manager=None)
    cluster_manager = SimpleNamespace(datacenters={"DC1": dc}, in_transit_tasks=[])
    env = SimpleNamespace(
        cluster_manager=cluster_manager,
        in_transit_tasks=[],
        deferred_tasks=[],
    )

    records = collect_unfinished_lcwra_audit_records(
        env,
        controller_name="RBC (Oracle SLA-Guarded LCWRA)",
        seed=0,
        end_time=pd.Timestamp("2026-01-01T01:00:00Z"),
    )

    assert len(records) == 1
    assert records[0]["audit_stage"] == "unfinished"
    assert records[0]["unfinished_location"] == "pending"
    assert records[0]["sla_deadline_expired_at_episode_end"] is True
    assert records[0]["candidate_count"] == 1
    assert records[0]["rejected_by_sla_guard_candidate_count"] == 0
