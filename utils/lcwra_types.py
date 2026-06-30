from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class CandidatePlan:
    """Estimated task-to-datacenter plan used only for LCWRA scheduling audits."""

    task_id: str
    origin_dc_id: int
    dest_dc_id: int
    dest_dc_name: str
    decision_time: pd.Timestamp
    transmission_delay_min: float
    arrival_time: pd.Timestamp
    estimated_queue_wait_min: float
    planned_start_time: pd.Timestamp
    planned_finish_time: pd.Timestamp
    low_carbon_window_start: Optional[pd.Timestamp]
    low_carbon_window_end: Optional[pd.Timestamp]
    low_carbon_overlap_ratio: float
    resource_feasible: bool
    deadline_feasible: bool
    reachable_lcw: bool
    predicted_task_proxy_dc_carbon_kg: float
    predicted_task_proxy_system_carbon_kg: float
    predicted_marginal_dc_carbon_kg: float
    predicted_dc_carbon_kg: float
    predicted_transmission_carbon_kg: float
    predicted_marginal_system_carbon_kg: float
    predicted_system_carbon_kg: float
    reason: str
    ci_source_mode: Optional[str] = None
    deadline_slack_min: Optional[float] = None
    sla_safe: bool = False
    sla_risk_score: Optional[float] = None
    queue_wait_safety_factor: Optional[float] = None
    selected_guard_stage: Optional[str] = None
    rejected_by_sla_guard: bool = False
