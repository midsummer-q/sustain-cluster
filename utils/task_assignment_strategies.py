# utils/task_assignment_strategies.py

import os
import random
import math
import numpy as np
import logging # Use standard logging

from utils.config_loader import load_yaml
from utils.reachability_estimator import DEFAULT_LCWRA_CONFIG, estimate_candidate_plan


DEFAULT_ORACLE_SLA_GUARDED_LCWRA_CONFIG = {
    **DEFAULT_LCWRA_CONFIG,
    "ci_source_mode": "oracle",
    "sla_guard_enabled": True,
    "min_deadline_slack_min": 15,
    "queue_wait_safety_factor": 1.2,
    "max_estimated_queue_wait_min": None,
    "objective": "sla_guarded_marginal_system_carbon",
    "fallback_order": [
        "reachable_lcw_and_sla_safe",
        "deadline_feasible_lowest_sla_risk",
        "lowest_predicted_marginal_system_carbon_feasible",
        "lowest_estimated_queue_wait",
    ],
}

# --- Base Class ---
class BaseRBCStrategy:
    """Base class for all Rule-Based Controller strategies."""
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None, **kwargs):
        """
        Selects a destination datacenter ID for the given task.

        Args:
            task (Task): The task object to be assigned.
            datacenters (dict): A dictionary mapping DC names (e.g., "DC1")
                                to SustainDC environment objects.
            logger (logging.Logger, optional): Logger instance. Defaults to None.

        Returns:
            int or None: The numerical dc_id of the selected datacenter,
                         or None if no suitable datacenter is found.
        """
        raise NotImplementedError

    def reset(self):
        """Resets any internal state of the strategy (optional)."""
        pass

# --- Concrete Strategy Implementations ---

class DistributeMostAvailable(BaseRBCStrategy):
    """
    Assigns the task to the datacenter with the MOST available CPU cores
    AMONG THOSE THAT CAN SCHEDULE the task.
    """
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None, **kwargs):
        if not datacenters:
             if logger: logger.error("MostAvailable: No datacenters provided.")
             return None

        # --- Filter DCs that can schedule the task ---
        schedulable_dcs = []
        for dc_name, dc in datacenters.items():
            # Check if the DC object has the can_schedule method and if it returns True
            if hasattr(dc, 'can_schedule') and dc.can_schedule(task):
                schedulable_dcs.append(dc)
            # else:
            #     if logger: logger.debug(f"MostAvailable: DC {dc.dc_id} cannot schedule task {task.job_name}. Skipping.")

        # --- If no DC can schedule the task ---
        if not schedulable_dcs:
             if logger: logger.warning(f"MostAvailable: No datacenter can schedule task {task.job_name}. Cannot assign.")
             return None # Indicate no assignment possible

        # --- Find the best DC among the schedulable ones ---
        try:
            # Find the DC with the maximum available cores among the filtered list
            best_dc = max(schedulable_dcs, key=lambda dc: getattr(dc, 'available_cores', -float('inf')))
            if logger: logger.info(f"MostAvailable choice for task {task.job_name}: DC{best_dc.dc_id} ({getattr(best_dc,'available_cores', 0):.1f} cores avail)")
            return best_dc.dc_id
        except Exception as e:
            if logger: logger.error(f"MostAvailable error during max selection: {e}")
            # Fallback: return first schedulable DC's ID if error occurs
            return schedulable_dcs[0].dc_id if schedulable_dcs else None


class DistributeRandom(BaseRBCStrategy):
    """Randomly assigns the task to one of the available datacenters."""
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None, **kwargs):
        if not datacenters:
             if logger: logger.error("Random: No datacenters provided.")
             return None

        # Select a random DC name, then get the object
        random_dc_name = random.choice(list(datacenters.keys()))
        random_dc = datacenters[random_dc_name]
        if logger:
            logger.info(f"Random choice for task {task.job_name}: DC{random_dc.dc_id}")
        return random_dc.dc_id


class DistributePriorityOrder(BaseRBCStrategy):
    """
    Assigns tasks following a fixed priority order of DC names,
    selecting the first one that can schedule the task.
    """
    def __init__(self, priority_order=["DC1", "DC2", "DC3", "DC4", "DC5"]):
         # Default order, can be customized during instantiation if needed
        self.priority_order = priority_order
        super().__init__()

    def __call__(self, task, datacenters: dict, logger: logging.Logger = None, **kwargs):
        if not datacenters:
             if logger: logger.error("PriorityOrder: No datacenters provided.")
             return None

        for dc_name in self.priority_order:
            dc = datacenters.get(dc_name)
            # Check if DC exists and can schedule the task
            if dc and hasattr(dc, 'can_schedule') and dc.can_schedule(task):
                if logger: logger.info(f"PriorityOrder choice for task {task.job_name}: {dc_name} (DC{dc.dc_id})")
                return dc.dc_id # Return numerical ID

        # No available datacenter found in the priority list that can schedule
        if logger:
            logger.warning(f"PriorityOrder: Task {task.job_name} could not be assigned! No suitable DC found in priority list.")
        return None # Indicate no suitable DC found


class DistributeLowestPrice(BaseRBCStrategy):
    """
    Assigns the task to the available datacenter with the lowest current
    electricity price ($/MWh).
    """
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None, **kwargs):
        if not datacenters:
            if logger: logger.error("LowestPrice: No datacenters provided.")
            return None

        candidates = []
        for dc_name, dc in datacenters.items():
            # Check if DC can schedule and has price info
            if hasattr(dc, 'can_schedule') and dc.can_schedule(task) and hasattr(dc, 'price_manager'):
                try:
                    price = dc.price_manager.get_current_price()
                    if price is not None:
                        candidates.append((price, dc)) # Store price and DC object
                    else:
                         if logger: logger.warning(f"LowestPrice: Could not get price for {dc_name}")
                except Exception as e:
                     if logger: logger.error(f"LowestPrice: Error getting price for {dc_name}: {e}")
            # else:
            #      if logger: logger.debug(f"LowestPrice: Skipping {dc_name} (cannot schedule or no price manager)")

        if not candidates:
            if logger:
                logger.warning(f"LowestPrice: Task {task.job_name} could not be assigned! No schedulable DC found with price info.")
            # Fallback: maybe assign randomly or to first available? Or return None? Let's return None.
            return None

        # Find the DC with the minimum price among candidates
        candidates.sort(key=lambda item: item[0]) # Sort by price (first element of tuple)
        best_price, best_dc = candidates[0]

        if logger:
            logger.info(f"LowestPrice choice for task {task.job_name}: DC{best_dc.dc_id} (Price: {best_price:.2f} $/MWh)")
        return best_dc.dc_id


class DistributeLeastPending(BaseRBCStrategy):
    """Assigns the task to the datacenter with the fewest pending tasks."""
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None, **kwargs):
        if not datacenters:
            if logger: logger.error("LeastPending: No datacenters provided.")
            return None

        # Find the DC object with the minimum pending task queue length
        try:
             best_dc = min(datacenters.values(), key=lambda dc: len(getattr(dc, 'pending_tasks', [])))
             pending_count = len(getattr(best_dc, 'pending_tasks', []))
             if logger: logger.info(f"LeastPending choice for task {task.job_name}: DC{best_dc.dc_id} ({pending_count} pending)")
             return best_dc.dc_id
        except Exception as e:
             if logger: logger.error(f"LeastPending error: {e}")
             # Fallback
             return list(datacenters.values())[0].dc_id if datacenters else None


class DistributeLowestCarbon(BaseRBCStrategy):
    """
    Assigns the task to the available datacenter with the lowest current
    carbon intensity (gCO2/kWh).
    """
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None, **kwargs):
        if not datacenters:
            if logger: logger.error("LowestCarbon: No datacenters provided.")
            return None

        candidates = []
        for dc_name, dc in datacenters.items():
             # Check if DC can schedule and has CI info getter
             if hasattr(dc, 'can_schedule') and dc.can_schedule(task) and hasattr(dc, 'get_current_carbon_intensity'):
                try:
                    # Use the helper method that should now exist
                    ci = dc.get_current_carbon_intensity(norm=False) # Get raw gCO2/kWh
                    if ci is not None:
                        candidates.append((ci, dc))
                    else:
                         if logger: logger.warning(f"LowestCarbon: Could not get CI for {dc_name}")
                except Exception as e:
                     if logger: logger.error(f"LowestCarbon: Error getting CI for {dc_name}: {e}")
             # else:
             #      if logger: logger.debug(f"LowestCarbon: Skipping {dc_name} (cannot schedule or no CI getter)")

        if not candidates:
            if logger:
                logger.warning(f"LowestCarbon: Task {task.job_name} could not be assigned! No schedulable DC found with CI info.")
            return None # Indicate no suitable DC

        # Find the DC with the minimum CI among candidates
        candidates.sort(key=lambda item: item[0]) # Sort by CI
        best_ci, best_dc = candidates[0]

        if logger:
            logger.info(f"LowestCarbon choice for task {task.job_name}: DC{best_dc.dc_id} (CI: {best_ci:.2f} gCO2/kWh)")
        return best_dc.dc_id


class DistributeReachableLowCarbon(BaseRBCStrategy):
    """
    Chooses a datacenter whose estimated execution window can actually overlap
    with a low-carbon window under transmission delay, queue wait, resources,
    and deadline constraints.
    """
    def __init__(self, config_path="configs/env/lcwra_config.yaml", config=None):
        super().__init__()
        self.config = dict(DEFAULT_LCWRA_CONFIG)
        self.config.update(config or self._load_config(config_path))
        self.last_candidate_plans = []
        self.last_selected_plan = None

    def __call__(self, task, datacenters: dict, logger: logging.Logger = None, **kwargs):
        current_time = kwargs.get("current_time")
        cluster_manager = kwargs.get("cluster_manager")
        if current_time is None or cluster_manager is None:
            if logger:
                logger.warning("ReachableLowCarbon missing current_time or cluster_manager; falling back to least pending.")
            return DistributeLeastPending()(task, datacenters, logger=logger)

        candidate_plans = []
        for dc_name, dc in datacenters.items():
            try:
                candidate_plans.append(
                    estimate_candidate_plan(
                        task=task,
                        dest_dc=dc,
                        dest_dc_name=dc_name,
                        current_time=current_time,
                        cluster_manager=cluster_manager,
                        config=self.config,
                    )
                )
            except Exception as exc:
                if logger:
                    logger.warning(f"ReachableLowCarbon failed to estimate {dc_name} for task {task.job_name}: {exc}")

        self.last_candidate_plans = candidate_plans
        if not candidate_plans:
            if logger:
                logger.warning(f"ReachableLowCarbon found no candidates for task {task.job_name}; using origin DC.")
            return getattr(task, "origin_dc_id", None)

        reachable = [plan for plan in candidate_plans if plan.reachable_lcw]
        if reachable:
            selected_plan = min(reachable, key=lambda plan: plan.predicted_marginal_system_carbon_kg)
            selected_plan.reason = "selected_reachable_low_carbon"
        else:
            feasible = [
                plan for plan in candidate_plans
                if plan.resource_feasible and plan.deadline_feasible
            ]
            if feasible:
                selected_plan = min(feasible, key=lambda plan: plan.predicted_marginal_system_carbon_kg)
                selected_plan.reason = "fallback_lowest_predicted_marginal_system_carbon_feasible"
            else:
                selected_plan = min(candidate_plans, key=lambda plan: plan.estimated_queue_wait_min)
                selected_plan.reason = "fallback_lowest_estimated_queue_wait"

        self.last_selected_plan = selected_plan
        self._attach_plan_to_task(task, selected_plan, candidate_plans)

        if logger:
            logger.info(
                f"ReachableLowCarbon choice for task {task.job_name}: DC{selected_plan.dest_dc_id} "
                f"(reachable={selected_plan.reachable_lcw}, overlap={selected_plan.low_carbon_overlap_ratio:.3f}, "
                f"predicted_marginal_system_carbon={selected_plan.predicted_marginal_system_carbon_kg:.6f} kg, "
                f"reason={selected_plan.reason})"
            )

        return selected_plan.dest_dc_id

    def _load_config(self, config_path):
        if not config_path or not os.path.exists(config_path):
            return {}
        loaded = load_yaml(config_path) or {}
        return loaded.get("lcwra", loaded)

    def _attach_plan_to_task(self, task, selected_plan, candidate_plans):
        task.planned_start_time = selected_plan.planned_start_time
        task.planned_finish_time = selected_plan.planned_finish_time
        task.planned_dest_dc_id = selected_plan.dest_dc_id
        task.planned_low_carbon_window_start = selected_plan.low_carbon_window_start
        task.planned_low_carbon_window_end = selected_plan.low_carbon_window_end
        task.low_carbon_overlap_ratio = selected_plan.low_carbon_overlap_ratio
        task.reachable_lcw = selected_plan.reachable_lcw
        task.selected_plan_reason = selected_plan.reason
        task.lcwra_selected_plan = selected_plan
        task.lcwra_candidate_plans = candidate_plans
        task.lcwra_config = dict(self.config)


class DistributeOracleSLAGuardedLCWRA(DistributeReachableLowCarbon):
    """
    Oracle smoke-test variant of LCWRA.

    This strategy intentionally reuses the existing LCWRA future-CI estimator,
    which reads ``dc.ci_manager.carbon_smooth``. It is an oracle benchmark for
    Layer 3 value testing, not an online forecast algorithm.
    """
    def __init__(self, config_path="configs/env/oracle_sla_guarded_lcwra_config.yaml", config=None):
        BaseRBCStrategy.__init__(self)
        self.config = dict(DEFAULT_ORACLE_SLA_GUARDED_LCWRA_CONFIG)
        self.config.update(config or self._load_config(config_path))
        self.config["ci_source_mode"] = "oracle"
        self.last_candidate_plans = []
        self.last_selected_plan = None

    def __call__(self, task, datacenters: dict, logger: logging.Logger = None, **kwargs):
        current_time = kwargs.get("current_time")
        cluster_manager = kwargs.get("cluster_manager")
        if current_time is None or cluster_manager is None:
            if logger:
                logger.warning("OracleSLAGuardedLCWRA missing current_time or cluster_manager; falling back to least pending.")
            return DistributeLeastPending()(task, datacenters, logger=logger)

        candidate_plans = []
        for dc_name, dc in datacenters.items():
            try:
                plan = estimate_candidate_plan(
                    task=task,
                    dest_dc=dc,
                    dest_dc_name=dc_name,
                    current_time=current_time,
                    cluster_manager=cluster_manager,
                    config=self.config,
                )
                self._apply_sla_guard(plan, task)
                candidate_plans.append(plan)
            except Exception as exc:
                if logger:
                    logger.warning(f"OracleSLAGuardedLCWRA failed to estimate {dc_name} for task {task.job_name}: {exc}")

        self.last_candidate_plans = candidate_plans
        if not candidate_plans:
            if logger:
                logger.warning(f"OracleSLAGuardedLCWRA found no candidates for task {task.job_name}; using origin DC.")
            return getattr(task, "origin_dc_id", None)

        selected_plan = self._select_plan(candidate_plans, logger=logger)
        self.last_selected_plan = selected_plan
        self._attach_plan_to_task(task, selected_plan, candidate_plans)

        if logger:
            logger.info(
                f"OracleSLAGuardedLCWRA choice for task {task.job_name}: DC{selected_plan.dest_dc_id} "
                f"(stage={selected_plan.selected_guard_stage}, reachable={selected_plan.reachable_lcw}, "
                f"sla_safe={selected_plan.sla_safe}, slack={selected_plan.deadline_slack_min:.2f} min, "
                f"risk={selected_plan.sla_risk_score:.2f}, "
                f"predicted_marginal_system_carbon={selected_plan.predicted_marginal_system_carbon_kg:.6f} kg)"
            )

        return selected_plan.dest_dc_id

    def _load_config(self, config_path):
        if not config_path or not os.path.exists(config_path):
            return {}
        loaded = load_yaml(config_path) or {}
        return loaded.get("oracle_sla_guarded_lcwra", loaded.get("lcwra", loaded))

    def _apply_sla_guard(self, plan, task):
        plan.ci_source_mode = "oracle"
        plan.queue_wait_safety_factor = float(self.config.get("queue_wait_safety_factor", 1.2))
        plan.deadline_slack_min = self._deadline_slack_min(task, plan)

        min_slack = float(self.config.get("min_deadline_slack_min", 15))
        estimated_wait = float(getattr(plan, "estimated_queue_wait_min", float("inf")))
        max_wait = self._max_estimated_queue_wait_min()
        plan.max_estimated_queue_wait_min = max_wait

        finite_wait = math.isfinite(estimated_wait)
        under_max_wait = self._within_queue_wait_guard(plan)
        plan.queue_wait_over_guard_min = (
            None
            if max_wait is None or not finite_wait
            else max(0.0, estimated_wait - float(max_wait))
        )
        guard_enabled = bool(self.config.get("sla_guard_enabled", True))
        plan.sla_safe = (
            bool(getattr(plan, "resource_feasible", False))
            and bool(getattr(plan, "deadline_feasible", False))
            and (not guard_enabled or float(plan.deadline_slack_min) >= min_slack)
            and finite_wait
            and under_max_wait
        )
        wait_penalty = float("inf") if not finite_wait else max(
            0.0,
            estimated_wait * (plan.queue_wait_safety_factor - 1.0),
        )
        slack_penalty = (
            float("inf")
            if plan.deadline_slack_min is None or not math.isfinite(float(plan.deadline_slack_min))
            else max(0.0, min_slack - float(plan.deadline_slack_min))
        )
        plan.sla_risk_score = slack_penalty + wait_penalty
        plan.rejected_by_sla_guard = bool(getattr(plan, "reachable_lcw", False) and not plan.sla_safe)
        plan.selected_guard_stage = None
        return plan

    def _max_estimated_queue_wait_min(self):
        max_wait = self.config.get("max_estimated_queue_wait_min")
        if max_wait in (None, "None"):
            return None
        return float(max_wait)

    def _deadline_slack_min(self, task, plan):
        sla_deadline = getattr(task, "sla_deadline", None)
        planned_finish = getattr(plan, "planned_finish_time", None)
        if sla_deadline is None or planned_finish is None:
            return float("-inf")
        if planned_finish is None or planned_finish is np.nan:
            return float("-inf")
        if pd_is_timestamp_max(planned_finish):
            return float("-inf")
        try:
            return (sla_deadline - planned_finish).total_seconds() / 60.0
        except Exception:
            return float("-inf")

    def _select_plan(self, candidate_plans, logger=None):
        for stage in self.config.get("fallback_order", []):
            selected = self._select_plan_for_stage(stage, candidate_plans, logger=logger)
            if selected is not None:
                return selected

        selected = min(candidate_plans, key=lambda plan: _finite_or_inf(plan.estimated_queue_wait_min))
        selected.reason = "fallback_lowest_estimated_queue_wait_after_empty_configured_stages"
        selected.selected_guard_stage = "lowest_estimated_queue_wait"
        return selected

    def _select_plan_for_stage(self, stage, candidate_plans, logger=None):
        if stage == "reachable_lcw_and_sla_safe":
            candidates = [
                plan for plan in candidate_plans
                if getattr(plan, "reachable_lcw", False) and getattr(plan, "sla_safe", False)
            ]
            if not candidates:
                return None
            selected = min(candidates, key=lambda plan: _finite_or_inf(plan.predicted_marginal_system_carbon_kg))
            selected.reason = "selected_oracle_sla_guarded_reachable_low_carbon"
            selected.selected_guard_stage = stage
            return selected

        if stage == "deadline_feasible_lowest_sla_risk":
            candidates = [
                plan for plan in candidate_plans
                if getattr(plan, "resource_feasible", False)
                and getattr(plan, "deadline_feasible", False)
                and self._within_queue_wait_guard(plan)
            ]
            if not candidates:
                return None
            selected = min(
                candidates,
                key=lambda plan: (
                    _finite_or_inf(getattr(plan, "sla_risk_score", float("inf"))),
                    _finite_or_inf(getattr(plan, "predicted_marginal_system_carbon_kg", float("inf"))),
                ),
            )
            selected.reason = "fallback_deadline_feasible_lowest_sla_risk"
            selected.selected_guard_stage = stage
            return selected

        if stage == "lowest_predicted_marginal_system_carbon_feasible":
            resource_feasible = [
                plan for plan in candidate_plans
                if getattr(plan, "resource_feasible", False)
            ]
            deadline_feasible = [
                plan for plan in resource_feasible
                if getattr(plan, "deadline_feasible", False)
            ]
            candidates = deadline_feasible or resource_feasible
            if not candidates:
                return None
            selected = min(candidates, key=lambda plan: _finite_or_inf(plan.predicted_marginal_system_carbon_kg))
            selected.reason = "fallback_lowest_predicted_marginal_system_carbon_feasible"
            selected.selected_guard_stage = stage
            return selected

        if stage == "lowest_estimated_queue_wait":
            if not candidate_plans:
                return None
            selected = min(candidate_plans, key=lambda plan: _finite_or_inf(plan.estimated_queue_wait_min))
            selected.reason = "fallback_lowest_estimated_queue_wait"
            selected.selected_guard_stage = stage
            return selected

        message = f"OracleSLAGuardedLCWRA skipping unknown fallback_order stage: {stage}"
        if logger:
            logger.warning(message)
        else:
            logging.getLogger(__name__).warning(message)
        return None

    def _within_queue_wait_guard(self, plan):
        max_wait = self._max_estimated_queue_wait_min()
        estimated_wait = _finite_or_inf(getattr(plan, "estimated_queue_wait_min", float("inf")))
        return math.isfinite(estimated_wait) and (max_wait is None or estimated_wait <= float(max_wait))


def _finite_or_inf(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return value if math.isfinite(value) else float("inf")


def pd_is_timestamp_max(value):
    try:
        import pandas as pd
        return pd.Timestamp(value) == pd.Timestamp.max
    except Exception:
        return False


class DistributeRoundRobin(BaseRBCStrategy):
    """Assigns tasks in a round-robin fashion across datacenters."""
    def __init__(self):
        self.last_assigned_dc_index = -1
        # Consistent order based on sorted numerical IDs
        self._dc_order_ids = []
        super().__init__()

    def reset(self):
        """Resets the round-robin index."""
        self.last_assigned_dc_index = -1
        self._dc_order_ids = [] # Clear the order cache
        if logging.getLogger().isEnabledFor(logging.DEBUG): # Avoid calculation if not debugging
             logging.debug("RoundRobin state reset.")

    def __call__(self, task, datacenters: dict, logger: logging.Logger = None, **kwargs):
        if not datacenters:
            if logger: logger.error("RoundRobin: No datacenters provided.")
            return None

        # Ensure a consistent order (sort by numerical dc_id)
        # Update the order only if the set of keys changes (more robust)
        current_dc_keys = sorted(datacenters.keys()) # Example using names, could sort by dc_id too
        current_dc_ids = sorted([dc.dc_id for dc in datacenters.values()])

        # Rebuild ordered list if it's empty or the IDs have changed
        if not self._dc_order_ids or self._dc_order_ids != current_dc_ids:
            self._dc_order_ids = current_dc_ids
            # Optionally reset index when DC set changes, or just continue cycle
            # self.last_assigned_dc_index = -1 # Uncomment to reset index on change
            if logger: logger.debug(f"RoundRobin order updated: {self._dc_order_ids}")

        if not self._dc_order_ids: # Should not happen if datacenters is not empty
             if logger: logger.error("RoundRobin: Failed to establish DC order.")
             return list(datacenters.values())[0].dc_id # Fallback

        # Increment index and wrap around
        self.last_assigned_dc_index = (self.last_assigned_dc_index + 1) % len(self._dc_order_ids)
        selected_dc_id = self._dc_order_ids[self.last_assigned_dc_index]

        # # Optional: Check if the selected DC can schedule the task
        # selected_dc = next((dc for dc in datacenters.values() if dc.dc_id == selected_dc_id), None)
        # if selected_dc and not selected_dc.can_schedule(task):
        #     if logger: logger.warning(f"RoundRobin selected DC{selected_dc_id} but it cannot schedule task {task.job_name}. Assigning anyway.")
        #     # Policy: Assign anyway, let the DC queue handle it. Or could try next DC.

        if logger:
            logger.info(f"RoundRobin choice for task {task.job_name}: DC{selected_dc_id} (index {self.last_assigned_dc_index})")

        return selected_dc_id # Return numerical ID


class DistributeLocalOnly(BaseRBCStrategy):
    """Assigns the task strictly to its origin datacenter."""
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None, **kwargs):
        if not hasattr(task, 'origin_dc_id') or task.origin_dc_id is None:
            if logger: logger.error(f"LocalOnly: Task {task.job_name} missing valid origin_dc_id.")
            # Decide fallback: Maybe assign randomly? Or return None? Let's return None.
            return None

        origin_id = task.origin_dc_id

        # Optional: Check if the origin DC actually exists in the current cluster setup
        origin_dc_exists = any(dc.dc_id == origin_id for dc in datacenters.values())
        if not origin_dc_exists:
             if logger: logger.error(f"LocalOnly: Origin DC {origin_id} for task {task.job_name} not found in current cluster configuration.")
             # Fallback strategy needed here as well. Assign randomly? Or None?
             # Let's assign randomly among existing DCs as a simple fallback.
             if not datacenters: return None
             fallback_dc = random.choice(list(datacenters.values()))
             if logger: logger.warning(f"LocalOnly: Assigning task {task.job_name} randomly to DC{fallback_dc.dc_id} as origin DC{origin_id} is missing.")
             return fallback_dc.dc_id


        if logger:
            logger.info(f"LocalOnly choice for task {task.job_name}: Assigning to origin DC{origin_id}")

        # Return the numerical origin ID
        return origin_id
# --- Add other strategies similarly ---

class DistributeLowestUtilization(BaseRBCStrategy):
    """Assigns the task to the datacenter with the highest overall average resource availability."""
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None, **kwargs):
        if not datacenters:
             if logger: logger.error("LowestUtilization: No datacenters provided.")
             return None

        def calculate_availability_score(dc):
            cpu_total = getattr(dc, 'total_cores', 0)
            gpu_total = getattr(dc, 'total_gpus', 0)
            mem_total = getattr(dc, 'total_mem_GB', 0)

            # Use getattr with defaults for safety
            cpu_avail = getattr(dc, 'available_cores', 0) / cpu_total if cpu_total > 0 else 0
            gpu_avail = getattr(dc, 'available_gpus', 0) / gpu_total if gpu_total > 0 else 0
            mem_avail = getattr(dc, 'available_mem', 0) / mem_total if mem_total > 0 else 0

            # Average availability - weights could be added
            return (cpu_avail + gpu_avail + mem_avail) / 3.0

        try:
            # Find DC with the maximum availability score
            best_dc = max(datacenters.values(), key=calculate_availability_score)
            if logger:
                score = calculate_availability_score(best_dc)
                logger.info(f"LowestUtilization (Max Avail) choice for task {task.job_name}: DC{best_dc.dc_id} (Score: {score:.3f})")
            return best_dc.dc_id
        except Exception as e:
            if logger: logger.error(f"LowestUtilization error: {e}")
            # Fallback
            return list(datacenters.values())[0].dc_id if datacenters else None


# Example of a Weighted Strategy (Needs helper methods in SustainDC)
# class DistributeWeighted(BaseRBCStrategy):
#     """
#     Assigns the task based on a weighted combination of normalized cost,
#     carbon intensity, and resource availability. Lower score is better.
#     NOTE: Requires normalization and assumes getter methods exist.
#     """
#     def __init__(self, weights={'cost': 0.3, 'carbon': 0.5, 'utilization': 0.2}):
#         self.weights = weights
#         super().__init__()

#     def __call__(self, task, datacenters: dict, logger: logging.Logger = None):
#         if not datacenters: return None

#         dc_scores = []
#         # --- Need to get ranges or use running stats for normalization ---
#         # Example: Placeholder normalization - replace with real stats
#         all_prices = [dc.price_manager.get_current_price() for dc in datacenters.values() if hasattr(dc,'price_manager')]
#         all_cis = [dc.get_current_carbon_intensity(norm=False) for dc in datacenters.values() if hasattr(dc,'get_current_carbon_intensity')]
#         min_price, max_price = min(all_prices) if all_prices else 0, max(all_prices) if all_prices else 1
#         min_ci, max_ci = min(all_cis) if all_cis else 0, max(all_cis) if all_cis else 1
#         price_range = max(1e-6, max_price - min_price)
#         ci_range = max(1e-6, max_ci - min_ci)
#         # --- End Placeholder Normalization ---

#         for dc_name, dc in datacenters.items():
#             try:
#                 norm_cost = (dc.price_manager.get_current_price() - min_price) / price_range if hasattr(dc,'price_manager') else 0.5
#                 norm_ci = (dc.get_current_carbon_intensity(norm=False) - min_ci) / ci_range if hasattr(dc,'get_current_carbon_intensity') else 0.5

#                 cpu_util = 1.0 - (getattr(dc,'available_cores',0) / getattr(dc,'total_cores',1))
#                 gpu_util = 1.0 - (getattr(dc,'available_gpus',0) / getattr(dc,'total_gpus',1))
#                 mem_util = 1.0 - (getattr(dc,'available_mem',0) / getattr(dc,'total_mem_GB',1))
#                 avg_util = (cpu_util + gpu_util + mem_util) / 3.0

#                 # Lower score is better: lower cost, lower ci, lower utilization (higher availability)
#                 score = (norm_cost * self.weights['cost'] +
#                          norm_ci * self.weights['carbon'] +
#                          avg_util * self.weights['utilization']) # Lower utilization = lower score = better? Check logic.

#                 dc_scores.append((score, dc))
#             except Exception as e:
#                  if logger: logger.error(f"Weighted scoring error for {dc_name}: {e}")

#         if not dc_scores:
#             if logger: logger.warning("Weighted: No DCs could be scored.")
#             return list(datacenters.values())[0].dc_id if datacenters else None

#         dc_scores.sort(key=lambda item: item[0]) # Sort by score (ascending)
#         best_score, best_dc = dc_scores[0]

#         if logger:
#             logger.info(f"Weighted choice for task {task.job_name}: DC{best_dc.dc_id} (Score: {best_score:.3f})")
#         return best_dc.dc_id
