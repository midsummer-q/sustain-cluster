"""System-level metric aggregation for facility-aware carbon evaluation."""


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def aggregate_system_metrics(cluster_info: dict) -> dict:
    """Aggregate facility-level totals without mutating the input structure.

    These totals are system-level / facility-level measurements from datacenter
    operations plus transmission overhead. They are intentionally separate from
    legacy task-level carbon proxy rewards that estimate emissions from tasks.
    """
    if not isinstance(cluster_info, dict):
        cluster_info = {}

    datacenter_infos = cluster_info.get("datacenter_infos", {})

    total_dc_energy_kwh = 0.0
    total_dc_carbon_emissions_kg = 0.0
    total_dc_energy_cost_usd = 0.0
    finished_tasks_count = 0
    sla_met_count = 0
    sla_violation_count = 0

    for dc_info in datacenter_infos.values():
        common = dc_info.get("__common__", {}) if isinstance(dc_info, dict) else {}
        sla = common.get("__sla__", {}) if isinstance(common, dict) else {}

        total_dc_energy_kwh += _as_float(common.get("energy_consumption_kwh", 0.0))
        total_dc_carbon_emissions_kg += _as_float(common.get("carbon_emissions_kg", 0.0))
        total_dc_energy_cost_usd += _as_float(common.get("energy_cost_USD", 0.0))
        finished_tasks_count += _as_int(common.get("finished_tasks_count", 0))
        sla_met_count += _as_int(sla.get("met", 0))
        sla_violation_count += _as_int(sla.get("violated", 0))

    total_transmission_energy_kwh = _as_float(cluster_info.get("transmission_energy_total_kwh", 0.0))
    total_transmission_carbon_emissions_kg = _as_float(cluster_info.get("transmission_emissions_total_kg", 0.0))
    total_transmission_cost_usd = _as_float(cluster_info.get("transmission_cost_total_usd", 0.0))

    total_system_energy_kwh = total_dc_energy_kwh + total_transmission_energy_kwh
    total_system_carbon_emissions_kg = total_dc_carbon_emissions_kg + total_transmission_carbon_emissions_kg
    total_system_energy_cost_usd = total_dc_energy_cost_usd + total_transmission_cost_usd
    sla_violation_rate = sla_violation_count / finished_tasks_count if finished_tasks_count > 0 else 0.0

    return {
        "total_dc_energy_kwh": total_dc_energy_kwh,
        "total_dc_carbon_emissions_kg": total_dc_carbon_emissions_kg,
        "total_dc_energy_cost_usd": total_dc_energy_cost_usd,
        "total_transmission_energy_kwh": total_transmission_energy_kwh,
        "total_transmission_carbon_emissions_kg": total_transmission_carbon_emissions_kg,
        "total_transmission_cost_usd": total_transmission_cost_usd,
        "total_system_energy_kwh": total_system_energy_kwh,
        "total_system_carbon_emissions_kg": total_system_carbon_emissions_kg,
        "total_system_energy_cost_usd": total_system_energy_cost_usd,
        "finished_tasks_count": finished_tasks_count,
        "sla_met_count": sla_met_count,
        "sla_violation_count": sla_violation_count,
        "sla_violation_rate": sla_violation_rate,
    }
