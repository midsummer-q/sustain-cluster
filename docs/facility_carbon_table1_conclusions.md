# Facility-Aware Carbon Table 1 Conclusions

This note summarizes the baseline results after adding facility-aware system carbon metrics.

## Main Conclusions

1. `RBC (Lowest Carbon)` minimizes the legacy task-level carbon proxy in this run, but it does not minimize all system costs.
   - `legacy_task_carbon_kg`: `1,946,895.85`, much lower than local-only, random, round-robin, and least-pending baselines.
   - `total_dc_carbon_emissions_kg`: `310,518.94`, also the lowest datacenter-side carbon among the reported RBC baselines.
   - This confirms that the old lowest-carbon objective is effective for the narrow task-level / current-CI carbon signal.

2. The same `RBC (Lowest Carbon)` result shows a clear system-level tradeoff.
   - `total_system_carbon_emissions_kg`: `311,533.30`, still the lowest in this table, but it includes `1,014.36 kg` transmission carbon.
   - `total_system_energy_kwh`: `1,124,785.37`, higher than local-only, random, round-robin, least-pending, and lowest-price.
   - `sla_violation_count`: `247.30`, much higher than local-only, random, round-robin, and least-pending.
   - The lower carbon result is therefore not free: it is achieved with more energy use and materially worse SLA behavior.

3. `RBC (Lowest Price)` is best for monetary cost, not carbon.
   - `total_system_energy_cost_usd`: `97,690.70`, the lowest reported system energy cost.
   - `total_system_carbon_emissions_kg`: `325,438.39`, higher than `RBC (Lowest Carbon)`.
   - This separates economic optimization from facility-aware carbon optimization.

4. Local or balanced baselines avoid most SLA violations, but they do not minimize carbon.
   - `RBC (Local Only)`: zero transmission carbon, but `total_system_carbon_emissions_kg` is `327,499.73`.
   - `RBC (Round Robin)` and `RBC (Least Pending)` have zero SLA violations, but system carbon is around `329k kg`.
   - These baselines are stable operationally but less carbon-efficient under this workload.

## Interpretation

The first-layer metrics make the old proxy-vs-system distinction visible:

- `legacy_task_carbon_kg` measures a task-level proxy based on task resource demand and destination current carbon intensity.
- `total_dc_carbon_emissions_kg` measures facility-side datacenter emissions from the simulator.
- `total_transmission_carbon_emissions_kg` measures network-transfer emissions.
- `total_system_carbon_emissions_kg = total_dc_carbon_emissions_kg + total_transmission_carbon_emissions_kg`.

In this particular table, `RBC (Lowest Carbon)` also has the lowest `total_system_carbon_emissions_kg`, so this run does not show a reversal where legacy carbon improves while system carbon worsens. It does show that the legacy objective hides important operational costs: transmission carbon, higher total energy, and substantially higher SLA violations.

## Why Layer 2 Is Needed

The next issue is reachability. `lowest_carbon` chooses by current carbon intensity among currently schedulable datacenters. It does not estimate whether the task can actually arrive, wait in queue, start, and finish during a low-carbon execution window before its deadline.

Layer 2 should therefore add `reachable_low_carbon` and audit metrics:

- `reachable_lcw_selection_rate`
- `low_carbon_miss_rate`
- `avg_low_carbon_overlap_ratio`
- `planned_start_mae_min`
- `planned_finish_mae_min`

These metrics will show whether a selected low-carbon target was physically and temporally reachable, not just theoretically attractive by current carbon intensity.
