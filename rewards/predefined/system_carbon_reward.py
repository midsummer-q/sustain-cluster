from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward


@register_reward("system_carbon")
class SystemCarbonReward(BaseReward):
    """
    Facility-aware system-level carbon objective.

    Unlike the legacy carbon_emissions reward, this does not estimate carbon by
    iterating over current tasks. It reads aggregated facility/system emissions
    from cluster_info, optionally including transmission emissions.
    """
    def __init__(self, normalize_factor: float = 100.0, include_transmission: bool = True):
        super().__init__()
        self.normalize_factor = float(normalize_factor) if normalize_factor != 0 else 1.0
        self.include_transmission = include_transmission

    def __call__(self, cluster_info, current_tasks, current_time):
        dc_carbon = float(cluster_info.get("total_dc_carbon_emissions_kg", 0.0))

        if self.include_transmission:
            if "total_system_carbon_emissions_kg" in cluster_info:
                carbon = float(cluster_info.get("total_system_carbon_emissions_kg", 0.0))
            else:
                transmission_carbon = float(cluster_info.get("total_transmission_carbon_emissions_kg", 0.0))
                carbon = dc_carbon + transmission_carbon
        else:
            carbon = dc_carbon

        reward = -carbon / self.normalize_factor
        self.last_reward = reward
        return reward

    def get_last_value(self):
        return self.last_reward
