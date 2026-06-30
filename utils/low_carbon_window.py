import pandas as pd
import numpy as np


def get_future_ci_values(dc, horizon_steps: int, norm: bool = False) -> list[float]:
    """Read future CI values from the datacenter's local CI manager."""
    ci_manager = getattr(dc, "ci_manager", None)
    if ci_manager is None or horizon_steps <= 0:
        return []

    series = getattr(ci_manager, "norm_carbon", None) if norm else getattr(ci_manager, "carbon_smooth", None)
    time_step = getattr(ci_manager, "time_step", None)
    if series is None or time_step is None:
        return []

    start = max(0, int(time_step))
    end = min(len(series), start + int(horizon_steps))
    if start >= end:
        return []

    return [float(value) for value in series[start:end]]


def find_low_carbon_windows(
    dc,
    current_time,
    timestep_minutes: int,
    horizon_steps: int,
    low_carbon_quantile: float,
    min_window_steps: int = 1,
):
    """Find contiguous low-carbon windows in the local CI forecast horizon."""
    future_ci_values = get_future_ci_values(dc, horizon_steps=horizon_steps, norm=False)
    if not future_ci_values:
        return []

    percentile = low_carbon_quantile * 100.0 if low_carbon_quantile <= 1.0 else low_carbon_quantile
    threshold = float(np.percentile(future_ci_values, percentile))
    min_window_steps = max(1, int(min_window_steps))
    windows = []
    window_start_idx = None

    for idx, ci in enumerate(future_ci_values):
        if ci <= threshold:
            if window_start_idx is None:
                window_start_idx = idx
        elif window_start_idx is not None:
            _append_window_if_long_enough(
                windows, current_time, timestep_minutes, window_start_idx, idx, threshold, min_window_steps
            )
            window_start_idx = None

    if window_start_idx is not None:
        _append_window_if_long_enough(
            windows,
            current_time,
            timestep_minutes,
            window_start_idx,
            len(future_ci_values),
            threshold,
            min_window_steps,
        )

    return windows


def compute_window_overlap_ratio(planned_start_time, planned_finish_time, windows) -> float:
    """Return execution-time overlap with low-carbon windows divided by task duration."""
    start = pd.Timestamp(planned_start_time)
    finish = pd.Timestamp(planned_finish_time)
    duration_seconds = (finish - start).total_seconds()
    if duration_seconds <= 0:
        return 0.0

    overlap_seconds = 0.0
    for window in windows:
        window_start, window_end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        overlap_start = max(start, window_start)
        overlap_end = min(finish, window_end)
        if overlap_end > overlap_start:
            overlap_seconds += (overlap_end - overlap_start).total_seconds()

    return min(1.0, max(0.0, overlap_seconds / duration_seconds))


def _append_window_if_long_enough(windows, current_time, timestep_minutes, start_idx, end_idx, threshold, min_window_steps):
    if end_idx - start_idx < min_window_steps:
        return

    start = pd.Timestamp(current_time) + pd.Timedelta(minutes=start_idx * timestep_minutes)
    end = pd.Timestamp(current_time) + pd.Timedelta(minutes=end_idx * timestep_minutes)
    windows.append((start, end, threshold))
