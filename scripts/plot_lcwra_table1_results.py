from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplconfig").resolve()))
import matplotlib.pyplot as plt
import numpy as np


controllers = [
    "Least\nPending",
    "Local\nOnly",
    "Lowest\nCarbon",
    "Lowest\nPrice",
    "Most\nAvailable",
    "Random",
    "LCWRA",
    "Round\nRobin",
]

system_carbon = np.array([328297.11, 327499.73, 311820.47, 325709.63, 326104.14, 329511.67, 314445.47, 329739.24])
carbon_std = np.array([7354.66, 7180.35, 7648.00, 6939.34, 6636.20, 7653.32, 6560.19, 7249.46])
sla_rate = np.array([21.96, 0.02, 21.34, 21.56, 21.40, 21.00, 15.48, 20.93])
sla_std = np.array([0.21, 0.03, 0.26, 0.90, 0.22, 0.12, 0.57, 0.11])
trans_cost = np.array([3735.18, 0.00, 3274.04, 3562.30, 2927.67, 3577.67, 2484.44, 3580.40])
trans_std = np.array([42.44, 0.00, 206.34, 215.05, 32.36, 29.36, 101.24, 48.17])

lcwra_idx = controllers.index("LCWRA")
colors = ["#8a8f98"] * len(controllers)
colors[lcwra_idx] = "#1f77b4"
colors[2] = "#2ca02c"
colors[1] = "#6f6f6f"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

fig = plt.figure(figsize=(15, 9), dpi=180)
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(controllers, system_carbon / 1000, yerr=carbon_std / 1000, color=colors, capsize=3, edgecolor="white", linewidth=0.8)
ax1.set_title("System CO2 Emissions")
ax1.set_ylabel("tCO2")
ax1.axhline(system_carbon[lcwra_idx] / 1000, color="#1f77b4", linestyle="--", linewidth=1, alpha=0.75)
ax1.grid(axis="y", alpha=0.25)
ax1.text(lcwra_idx, system_carbon[lcwra_idx] / 1000 + 4.5, "LCWRA\n314.4 t", ha="center", color="#1f77b4", fontweight="bold")

ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(controllers, sla_rate, yerr=sla_std, color=colors, capsize=3, edgecolor="white", linewidth=0.8)
ax2.set_title("SLA Violation Rate")
ax2.set_ylabel("%")
ax2.axhline(sla_rate[lcwra_idx], color="#1f77b4", linestyle="--", linewidth=1, alpha=0.75)
ax2.grid(axis="y", alpha=0.25)
ax2.text(lcwra_idx, sla_rate[lcwra_idx] + 1.0, "LCWRA\n15.48%", ha="center", color="#1f77b4", fontweight="bold")

ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(controllers, trans_cost, yerr=trans_std, color=colors, capsize=3, edgecolor="white", linewidth=0.8)
ax3.set_title("Transmission Cost")
ax3.set_ylabel("USD")
ax3.axhline(trans_cost[lcwra_idx], color="#1f77b4", linestyle="--", linewidth=1, alpha=0.75)
ax3.grid(axis="y", alpha=0.25)
ax3.text(lcwra_idx, trans_cost[lcwra_idx] + 160, "LCWRA\n$2.48k", ha="center", color="#1f77b4", fontweight="bold")

ax4 = fig.add_subplot(gs[1, 1])
sizes = 80 + (trans_cost / max(trans_cost.max(), 1)) * 380
ax4.scatter(system_carbon / 1000, sla_rate, s=sizes, c=colors, alpha=0.85, edgecolor="white", linewidth=1.0)
for i, label in enumerate(controllers):
    ax4.text(system_carbon[i] / 1000 + 0.5, sla_rate[i] + 0.55, label.replace("\n", " "), fontsize=8)
ax4.set_title("Carbon-SLA Trade-off\n(point size = transmission cost)")
ax4.set_xlabel("System CO2 Emissions (tCO2)")
ax4.set_ylabel("SLA Violation Rate (%)")
ax4.grid(alpha=0.25)

inset = ax4.inset_axes([0.58, 0.06, 0.39, 0.38])
audit_labels = ["Reachable\nselection", "Planned\noverlap", "Actual\noverlap", "Actual\nmiss"]
audit_vals = [0.78, 0.73, 0.71, 0.27]
audit_colors = ["#1f77b4", "#4c9fd6", "#65b6e8", "#d95f02"]
audit_x = np.arange(len(audit_labels))
inset.bar(audit_x, audit_vals, color=audit_colors, edgecolor="white", linewidth=0.7)
inset.set_xticks(audit_x)
inset.set_xticklabels(audit_labels)
inset.set_ylim(0, 1)
inset.set_title("LCWRA audit", fontsize=9)
inset.tick_params(axis="both", labelsize=7)
inset.grid(axis="y", alpha=0.2)
for i, value in enumerate(audit_vals):
    inset.text(i, value + 0.03, f"{value:.2f}", ha="center", fontsize=7)

fig.suptitle("Table 1 Controller Comparison: Facility-aware Carbon + Reachable Low-carbon Routing", fontsize=15, fontweight="bold", y=0.98)
fig.text(
    0.5,
    0.015,
    "LCWRA balances near-lowest carbon with lower SLA violations and transmission cost among cross-DC strategies.",
    ha="center",
    fontsize=10,
)

output_path = Path("outputs/lcwra_table1_visualization.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, bbox_inches="tight", facecolor="white")
print(output_path.resolve())
