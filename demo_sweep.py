"""
Demo Simulation Job for World C:
Executes a high-resolution Kuramoto critical sweep using colony_lib
and saves order parameter statistics and phase diagram.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from colony_lib.dynamics import simulate_kuramoto
from colony_lib.bifurcation import detect_critical_point

print("Starting Kuramoto Criticality Job in World C...")
K_values = np.linspace(0.5, 4.0, 15)
mean_rs = []
suscs = []

for K in K_values:
    res = simulate_kuramoto(n_oscillators=300, K=float(K), t_max=30.0, dt=0.03, seed=42)
    mean_rs.append(res["steady_mean_r"])
    suscs.append(res["susceptibility"])
    print(f"K = {K:.2f} -> steady r = {res['steady_mean_r']:.3f}, susc = {res['susceptibility']:.2f}")

mean_rs = np.array(mean_rs)
suscs = np.array(suscs)

crit = detect_critical_point(K_values, suscs)
print(f"Critical point detected at K_c = {crit['k_critical']:.4f}")

# Save plot
fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:blue'
ax1.set_xlabel('Coupling Strength K')
ax1.set_ylabel('Order Parameter <r>', color=color)
ax1.plot(K_values, mean_rs, 'o-', color=color, label='<r>')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Susceptibility chi', color=color)
ax2.plot(K_values, suscs, 's--', color=color, label='chi')
ax2.tick_params(axis='y', labelcolor=color)
ax2.axvline(crit['k_critical'], color='green', linestyle=':', label=f"K_c = {crit['k_critical']:.3f}")

plt.title('World C Demo: High-Resolution Kuramoto Criticality')
fig.tight_layout()
plt.savefig('kuramoto_sweep.png', dpi=150)
plt.close()

# Save JSON metrics
metrics = {
    "K_values": list(K_values),
    "mean_r": list(mean_rs),
    "susceptibility": list(suscs),
    "critical_point": crit
}
with open("kuramoto_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Job completed successfully. Generated kuramoto_sweep.png and kuramoto_metrics.json.")
