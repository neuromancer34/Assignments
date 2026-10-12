import matplotlib.pyplot as plt
import numpy as np

# PARAMETERS
alpha = 1.5  # Rabbit growth rate
beta = 1.0  # Rabbit decay rate
gamma = 3.0  # Fox decay rate
delta = 1.0  # Fox growth

C = 0.00005  # CFL factor
dt = C / max(alpha, gamma)

t_max = 60.0
time_array = np.arange(0, t_max, dt)
n_pop = len(time_array)

# OUTPUT 1: TIME EVOLUTION PLOT
r = np.zeros(n_pop)
f = np.zeros(n_pop)
r[0] = 0.1
f[0] = 0.3

for i in range(1, n_pop):
    r_dot = alpha * r[i - 1] - beta * r[i - 1] * f[i - 1]
    f_dot = -gamma * f[i - 1] + delta * r[i - 1] * f[i - 1]

    r[i] = r[i - 1] + r_dot * dt
    f[i] = f[i - 1] + f_dot * dt

plt.figure(1, figsize=(10, 5))
plt.plot(time_array, r, label="Rabbits r(t)", color="royalblue", lw=1.5)
plt.plot(time_array, f, label="Foxes f(t)", color="crimson", lw=1.5)
plt.title("Predator-Prey Population Evolution Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Population")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# OUTPUT 2: PHASE SPACE DIAGRAM
values_to_test = [(0.01, 2.0), (1.0, 1.0), (2.0, 2.0), (1.5, 1.5)]

plt.figure(2, figsize=(8, 6))
plt.title("Phase Space Diagram")
plt.xlabel("Rabbit Population (r)")
plt.ylabel("Fox Population (f)")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True, linestyle="--", alpha=0.6)

for r0, f0 in values_to_test:
    r_traj = np.zeros(n_pop)
    f_traj = np.zeros(n_pop)
    r_traj[0] = r0
    f_traj[0] = f0

    for i in range(1, n_pop):
        r_dot = alpha * r_traj[i - 1] - beta * r_traj[i - 1] * f_traj[i - 1]
        f_dot = -gamma * f_traj[i - 1] + delta * r_traj[i - 1] * f_traj[i - 1]

        r_traj[i] = r_traj[i - 1] + r_dot * dt
        f_traj[i] = f_traj[i - 1] + f_dot * dt

    plt.plot(r_traj, f_traj, label=f"({r0}, {f0})", lw=1.2)

# Highlight fixed points
plt.plot(
    0,
    0,
    marker="o",
    color="black",
    markersize=7,
    label="Fixed Point (0, 0) [Saddle]",
)
plt.plot(
    gamma / delta,
    alpha / beta,
    marker="o",
    color="gold",
    markeredgecolor="black",
    markersize=9,
    label=f"Fixed Point ({gamma / delta}, {alpha / beta}) [Stable]",
)

plt.legend()
plt.show()
