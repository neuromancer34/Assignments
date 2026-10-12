import matplotlib.pyplot as plt
import numpy as np

# PARAMETERS
A = 3.0
B = 2.0
C = 1.0
D = 2.0
E = 1.0
F = 1.0

CFL = 0.00005  # CFL factor
dt = CFL / max(A, B, C, D, E, F)

t_max = 60.0
time_array = np.arange(0, t_max, dt)
n_pop = len(time_array)

# OUTPUT 1: TIME EVOLUTION PLOT
r = np.zeros(n_pop)
s = np.zeros(n_pop)
r[0] = 0.1
s[0] = 0.3

for i in range(1, n_pop):
    r_dot = A * s[i - 1] - B * r[i - 1] * s[i - 1] - C * (r[i - 1] ** 2)
    s_dot = D * r[i - 1] + E * r[i - 1] * s[i - 1] - F * (s[i - 1] ** 2)

    r[i] = r[i - 1] + r_dot * dt
    s[i] = s[i - 1] + s_dot * dt

plt.figure(1, figsize=(10, 5))
plt.plot(time_array, r, label="Rabbits r(t)", color="royalblue", lw=1.5)
plt.plot(time_array, s, label="Sheep s(t)", color="crimson", lw=1.5)
plt.title("Rabbit-Sheep Population Evolution Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Population")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# OUTPUT 2: PHASE SPACE DIAGRAM
values_to_test = [(0.01, 2.0), (1.0, 1.0), (2.0, 2.0), (1.5, 1.5)]

plt.figure(2, figsize=(8, 6))
plt.title("Phase Space Diagram")
plt.xlabel("Rabbit Population (r)")
plt.ylabel("Sheep Population (s)")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True, linestyle="--", alpha=0.6)

for r0, s0 in values_to_test:
    r_traj = np.zeros(n_pop)
    s_traj = np.zeros(n_pop)
    r_traj[0] = r0
    s_traj[0] = s0

    for i in range(1, n_pop):
        r_dot = (
            A * s_traj[i - 1]
            - B * r_traj[i - 1] * s_traj[i - 1]
            - C * (r_traj[i - 1] ** 2)
        )
        s_dot = (
            D * r_traj[i - 1]
            + E * r_traj[i - 1] * s_traj[i - 1]
            - F * (s_traj[i - 1] ** 2)
        )

        r_traj[i] = r_traj[i - 1] + r_dot * dt
        s_traj[i] = s_traj[i - 1] + s_dot * dt

    plt.plot(r_traj, s_traj, label=f"({r0}, {s0})", lw=1.2)

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
    1.2,
    2.4,
    marker="o",
    color="gold",
    markeredgecolor="black",
    markersize=8,
    label="Fixed Point (1.2, 2.4) [Stable]",
)

plt.legend()
plt.show()
