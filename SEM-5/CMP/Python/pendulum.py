import matplotlib.pyplot as plt
import numpy as np

# SETUP
g = 9.81
L = 1.0
t_max = 20.0
dt = 0.01
time_array = np.arange(0, t_max, dt)
n_steps = len(time_array)
angles_to_test = [0.5, 1.5, 2.5, 3.1]
calculated_periods = []

plt.figure(1, figsize=(7, 5))
plt.title("Phase Space Diagram")
plt.xlabel("Angular Position (rad)")
plt.ylabel("Angular Velocity (rad/s)")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True)

for theta_0 in angles_to_test:
    theta = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    theta[0] = theta_0

    # Euler-Cromer Method for calculating Angular Velocity and Angular Positions
    for i in range(1, n_steps):
        omega[i] = omega[i - 1] - (g / L) * np.sin(theta[i - 1]) * dt
        theta[i] = theta[i - 1] + omega[i] * dt
    plt.plot(theta, omega, label=f"Angle: {theta_0} rad")

    # Find Peaks
    peak_indices = []
    for i in range(1, n_steps):
        if omega[i - 1] > 0 and omega[i] <= 0:
            peak_indices.append(i)

    # Calculate Period
    if len(peak_indices) >= 2:
        peak_times = time_array[peak_indices]
        periods = np.diff(peak_times)
        avg_period = np.mean(periods)
    else:
        avg_period = 0.0
    calculated_periods.append(avg_period)

plt.legend()
plt.figure(2, figsize=(7, 5))
plt.plot(angles_to_test, calculated_periods, marker="o", markersize=8, color="teal")
for x, y in zip(angles_to_test, calculated_periods):
    plt.annotate(
        f"({x}, {y:.2f})",
        (x, y),
        textcoords="offset points",
        xytext=(8, -15),
        fontsize=10,
        ha="left",
    )
plt.title("Time Period vs. Initial Angle")
plt.xlabel("Initial Angle (rad)")
plt.ylabel("Time Period (seconds)")
plt.axhline(0, color="black", linewidth=0.5)
plt.grid(True)
plt.show()
