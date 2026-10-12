import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# PARAMETERS
L = 10.0  # Box dimensions (L x L)
g = 9.81  # Gravitational acceleration
m = 1.0  # Mass of the particle
x, y = (L / 2), (L / 2)  # Initial coordinates
vx, vy = 5.0, 5.0  # Initial velocities
t_max = 20.0  # Simulation time

dx = 0.1
C = 0.1  # CFL factor
v_max = np.sqrt(vx**2 + vy**2)
dt = 0.5 * (dx / v_max) * C
steps_per_frame = 20
t = 0.0

# Initial energy at t = 0
K0 = 0.5 * m * (vx**2 + vy**2)
U0 = m * g * y
E0 = K0 + U0

# tracking lists starting at t = 0
x_history = [x]
y_history = [y]
t_history = [0.0]
k_history = [K0]
u_history = [U0]
e_history = [E0]

# PLOT
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left Panel: Particle Motion in the 2D Box
ax1.set_xlim(0, L)
ax1.set_ylim(0, L)
ax1.set_aspect("equal")
ax1.set_xlabel("X Position")
ax1.set_ylabel("Y Position")
ax1.grid(True, linestyle="--", alpha=0.5)

(trail_line,) = ax1.plot([], [], color="royalblue", lw=1.5, alpha=0.7)
(particle_dot,) = ax1.plot([], [], "ro", markersize=8)

# Right Panel: Energy vs. Time
ax2.set_xlim(0, t_max)
ax2.set_ylim(0, 1.2 * (E0 + m * g * L))
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Energy (J)")
ax2.grid(True, linestyle="--", alpha=0.5)

(line_k,) = ax2.plot([], [], color="crimson", lw=1.5, label="Kinetic (K)")
(line_u,) = ax2.plot([], [], color="forestgreen", lw=1.5, label="Potential (U)")
(line_e,) = ax2.plot([], [], color="black", lw=1.8, linestyle="--", label="Total (E)")
ax2.legend(loc="upper right")


# DYNAMICS
def update(frame):
    global x, y, vx, vy, t

    for i in range(steps_per_frame):
        x += vx * dt
        vy -= g * dt
        y += vy * dt
        t += dt

        # Wall reflection on X-axis
        if x < 0:
            x = -x
            vx = -vx
        elif x > L:
            x = 2 * L - x
            vx = -vx

        # Wall reflection on Y-axis
        if y < 0:
            y = -y
            vy = -vy
        elif y > L:
            y = 2 * L - y
            vy = -vy

        # Compute energy components
        K = 0.5 * m * (vx**2 + vy**2)
        U = m * g * y
        E = K + U

        x_history.append(x)
        y_history.append(y)
        t_history.append(t)
        k_history.append(K)
        u_history.append(U)
        e_history.append(E)

    # Update Trajectory of Particle
    trail_line.set_data(x_history, y_history)
    particle_dot.set_data([x], [y])
    ax1.set_title(f"Time: {t:.2f} s | Pos: ({x:.2f}, {y:.2f})")

    # Update Energy
    line_k.set_data(t_history, k_history)
    line_u.set_data(t_history, u_history)
    line_e.set_data(t_history, e_history)
    ax2.set_title(
        f"Total Energy E = {E:.2f} | Kinetic Energy K = {K:.2f} | Potential Energy U = {U:.2f} (J)"
    )

    return trail_line, particle_dot, line_k, line_u, line_e


# ANIMATION
total_frames = int(t_max / (dt * steps_per_frame))

ani = animation.FuncAnimation(
    fig,
    update,
    frames=total_frames,
    blit=False,
    repeat=False,
    interval=20,
)

plt.tight_layout()
plt.show()
