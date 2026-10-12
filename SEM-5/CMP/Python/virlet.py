import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

## PARAMETERS
N = 27  # Number of gas particles
L = 6.0  # Box size
m = 1.0  # Particle mass

epsilon = 1.5  # LJ interaction energy
sigma = 0.5  # LJ particle size
R = 0.5 * sigma  # Particle radius
r_min = (2 ** (1 / 6)) * sigma
damping = 0.5

r_cutoff = 2.5 * sigma
n_cells = int(L / r_cutoff)  # Number of grid cells per axis
cell_size = L / n_cells

dt = 0.001  # Time step
steps_per_frame = 40
# t_max = 30.0  # Max time

# INITIAL POSITIONS
np.random.seed(42)
coords = np.linspace(1.0, L - 1.0, 3)
pos = np.array([[x, y, z] for x in coords for y in coords for z in coords])

# INITIAL VELOCITIES
vel = (np.random.rand(N, 3) - 0.5) * 10.0


## STANDARD METHOD
def compute_forces_standard(pos):
    forces = np.zeros((N, 3))
    potential_energy = 0.0
    r_clip = 0.7 * sigma

    for i in range(N):
        for j in range(i + 1, N):
            dr = pos[i] - pos[j]
            r = max(np.linalg.norm(dr), r_clip)

            r6 = (sigma / r) ** 6
            r12 = r6**2

            potential_energy += 4.0 * epsilon * (r12 - r6)
            f_mag = 24.0 * epsilon * (2.0 * r12 - r6) / (r**2)
            f_vec = f_mag * dr

            forces[i] += f_vec
            forces[j] -= f_vec

    return forces, potential_energy


## NEIGHBORING CELLS METHOD
def compute_forces_grid(pos):
    forces = np.zeros((N, 3))
    potential_energy = 0.0
    r_clip = 0.7 * sigma

    grid = {}
    for i in range(N):
        cx = int(pos[i, 0] / cell_size) % n_cells
        cy = int(pos[i, 1] / cell_size) % n_cells
        cz = int(pos[i, 2] / cell_size) % n_cells
        cell_key = (cx, cy, cz)
        if cell_key not in grid:
            grid[cell_key] = []
        grid[cell_key].append(i)

    for (cx, cy, cz), particles in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_key = (
                        (cx + dx) % n_cells,
                        (cy + dy) % n_cells,
                        (cz + dz) % n_cells,
                    )
                    if neighbor_key in grid:
                        for i in particles:
                            for j in grid[neighbor_key]:
                                if i < j:
                                    dr = pos[i] - pos[j]
                                    r_sq = np.dot(dr, dr)
                                    if r_sq < r_cutoff**2:
                                        r = max(np.sqrt(r_sq), r_clip)
                                        r6 = (sigma / r) ** 6
                                        r12 = r6**2

                                        potential_energy += 4.0 * epsilon * (r12 - r6)
                                        f_mag = (
                                            24.0 * epsilon * (2.0 * r12 - r6) / (r**2)
                                        )
                                        f_vec = f_mag * dr

                                        dv = vel[i] - vel[j]
                                        r_hat = dr / r
                                        v_normal = np.dot(dv, r_hat)

                                        if v_normal < 0:
                                            f_dissipative = (
                                                -damping * m * v_normal * r_hat
                                            )
                                            f_vec += f_dissipative

                                        forces[i] += f_vec
                                        forces[j] -= f_vec

    return forces, potential_energy


# Select grid method for active animation execution
compute_forces_and_energy = compute_forces_grid


# BENCHMARK
def run_benchmark(n_iterations=500):
    # Standard method timing
    start = time.perf_counter()
    for _ in range(n_iterations):
        compute_forces_standard(pos)
    t_standard = time.perf_counter() - start

    # Grid cell method timing
    start = time.perf_counter()
    for _ in range(n_iterations):
        compute_forces_grid(pos)
    t_grid = time.perf_counter() - start

    print(f" BENCHMARK (N = {N} Particles):")
    print(f" Standard Method:   {t_standard:.5f} s")
    print(f" Grid Cell Method:   {t_grid:.5f} s")


run_benchmark(n_iterations=500)

# Initialization
forces, U0 = compute_forces_and_energy(pos)
acc = forces / m

t = 0.0
t_hist, k_hist, u_hist, e_hist = [0.0], [0.5 * m * np.sum(vel**2)], [U0], []
e_hist.append(k_hist[0] + u_hist[0])


## PLOT
fig = plt.figure(figsize=(14, 5))

# Gas Particle dynamics in 3D Box
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax1.set_xlim(0, L)
ax1.set_ylim(0, L)
ax1.set_zlim(0, L)
ax1.set_aspect("equal")
ax1.set_xlabel("X Position")
ax1.set_ylabel("Y Position")
ax1.set_zlabel("Z Position")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.set_title("3D Molecular Dynamics")

colors = plt.cm.plasma(np.linspace(0, 1, N))
scatter = ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color=colors, s=100)

# Energy Plot
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_xlim(0, 25.0)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Energy (J)")
ax2.grid(True, linestyle="--", alpha=0.5)
(line_k,) = ax2.plot([], [], color="crimson", lw=1.5, label="Kinetic (K)")
(line_u,) = ax2.plot([], [], color="forestgreen", lw=1.5, label="Potential (U)")
(line_e,) = ax2.plot([], [], color="black", lw=2.0, linestyle="--", label="Total (E)")
ax2.legend(loc="upper right")

# Panel 2: LJ Curve
ax3 = fig.add_subplot(1, 3, 3)
r_range = np.linspace(0.85 * sigma, 2.5 * sigma, 300)
V_lj = 4.0 * epsilon * (((sigma / r_range) ** 12) - ((sigma / r_range) ** 6))
ax3.plot(r_range, V_lj, color="navy", lw=2, label=r"$V(r)$")
ax3.axhline(0, color="gray", ls="--", lw=1)
ax3.axvline(sigma, color="orange", ls=":", label=r"$\sigma$")
ax3.axvline(r_min, color="crimson", ls=":", label=r"r_min: $2^{1/6}\sigma$")
ax3.set_ylim(-1.5 * epsilon, 2.0 * epsilon)
ax3.set_xlabel("Distance r")
ax3.set_ylabel("Potential V(r)")
ax3.set_title("Lennard-Jones Potential Curve")
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend(loc="upper right")


## VELOCITY VERLET ANIMATION
def update(frame):
    global pos, vel, acc, t

    for _ in range(steps_per_frame):
        # Position Advance
        pos += vel * dt + 0.5 * acc * (dt**2)

        # Wall Reflections
        for i in range(N):
            for d in range(3):
                if pos[i, d] < R:
                    pos[i, d] = 2 * R - pos[i, d]
                    vel[i, d] = -vel[i, d]
                elif pos[i, d] > L - R:
                    pos[i, d] = 2 * (L - R) - pos[i, d]
                    vel[i, d] = -vel[i, d]

        # Force applied
        new_forces, potential_energy = compute_forces_and_energy(pos)
        new_acc = new_forces / m

        # Velocity Advance
        vel += 0.5 * (acc + new_acc) * dt
        acc = new_acc
        t += dt

        # Energy calculation
        kinetic_energy = 0.5 * m * np.sum(vel**2)
        total_energy = kinetic_energy + potential_energy

        t_hist.append(t)
        k_hist.append(kinetic_energy)
        u_hist.append(potential_energy)
        e_hist.append(total_energy)

        if total_energy < 0.0:
            ani.event_source.stop()
            break

    # Update 3D scatter coordinates
    scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])

    ax1.set_title(f"Time: {t:.2f}s | No. of Particles: {N}")

    line_k.set_data(t_hist, k_hist)
    line_u.set_data(t_hist, u_hist)
    line_e.set_data(t_hist, e_hist)

    ax2.set_ylim(min(u_hist) - 2.0, max(e_hist) + 5.0)
    ax2.set_title(f"Total System Energy E = {e_hist[-1]:.1f} J")

    return scatter, line_k, line_u, line_e


# Run Animation
# total_frames = int(t_max / (dt * steps_per_frame))
ani = animation.FuncAnimation(fig, update, blit=False, repeat=False, interval=20)

plt.tight_layout()
plt.show()
