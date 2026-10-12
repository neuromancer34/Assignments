import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. VISUALIZATION FUNCTIONS
# ==========================================
def plot_heatmap(final_x, final_y, dimension_name):
    """Plots a 2D Heatmap."""
    plt.figure(figsize=(8, 6))
    plt.hist2d(final_x, final_y, bins=100, cmap="viridis")
    cbar = plt.colorbar()
    cbar.set_label("Frequency")
    plt.title(
        f"{dimension_name} Random Walk Heatmap (Walkers: {n_walkers}, Steps: {n_steps})"
    )
    plt.xlabel("Final X-Position")
    plt.ylabel("Final Y-Position")
    plt.show()


# ==========================================
# 2. SIMULATION FUNCTION
# ==========================================
def simulate_walk(choices, n_walkers, step_counts):
    """Simulates a random walk."""
    n_intervals = len(step_counts)
    fwhm = np.zeros(n_intervals)
    origin_prob = np.zeros(n_intervals)
    dimensions = len(choices[0])

    current_positions = np.zeros((n_walkers, dimensions))
    last_checkpoint = 0
    for i in range(n_intervals):
        current_checkpoint = step_counts[i]
        steps_to_take = current_checkpoint - last_checkpoint
        num_choices = len(choices)
        random_indices = np.random.choice(num_choices, size=(n_walkers, steps_to_take))
        steps = choices[random_indices]
        total_steps = np.sum(steps, axis=1)
        current_positions = current_positions + total_steps

        # FWHM
        std_dev = np.std(current_positions, axis=0)
        fwhm[i] = np.mean(2.355 * std_dev)

        # Origin Probability:
        distance_from_origin = np.sum(np.abs(current_positions), axis=1)
        walkers_at_origin = np.sum(distance_from_origin == 0)
        origin_prob[i] = walkers_at_origin / n_walkers

        last_checkpoint = current_checkpoint

    return fwhm, origin_prob, current_positions


# ==========================================
# 3. LINE FITTING FUNCTION
# ==========================================
def line_fit(step_counts, origin_probs):
    """Fits a line to the origin probabilities."""
    # Check for zero error
    valid_mask = origin_probs > 0
    valid_steps = step_counts[valid_mask]
    valid_probs = origin_probs[valid_mask]
    log_steps = np.log10(valid_steps)
    log_probs = np.log10(valid_probs)

    # Calculates slope and intercept and performs line fit
    slope, intercept = np.polyfit(log_steps, log_probs, 1)
    fit_line = (slope * np.log10(step_counts)) + intercept
    actual_fit = 10**fit_line

    return actual_fit, slope


# ==========================================
# 4. SETUP
# ==========================================
print("Starting Random Walk Simulator...")
n_walkers = int(input("Enter the number of walkers (e.g., 100000): "))
n_steps = int(input("Enter the number of steps (e.g., 1000): "))
step_counts = np.array(range(100, n_steps + 1, 100))

choices_1d = np.array([[-1], [1]])
choices_2d = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
choices_3d = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
)


# ==========================================
# 5. FINAL POSITION HISTOGRAM
# ===========================================

# 1D
print("Simulating 1D Walk...")
fwhm_1d, prob_1d, final_1d = simulate_walk(choices_1d, n_walkers, step_counts)
plt.figure(figsize=(8, 5))
plt.hist(final_1d[:, 0], bins=100, edgecolor="black", alpha=0.7)
plt.grid(True, linestyle="--", alpha=0.6)
plt.title(f"1D Random Walk (Walkers: {n_walkers}, Steps: {n_steps})")
plt.xlabel("Final Position")
plt.ylabel("Frequency")
plt.show()

# 2D
print("Simulating 2D Walk...")
fwhm_2d, prob_2d, final_2d = simulate_walk(choices_2d, n_walkers, step_counts)
plot_heatmap(final_2d[:, 0], final_2d[:, 1], "2D")

# 3D
print("Simulating 3D Walk...")
fwhm_3d, prob_3d, final_3d = simulate_walk(choices_3d, n_walkers, step_counts)
plot_heatmap(final_3d[:, 0], final_3d[:, 1], "3D Top-Down")


# ==========================================
# 6. FWHM-ORIGIN PLOTS & FITTING
# ==========================================

# FWHM Plot
print("Plotting FWHM...")
plt.figure(figsize=(8, 5))
plt.plot(step_counts, fwhm_1d, marker="o", color="blue", label="1D")
plt.plot(step_counts, fwhm_2d, marker="s", color="green", label="2D")
plt.plot(step_counts, fwhm_3d, marker="^", color="purple", label="3D")
plt.grid(True, linestyle="--", alpha=0.6)
plt.title(f"Random Walk FWHM Comparison (Walkers: {n_walkers}, Steps: {n_steps})")
plt.xlabel("Number of Steps")
plt.ylabel("FWHM")
plt.legend()
plt.show()

# Calculate the lines of best fit for Origin Plot
print("Plotting Origin-return distribution and calculating Slopes...")
fit_1d, slope_1d = line_fit(step_counts, prob_1d)
fit_2d, slope_2d = line_fit(step_counts, prob_2d)
fit_3d, slope_3d = line_fit(step_counts, prob_3d)

print(f"1D Slope: {slope_1d:.3f}")
print(f"2D Slope: {slope_2d:.3f}")
print(f"3D Slope: {slope_3d:.3f}")

# Origin-Return Plot
plt.figure(figsize=(8, 5))
plt.plot(step_counts, prob_1d, marker="o", color="blue", label="1D")
plt.plot(step_counts, prob_2d, marker="s", color="green", label="2D")
plt.plot(step_counts, prob_3d, marker="^", color="purple", label="3D")

plt.plot(step_counts, fit_1d, color="blue", linestyle="--", label="1D Fit Line")
plt.plot(step_counts, fit_2d, color="green", linestyle="--", label="2D Fit Line")
plt.plot(step_counts, fit_3d, color="purple", linestyle="--", label="3D Fit Line")

plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.title(f"Random Walk Origins Comparison (Walkers: {n_walkers}, Steps: {n_steps})")
plt.xlabel("log(Number of Steps)")
plt.ylabel("log(Probability of returning to origin)")
plt.legend()
plt.show()
