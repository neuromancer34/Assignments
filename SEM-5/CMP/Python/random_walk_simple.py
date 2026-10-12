import random
import statistics

import matplotlib.pyplot as plt

n_walkers = 1000
n_steps = 1000
step_counts = list(range(100, n_steps + 1, 100))

## 1D Walk
fwhm_values = []
origin_probabilities = []
for current_steps in range(100, n_steps + 1, 100):
    distances = []
    for walkers in range(n_walkers):
        positions = [0]
        for i in range(current_steps):
            step = random.choice([-1, 1])
            new_position = positions[-1] + step
            positions.append(new_position)
        distances.append(positions[-1])
    fwhm_values.append(2.355 * statistics.stdev(distances))
    origin_probabilities.append(distances.count(0) / n_walkers)

plt.hist(distances, edgecolor="black")
plt.title("1D Random Walk")
plt.xlabel("Final Position")
plt.ylabel("Frequency")
plt.show()

plt.plot(step_counts, fwhm_values)
plt.title("1D Random Walk FWHM")
plt.xlabel("Number of Steps")
plt.ylabel("FWHM")
plt.show()

plt.plot(step_counts, origin_probabilities)
plt.xscale("log")
plt.yscale("log")
plt.title("1D Random Walk Origins")
plt.xlabel("Number of Steps")
plt.ylabel("Probability of returning to origin")
plt.show()


## 2D Walk
fwhm_x_values = []
origin_probabilities_2d = []
for current_steps in range(100, n_steps + 1, 100):
    final_x = []
    final_y = []
    for walkers in range(n_walkers):
        x_positions = [0]
        y_positions = [0]
        choices = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for i in range(current_steps):
            step_x, step_y = random.choice(choices)
            new_x = x_positions[-1] + step_x
            new_y = y_positions[-1] + step_y
            x_positions.append(new_x)
            y_positions.append(new_y)
        final_x.append(x_positions[-1])
        final_y.append(y_positions[-1])
    fwhm_x_values.append(2.355 * statistics.stdev(final_x))
    origin_count = 0
    for i in range(n_walkers):
        if final_x[i] == 0 and final_y[i] == 0:
            origin_count += 1
    origin_probabilities_2d.append(origin_count / n_walkers)

plt.hist2d(final_x, final_y)
cbar = plt.colorbar()
cbar.set_label("Frequency")
plt.title("2D Random Walk")
plt.xlabel("Final X-Position")
plt.ylabel("Final Y-Position")
plt.show()

plt.plot(step_counts, fwhm_x_values)
plt.title("2D Random Walk FWHM")
plt.xlabel("Number of Steps")
plt.ylabel("FWHM")
plt.show()

plt.plot(step_counts, origin_probabilities_2d)
plt.xscale("log")
plt.yscale("log")
plt.title("2D Random Walk Origins")
plt.xlabel("Number of Steps")
plt.ylabel("Probability of returning to origin")
plt.show()

## 3D Walk
fwhm_x_values = []
origin_probabilities_3d = []
for current_steps in range(100, n_steps + 1, 100):
    final_x = []
    final_y = []
    final_z = []
    for walkers in range(n_walkers):
        x_positions = [0]
        y_positions = [0]
        z_positions = [0]
        choices = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for i in range(current_steps):
            step_x, step_y, step_z = random.choice(choices)
            new_x = x_positions[-1] + step_x
            new_y = y_positions[-1] + step_y
            new_z = z_positions[-1] + step_z
            x_positions.append(new_x)
            y_positions.append(new_y)
            z_positions.append(new_z)
        final_x.append(x_positions[-1])
        final_y.append(y_positions[-1])
        final_z.append(z_positions[-1])
    fwhm_x_values.append(2.355 * statistics.stdev(final_x))
    origin_count = 0
    for i in range(n_walkers):
        if final_x[i] == 0 and final_y[i] == 0 and final_z[i] == 0:
            origin_count += 1
    origin_probabilities_3d.append(origin_count / n_walkers)

plt.hist2d(final_x, final_y)
cbar = plt.colorbar()
cbar.set_label("Frequency")
plt.title("3D Random Walk")
plt.xlabel("Final X-Position")
plt.ylabel("Final Y-Position")
plt.show()

plt.plot(step_counts, fwhm_x_values)
plt.title("3D Random Walk FWHM")
plt.xlabel("Number of Steps")
plt.ylabel("FWHM")
plt.show()

plt.plot(step_counts, origin_probabilities_3d)
plt.xscale("log")
plt.yscale("log")
plt.title("3D Random Walk Origins")
plt.xlabel("Number of Steps")
plt.ylabel("Probability of returning to origin")
plt.show()
