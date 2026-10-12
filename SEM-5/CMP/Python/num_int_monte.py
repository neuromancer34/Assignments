import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
import numpy as np

N = 10000  # Number of samples

# Generate the random points
r1 = np.random.uniform(0, 1, N)
r2 = np.random.uniform(0, 1, N)

f_x = np.exp(-10 * (r1 - 0.5) ** 2)  # Function

n = np.sum(r2 <= f_x)  # Number of valid points
prob = n / N  # Probabilty/Integral(for Area=1)
valid_x = r1[r2 <= f_x]

plt.hist(valid_x, bins=50, density=True, edgecolor="black")

x_smooth = np.linspace(0, 1, 100)
y_smooth = np.exp(-10 * (x_smooth - 0.5) ** 2)
A = np.trapezoid(y_smooth, x_smooth)  # Area under curve/Integral
n_prob = y_smooth / A  # Scaling Factor

plt.plot(x_smooth, n_prob, "r-", linewidth=2)
plt.show()
print(f"Integral by Monte Carlo: {prob:.3f}")
print(f"Integral by Trapezoidal Method: {A:.3f}")


sample_sizes = range(10, N + 1, 30)
int_estimates = []

for n_pts in sample_sizes:
    r1_c = np.random.uniform(0, 1, n_pts)
    r2_c = np.random.uniform(0, 1, n_pts)
    int_estimates.append(np.mean(r2_c <= np.exp(-10 * (r1_c - 0.5) ** 2)))  # Integral

plt.plot(sample_sizes, int_estimates)
plt.xlabel("Number of points (n)")
plt.ylabel("Estimated integral")
plt.show()
print(f"Final Integral: {int_estimates[-1]:.3f}")
