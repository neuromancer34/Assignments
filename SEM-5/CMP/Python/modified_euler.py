import matplotlib.pyplot as plt
import numpy as np

# Parameters
omega = 2.0  # Angular frequency
x = 1.0
v = 0.0
h = 0.1
t = np.linspace(0, 10, h)
xs = []

for i in range (len(t)-1):
    xs.append(x)
    # Predict
    xp =  
    vp = 
    # Correct
    x = 
    v = 

# Plot Phase Space
plt.figure()
plt.plot(x, v, label="Phase Space Trajectory")
plt.plot(x[0], v[0], "go", label="Start")
plt.plot(x[-1], v[-1], "ro", label="End")
plt.xlabel("Position (x)")
plt.ylabel("Velocity (v)")
plt.title("Phase Space Plot of Simple Harmonic Oscillator")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.show()
