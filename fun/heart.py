import numpy as np
import matplotlib.pyplot as plt

# Heart shape function
def heart(t):
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    return x, y

# Generate points
t = np.linspace(0, 2 * np.pi, 1000)
x, y = heart(t)

# Plot the heart
plt.figure(figsize=(6, 6))
plt.plot(x, y, color='red',mec = 'red',mfc = 'red')
plt.fill(x,y,color = 'red')
plt.axis('equal')
plt.axis('off')
plt.title("My Heart", fontsize=16)
plt.show()