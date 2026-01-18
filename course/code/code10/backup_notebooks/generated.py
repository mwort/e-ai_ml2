import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the polynomial function
def polynomial_function(x, y):
    return x**2 + y**2  # Example: z = x^2 + y^2

# Create a grid of points
x = np.linspace(-5, 5, 100)
y = np.linspace(-6, 6, 100)
x, y = np.meshgrid(x, y)
z = polynomial_function(x, y)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

# Set labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Save the plot
plt.savefig('my2d.png')
plt.show()