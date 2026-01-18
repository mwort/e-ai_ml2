import numpy as np
import matplotlib.pyplot as plt

x_values = np.linspace(-10, 10, 400)
y_values = f(x_values)

plt.plot(x_values, y_values)
plt.title('Plot of f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.savefig('plot.png')
plt.show()