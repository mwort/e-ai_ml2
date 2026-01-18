import numpy as np
import matplotlib.pyplot as plt

# Define the polynomial function of mixed degree
def f(x):
    # Example polynomial: f(x) = 2x^3 - 3x^2 + 5x - 7
    return 2 * x**3 - 3 * x**2 + 5 * x - 7

# Define the range for x
x_values = np.linspace(-10, 10, 1000)

# Evaluate the polynomial
y_values = f(x_values)

# Find the maximum absolute value of the polynomial
max_abs_value = np.max(np.abs(y_values))

# Normalize the polynomial
def normalized_f(x):
    return (f(x) / max_abs_value) * 10  # Scale to ensure |f(x)| < 10

# Evaluate the normalized polynomial
normalized_y_values = normalized_f(x_values)

# Plot the normalized polynomial
plt.figure(figsize=(10, 6))
plt.plot(x_values, normalized_y_values, label='Normalized Polynomial', color='blue')
plt.axhline(10, color='red', linestyle='--', label='y = 10')
plt.axhline(-10, color='red', linestyle='--', label='y = -10')
plt.title('Normalized Polynomial Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.ylim(-11, 11)
plt.xlim(-10, 10)
plt.grid()
plt.legend()
plt.show()