import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the compiled Fortran shared library
fortran_lib = ctypes.CDLL("./fortran_interface.so")

# Define function prototype
fortran_lib.f_sin_cos.argtypes = [ctypes.POINTER(ctypes.c_double)]
fortran_lib.f_sin_cos.restype = ctypes.c_double

# Generate 100 values of x between 0 and 2 pi
x_values = np.linspace(0, 2 * np.pi, 100)
y_values = np.zeros_like(x_values)

# Compute f(x) for each x, ensuring we pass it as a pointer
for i, x in enumerate(x_values):
    x_c = ctypes.c_double(x)  # Convert to C double
    y_values[i] = fortran_lib.f_sin_cos(ctypes.byref(x_c))  # Pass by reference

# Plot the function
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label=r"$f(x) = \sin(x) \cdot \cos(x)$", color='b')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title(r"Plot of $f(x) = \sin(x) \cdot \cos(x)$")
plt.axhline(0, color='gray', linestyle="--", linewidth=0.8)
plt.axvline(0, color='gray', linestyle="--", linewidth=0.8)
plt.legend()
plt.grid()
plt.show()