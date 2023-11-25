import numpy as np
import matplotlib.pyplot as plt

# Function definition 
def func_sp(x, a, b, c, d, e, f):
    return a * np.sin(b * x) * np.exp(-c * x**2) + d * np.sin(e * x + f)

# Generating data points
np.random.seed(42)  # Seeding
num_points = 1000  # Number of data points
x_values = np.linspace(0, 0.5, num_points)  # X values from 0 to 0.5

# Parameters for the function
a, b, c, d, e, f = 2, 5, 10, 1, 20, np.pi / 3  # Function parameters
noise = np.random.normal(0, 0.1, num_points)  # Adding noise
y_values = func_sp(x_values, a, b, c, d, e, f) + noise  # Generating y values with added noise

# Saving data to a text file
data = np.column_stack((x_values, y_values))
np.savetxt('generated_data_sp.txt', data, fmt='%.8f', delimiter='\t', header='X\tY', comments='')

# Plotting the generated data
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, linestyle="-", label='Generated Data')
plt.title('Generated Data with Noise')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.savefig("sp.png")

