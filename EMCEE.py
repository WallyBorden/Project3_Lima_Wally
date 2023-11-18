import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt

# Read data from CSV file skipping the first line
data = pd.read_csv('C4_2.csv')

# Filter x-values within the range of 0 to 1 (inclusive)
filtered_data = data[(data['in s'] >= 0) & (data['in s'] <= 1)]
x_data = filtered_data['in s'].values
y_data = filtered_data['C4 in V'].values

# Define the combined model function (sinusoidal + exponential)
def model(params, x):
    # params: [amplitude_sin, frequency_sin, phase_sin, exp_const, exp_coeff]
    amp_sin, freq_sin, phase_sin, exp_const, exp_coeff = params
    return amp_sin * np.sin(2 * np.pi * freq_sin * x + phase_sin) + exp_const * np.exp(exp_coeff * x)

# Define likelihood function
def log_likelihood(params):
    y_model = model(params, x_data)
    sigma = 0.1  # Assuming some constant measurement uncertainty
    return -0.5 * np.sum((y_data - y_model)**2 / sigma**2)

# Define prior function
def log_prior(params):
    # Prior bounds: [amplitude_sin, frequency_sin, phase_sin, exp_const, exp_coeff]
    if (
        0.0 < params[0] < 10.0 and  # Amplitude of sinusoidal function
        0.1 < params[1] < 1.0 and   # Frequency of sinusoidal function
        -np.pi < params[2] < np.pi and  # Phase of sinusoidal function
        0.0 < params[3] < 10.0 and  # Exponential constant
        -1.0 < params[4] < 1.0      # Exponential coefficient
    ):
        return 0.0  # log(1)
    return -np.inf  # log(0)

# Define posterior function
def log_posterior(params):
    prior = log_prior(params)
    if not np.isfinite(prior):
        return -np.inf
    return prior + log_likelihood(params)

# Set up the MCMC sampler
n_dim = 5  # Number of parameters in the model
n_walkers = 32  # Number of walkers
n_steps = 1000  # Number of steps for each walker

# Initialize walkers
initial_guess = [1.0, 0.5, 0.0, 1.0, -0.5]  # Initial guess for parameters
starting_positions = [initial_guess + 1e-4 * np.random.randn(n_dim) for _ in range(n_walkers)]

# Create the sampler
sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior)

# Run the sampler
sampler.run_mcmc(starting_positions, n_steps, progress=True)

# Extract the samples
samples = sampler.get_chain(flat=True)

# Extract the best-fit parameters from the samples (for example, using median)
best_params = np.median(samples, axis=0)

# Generate the best-fit curve using the best-fit parameters and x values
best_fit_curve = model(best_params, x_data)

# Plot the data and the best-fit curve using original x values
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, best_fit_curve, color='red', label='Best Fit (Sinusoidal + Exponential)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Fit to Data (Sinusoidal + Exponential Model)')
plt.grid(True)
plt.show()

