import emcee
import corner
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from numpy.random import uniform

# Read data from a CSV file skipping the first line
data = pd.read_csv('C4_2.csv')  # Replace 'your_data.csv' with your CSV file path

# Filter x-values within the range of 0 to 1 (inclusive)
filtered_data = data[(data['in s'] >= 0) & (data['in s'] <= 1)]
x = filtered_data['in s'].values
y = filtered_data['C4 in V'].values

order = 8

# Ensure x-values are between 0 and 1
x_normalized = (x - min(x)) / (max(x) - min(x))

def model(x, params):
    return sum([c * x ** i for i, c in enumerate(params)])

def logposterior(parameters):
    return loglikelihood(parameters) + logprior(parameters)
   
def loglikelihood(parameters):
    return -0.5 * ((y - model(x_normalized, parameters)) ** 2.0).sum()
   
def logprior(parameters):
    for p in parameters:
        if p > 20000 or p < -20000:
            return -np.inf
    return 0

# Create the emcee sampler
num_walkers = 100
num_iterations = 500

sampler = emcee.EnsembleSampler(num_walkers, order, logposterior)

# Set the initial walker positions
initial_state = np.zeros((num_walkers, order))
initial_state += uniform(-10000, 10000, size=(num_walkers, order))

# Run the MCMC for fewer iterations and walkers
state = sampler.run_mcmc(initial_state, num_iterations)
chain = sampler.get_chain()
logp = sampler.get_log_prob()

# Get the maximum posterior values from the final iteration of the MCMC chain
i = logp[-1, :].argmax()
maxp = chain[-1, i, :]
print(maxp)

plt.figure()
xref_normalized = np.arange(0, 1, 0.01)
xref = xref_normalized * (max(x) - min(x)) + min(x)
plt.scatter(x, y, label='data')
plt.plot(xref, model(xref_normalized, maxp), label='max posterior fit',
         linewidth=4.0, color='orange')
plt.legend()

_ = corner.corner(chain[-1, :, :])
plt.show()


