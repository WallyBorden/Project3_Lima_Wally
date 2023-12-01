import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
import pkg_resources
import os

# Get the directory path of the current working directory
current_dir = os.getcwd()

# Construct the file path for your data file
file_path = os.path.join(current_dir, 'generated_data_sp.txt')
x, data = np.loadtxt(file_path, unpack=True, skiprows=1)

# Define the model function
def model_func(x, a, b, c, d, e, f):
    return a * np.sin(b * x) * np.exp(-c * x**2) + d * np.sin(e * x + f)

# Define the log likelihood function
def log_likelihood(data, x, params, sigma=0.1):
    a, b, c, d, e, f = params
    model = model_func(x, a, b, c, d, e, f)
    return -0.5 * np.sum((data - model)**2 / sigma**2)

def log_likelihood_emcee(params, x, data, sigma=0.1):
    a, b, c, d, e, f = params
    model = model_func(x, a, b, c, d, e, f)
    return -0.5 * np.sum((data - model)**2 / sigma**2)    

def run_emcee(data, x, initial_guess, num_walkers=100, num_steps=5000):
    ndim = len(initial_guess)
    nwalkers = num_walkers

    # Initial positions of walkers
    pos = [initial_guess + 1e-4*np.random.randn(ndim) for _ in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_emcee, args=(x, data))
    
    # Run MCMC
    sampler.run_mcmc(pos, num_steps, progress=True)

    # Extract the samples
    samples = sampler.get_chain()

    return samples

# Metropolis-Hastings MCMC algorithm
def metropolis_hastings(data, x, initial_params, proposal_width=0.1, num_samples=10000):
    current_params = initial_params
    samples = [current_params]
    for i in range(num_samples):
        proposed_params = current_params + proposal_width * np.random.randn(len(initial_params))
        log_like_current = log_likelihood(data, x, current_params)
        log_like_proposed = log_likelihood(data, x, proposed_params)
        acceptance_ratio = min(1, np.exp(log_like_proposed - log_like_current))
        if acceptance_ratio >= 1 or np.random.rand() < acceptance_ratio:
            current_params = proposed_params
        samples.append(current_params)
    return np.array(samples)

# Run MCMC 
initial_guess = [3, 6, 12, 0.6, 22, np.pi/2.5]  # Starting guess for parameters
samples = metropolis_hastings(data, x, initial_guess, proposal_width=0.1, num_samples=5000)

# Run MCMC using emcee
samples_emcee = run_emcee(data, x, initial_guess, num_walkers=100, num_steps=5000)
final_params_emcee = samples_emcee[-1][-1]

#print(final_params_emcee)

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(x, data, label='Data', alpha=0.7)
plt.plot(x, model_func(x, *initial_guess), label='Initial Guess', linestyle='--', color='blue', linewidth=2)
plt.plot(x, model_func(x, *samples[-1]), label='MCMC Fit', linestyle=':', color='red', linewidth=2)
plt.plot(x, model_func(x, *final_params_emcee), label='MCMC Fit (emcee)', linestyle=':', color='green', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('MCMC Fit to Data')
#plt.savefig("datafit.png")
plt.show()

# Plot Markov chain
plt.figure(figsize=(8, 8))
for i in range(len(initial_guess)):
    plt.subplot(len(initial_guess), 1, i+1)
    plt.plot(samples[:, i])
    plt.ylabel(f'Param {i+1}')
plt.xlabel('Iteration')
plt.tight_layout()
#plt.savefig("markov_chain.png")
plt.show()
# Create corner plot
labels = [f'Param {i+1}' for i in range(len(initial_guess))]
corner.corner(samples, labels=labels, truths=initial_guess, show_titles=True)
#plt.savefig("corner_plot.png")
plt.show()



