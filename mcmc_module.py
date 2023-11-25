import numpy as np
import emcee

def log_likelihood(params, x, data, sigma=0.1):
    a, b, c, d, e, f = params
    model = model_func_emcee(x, a, b, c, d, e, f)
    return -0.5 * np.sum((data - model)**2 / sigma**2)

def model_func_emcee(x, a, b, c, d, e, f):
    return a * np.sin(b * x) * np.exp(-c * x**2) + d * np.sin(e * x + f)

def run_emcee(data, x, initial_guess, num_walkers=100, num_steps=5000):
    ndim = len(initial_guess)
    nwalkers = num_walkers

    # Initial positions of walkers
    pos = [initial_guess + 1e-4*np.random.randn(ndim) for _ in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(x, data))
    
    # Run MCMC
    sampler.run_mcmc(pos, num_steps, progress=True)

    # Extract the samples
    samples = sampler.get_chain()

    return samples

