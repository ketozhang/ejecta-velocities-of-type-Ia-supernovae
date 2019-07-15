import numpy as np
from emcee import EnsembleSampler
from scipy.optimize import curve_fit
from scipy.stats import norm
from dataloader import import_sn_data

BIMODAL_PARAMS = np.loadtxt('bimodal_params.csv', delimiter=',')

def gaussian(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

def bimodal_gaussian(x, mu1=0, sigma1=1, mu2=0, sigma2=1, g=0.5):
    """Bimodal independent gaussian"""
    return g*gaussian(x, mu1, sigma1) + (1-g)*gaussian(x, mu2, sigma2)

def lnprob(params, x, v):
    """
    Arguments
    ---------
    params : array_like
         theta    - HV ejecta spread
         mu_HV    - HV mean
         sigma_HV - HV standard deviation
         mu_LV    - LV mean
         sigma_LV - LV standard deviation
    """
    lnp = lnprior(params)
    if not np.isfinite(lnp):
        return -np.inf

    lnl = lnlike(params, v)

    return lnl + lnp


def lnlike(params, v):
    scatter = params[-1]
    v_hat = model(*params[:-1])
    return -0.5*np.sum(((v-v_hat)/scatter)**2 + np.log(2*np.pi*scatter**2))


def lnprior(params):
    theta, mu_HV, sigma_HV, mu_LV, sigma_LV, scatter = params

    invalid = ~(
        (0 <= theta < 2*np.pi) and
        sigma_HV > 0 and
        sigma_LV > 0 and
        mu_HV > 0 and
        mu_LV > 0 and
        scatter > 0
    )
    if invalid:
        return -np.inf

    lnp = np.sum([
        norm.logpdf(mu_HV, BIMODAL_PARAMS[2], BIMODAL_PARAMS[3]*2),
        norm.logpdf(mu_LV, BIMODAL_PARAMS[0], BIMODAL_PARAMS[1]*2),
        norm.logpdf(theta, np.pi, 3.5)
    ])

    return lnp


def model(theta, mu_HV, sigma_HV, mu_LV, sigma_LV):
    los = np.random.uniform(-np.pi, np.pi)
    if -theta/2 < los < theta/2:
        return np.random.normal(mu_HV, sigma_HV)
    else:
        return np.random.normal(mu_HV, sigma_HV)

def lnprob(params, x, v):
    """
    Arguments
    ---------
    params : array_like
         theta    - HV ejecta spread
         mu_HV    - HV mean
         sigma_HV - HV standard deviation
         mu_LV    - LV mean
         sigma_LV - LV standard deviation
    """
    lnp = lnprior(params)
    if not np.isfinite(lnp):
        return -np.inf

    lnl = lnlike(params, v)

    return lnl + lnp


def lnlike(params, v):
    scatter = params[-1]
    v_hat = model(*params[:-1])
    return -0.5*np.sum(((v-v_hat)/scatter)**2 + np.log(2*np.pi*scatter**2))


def lnprior(params):
    theta, mu_HV, sigma_HV, mu_LV, sigma_LV, scatter = params

    invalid = ~(
        (0 <= theta < 2*np.pi) and
        sigma_HV > 0 and
        sigma_LV > 0 and
        mu_HV > 0 and
        mu_LV > 0 and
        scatter > 0
    )
    if invalid:
        return -np.inf

    lnp = np.sum([
        norm.logpdf(mu_HV, BIMODAL_PARAMS[2], BIMODAL_PARAMS[3]*2),
        norm.logpdf(mu_LV, BIMODAL_PARAMS[0], BIMODAL_PARAMS[1]*2),
        norm.logpdf(theta, np.pi, 3.5)
    ])

    return lnp


def model(theta, mu_HV, sigma_HV, mu_LV, sigma_LV):
    los = np.random.uniform(-np.pi, np.pi)
    if -theta/2 < los < theta/2:
        return np.random.normal(mu_HV, sigma_HV)
    else:
        return np.random.normal(mu_HV, sigma_HV)

if __name__ == "__main__":
    sn_data = import_sn_data()
    v = np.array(sn_data['v_siII'])

    nsteps = 10000
    ndim, nwalkers = 6, 100
    pos = [np.random.normal(param, param/2, nwalkers) for param in np.append(BIMODAL_PARAMS, 1)]
    pos = np.array(pos).T

    sampler = EnsembleSampler(nwalkers, ndim, lnprob, args=(np.arange(len(v)), v))
    sampler.run_mcmc(pos, nsteps);
    np.save('chain.npy', sampler.chain)