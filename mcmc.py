import time
from pathlib import Path
import emcee
import numpy as np
from scipy.stats import norm, ks_2samp
from models import gaussian, bimodal_gaussian
from dataloader import import_kaepora

PROJECT_PATH = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_PATH / 'results'
BIMODAL_PARAMS = np.loadtxt(str(RESULTS_PATH/"bimodal_params.csv"), delimiter=",")


def model(lv_dist, theta, delta_v, size=10000):
    # Take sample from the low velocity distribution
    lv = lv_dist.rvs(size)

    # Take sample of the line of sight
    los = np.random.uniform(0, 180, len(lv))

    # hv = lv + delta_v  # constant velocity
    hv = lv + delta_v * ((theta - los) / theta)  # linear velocity

    # For each data point in hv and lv:
    # Use hv's data point if line of sight is within theta otherwise use lv data point
    lv_cond = los > theta
    v_sim = np.choose(lv_cond, [hv, lv])
    return v_sim


def lnlike(v_data, v_sim, scatter):
    normalization = np.log(2 * np.pi * scatter**2)
    exponential = (ks_2samp(v_data, v_sim)[0] / scatter)**2
    return -0.5 * (exponential + normalization)
    # return ks_2samp(v_data, v_sim)[1]


def lnprior(theta, delta_v):
    invalid = ~(theta >= 0 and theta <= 80 and delta_v >= 0 and delta_v <= 6000)
    if invalid:
        return -np.inf

    lp = 0

    # lv_mu, lv_sigma = BIMODAL_PARAMS[:2]
    # hv_mu, hv_sigma = BIMODAL_PARAMS[2:4]
    # delta_v_dist = norm(hv_mu - lv_mu, np.sqrt(hv_sigma ** 2 + lv_sigma ** 2))
    # lp += delta_v_dist.logpdf(delta_v)

    return lp


def lprob(params, v_data, lv_dist, **kwargs):
    """
    Parameters
    ----------
    params : [type]
        theta -- HV ejecta spread
        delta_v -- (HV - LV)
        scatter -- intrinsic scatter
    data : array-like
    lv_dist : scipy.stats.rv_continuous
    """
    lp = lnprior(*params[:-1])
    if not np.isfinite(lp):
        return -np.inf
    v_sim = model(lv_dist, *params[:-1])
    lnl = lnlike(v_data, v_sim, params[-1])
    return lnl + lp


def simulate(v_data, lv_dist, **kwargs):
    steps = kwargs.pop("steps", 100)
    ndim = kwargs.pop("ndim", 2+1)
    nwalkers = kwargs.pop("nwalkers", 10)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lprob, args=[v_data, lv_dist], kwargs=kwargs
    )
    pos = np.array(
        [
            np.random.normal(45, 45 / 3, nwalkers),  # theta
            np.random.normal(3000, 3000 / 3, nwalkers),
            np.random.normal(500, 500/3, nwalkers)
        ]
    ).T
    sampler.run_mcmc(pos, steps)
    return sampler.chain


if __name__ == "__main__":
    start = time.perf_counter()
    v_data = import_kaepora()["v_siII"]
    lv_params = BIMODAL_PARAMS[:2]
    lv_dist = norm(*lv_params)
    kwargs = dict(
        nwalkers=20,
        steps=10000
    )
    chain = simulate(v_data, lv_dist, **kwargs)
    np.save("chain.npy", chain)
    end = time.perf_counter()
    print(f"\nMCMC Finished ({(end-start)/60:.1f}m)\nnwalkers={kwargs['nwalkers']}, steps={kwargs['steps']}")
