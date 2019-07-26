import emcee
import numpy as np
from scipy.stats import norm, ks_2samp
from models import gaussian, bimodal_gaussian
from dataloader import import_kaepora

BIMODAL_PARAMS = np.loadtxt("bimodal_params.csv", delimiter=",")


def model(lv_dist, theta, delta_v, **kwargs):
    size_ratio = BIMODAL_PARAMS[4] / (1-BIMODAL_PARAMS[4])
    hv_size = kwargs.pop("hv_size", 10000)
    lv_size = int(hv_size * size_ratio)

    lv = lv_dist.rvs(lv_size)

    hv = lv_dist.rvs(hv_size)
    los = np.random.uniform(0, 180, len(hv))
    lv_cond = los > theta
    hv = np.choose(lv_cond, [hv + delta_v, hv])
    return np.hstack((lv, hv))


def lnlike(v_data, v_sim):
    return ks_2samp(v_data, v_sim)[1]


def lnprior(theta, delta_v):
    invalid = ~(theta >= 0 and theta <= 180 and delta_v >= 0)
    if invalid:
        return -np.inf

    lp = 0

    lv_mu, lv_sigma = BIMODAL_PARAMS[:2]
    hv_mu, hv_sigma = BIMODAL_PARAMS[2:4]
    delta_v_dist = norm(hv_mu - lv_mu, np.sqrt(hv_sigma ** 2 + lv_sigma ** 2))
    lp += delta_v_dist.logpdf(delta_v)

    return lp


def lprob(params, v_data, lv_dist, **kwargs):
    """
    Parameters
    ----------
    params : [type]
        theta -- HV ejecta spread
        delta_v -- (HV - LV)
    data : array-like
    lv_dist : scipy.stats.rv_continuous
    """
    lp = lnprior(*params)
    if not np.isfinite(lp):
        return -np.inf
    v_sim = model(lv_dist, *params)
    lnl = lnlike(v_data, v_sim)

    return lnl + lp


def simulate(v_data, lv_dist, **kwargs):
    steps = kwargs.pop("steps", 100)
    ndim = kwargs.pop("ndim", 2)
    nwalkers = kwargs.pop("nwalkers", 10)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lprob, args=[v_data, lv_dist], kwargs=kwargs
    )
    pos = np.array(
        [
            np.random.normal(45, 90 / 3, nwalkers),  # theta
            np.random.normal(5000, 5000 / 3, nwalkers),
        ]
    ).T
    sampler.run_mcmc(pos, steps)
    return sampler.chain


if __name__ == "__main__":
    v_data = import_kaepora()["v_siII"]
    lv_params = BIMODAL_PARAMS[:2]
    lv_dist = norm(*lv_params)

    chain = simulate(v_data, lv_dist, nwalkers=20, steps=2000)
    np.save("chain.npy", chain)
