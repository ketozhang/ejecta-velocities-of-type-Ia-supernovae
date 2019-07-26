import emcee
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm


def gaussian(x, mu=0, sigma=1):
    """Gaussian PDF"""
    return norm.pdf(x, mu, sigma)


def bimodal_gaussian(x, mu1=0, sigma1=1, mu2=0, sigma2=1, g=0.5):
    """Bimodal independent Gaussian PDF.

    Parameters
    ----------
    mu1, sigma1 : float
        Parameters for Gaussian PDF with chance g.
    mu2, sigma2 : float
        Parameter for Gaussian PDF with chance 1-g.
    g : float
        Mixing parameter.
    """
    return g * gaussian(x, mu1, sigma1) + (1 - g) * gaussian(x, mu2, sigma2)


def fit_multimodal_gaussian(x, k, **kwargs):
    """Fit multimodal gaussian of mode k

    Parameters
    ----------
    x : array-like
        data

    Returns
    -------
    params : array-like
    loss : float
    """
    g = kwargs.pop("g", np.ones(k) * 1 / k)
    mu = kwargs.pop("mu", np.ones(k) * np.mean(x))
    sigma = kwargs.pop("sigma", np.ones(k) * np.std(x))
    params = np.concatenate([g, mu, sigma], axis=0)
    # bounds = [(None, None)] * k + [(None, None)] * k + [(0, None)] * k
    # constr = ({"type": "eq", "fun": lambda x: 1 - np.sum(x[:k])},)

    def lnprob(params, x, k):
        params = [params[:k], params[k : 2 * k], params[2 * k :]]
        lnp = lnprior(params)
        if not np.isfinite(lnp):
            return -np.inf

        lnl = lnlike(params, x)

        return lnl + lnp

    def lnlike(params, x):
        """Returns the log likelihood of gaussian(mu, sigma)"""
        g, mu, sigma = params

        lnl = 0
        for i in range(len(g)):
            mp = g[i] / np.sum(g)
            lnl += -0.5 * np.sum(
                np.log(mp)
                + (((x - mu[i]) ** 2 / sigma[i] ** 2))
                + np.log(2 * np.pi * sigma[i] ** 2)
            )
            # lnl += np.sum(np.log(g[i] / np.sum(g)) + np.log(gaussian(x, mu[i], sigma[i])))
        print(g, mu, sigma, "->", lnl)
        return lnl

    def lnprior(params):
        _, _, sigma = params

        invalid = np.any(sigma < 0)
        if invalid:
            return -np.inf
        else:
            return 0

    ndim, nwalkers = len(params), 100
    pos = np.array(
        [param + np.random.normal(param, 1, nwalkers) for param in params]
    ).T
    print(pos)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, k))
    sampler.run_mcmc(pos, 500)
    # def loss_func(params, x):
    #     """Returns the negative log likelihood of multimodal gaussian"""
    #     g = params[:k]
    #     mu = params[k : 2 * k]
    #     sigma = params[2 * k :]
    #     print(mu, sigma)

    #     loss = 0
    #     for i in range(k):
    #         loss += -(np.log(g[i]) + lnlike((mu[i], sigma[i]), x))

    #     print(g,mu,sigma,'->',loss)

    #     return loss

    # res = minimize(
    #     loss_func,
    #     x0=params,
    #     args=x,
    #     method="L-BFGS-B",
    #     # constraints=constr,
    #     bounds=bounds,
    # )
    return sampler.chain


if __name__ == "__main__":
    from dataloader import import_kaepora

    data = norm.rvs(loc=-10, scale=1, size=10000) + norm.rvs(
        loc=10, scale=1, size=10000
    )
    # data = import_kaepora()["v_siII"]
    # opt, loss = fit_multimodal_gaussian(data, 2, mu=[-1000, 1000])

    chain = fit_multimodal_gaussian(data, 2, mu=[-10, 10], sigma=[1, 1])
    # np.savetxt('modality_test.csv', sampler.chain, delimiter=',')

