from pathlib import Path
import emcee
import numpy as np
import matplotlib.pyplot as plt
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


def unimodal_fit(x, params):
    """Un-normalized unimodial Gaussian fit on some data `x`"""

    def loss(params):
        mu, sigma = params
        N = len(x)
        prob = N * gaussian(x, mu, sigma)
        lnl = sum(np.log(prob))
        return -lnl

    result = minimize(loss, params, method="Nelder-Mead")
    print(result.message)
    return result


def bimodal_fit(x, params):
    """Un-normalized bimodial Gaussian fit on some data `x`"""

    def loss(params):
        mu1, sigma1, mu2, sigma2, n1 = params
        N = len(x)
        prob1 = gaussian(x, mu1, sigma1)
        prob2 = gaussian(x, mu2, sigma2)
        prob = n1 * prob1 + (N - n1) * prob2
        lnl = sum(np.log(prob))
        return -lnl

    result = minimize(loss, params, method="Nelder-Mead")
    print(result.message)
    return result


if __name__ == "__main__":
    from dataloader import import_kaepora

    PROJECT_PATH = Path(__file__).resolve().parent
    RESULTS_PATH = PROJECT_PATH / "results"

    # Replace data with this for a sanity check
    # data = np.hstack(
    #     (
    #         norm.rvs(loc=10000, scale=700, size=10000),
    #         norm.rvs(loc=15000, scale=200, size=10000),
    #     )
    # )
    data = import_kaepora()["v_siII"]

    print("Unimodal")
    result = unimodal_fit(np.array(data), [np.mean(data), np.std(data)])
    fitted_params = result.x
    print(-result.fun, fitted_params)
    if result.success:
        np.savetxt(str(RESULTS_PATH / "unimodal_params.csv"), result.x, delimiter=",")

    print("\nBimodal")
    result = bimodal_fit(np.array(data), [11000, 700, 14000, 1200, 200])
    fitted_params = result.x
    fitted_params[4] = fitted_params[4] / len(data)
    print(-result.fun, fitted_params)
    if result.success:
        np.savetxt(str(RESULTS_PATH / "bimodal_params.csv"), result.x, delimiter=",")
