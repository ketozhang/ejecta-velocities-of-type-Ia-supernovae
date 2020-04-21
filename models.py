from pathlib import Path
import emcee
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm


def gaussian(x, mu=0, sigma=1, N=1):
    """Gaussian PDF

    Args:
        x (float or array_like): The x value to be evaluated.
        mu (float, optional): Expected value. Defaults to 0.
        sigma (float, optional): Standard deviation. Defaults to 1.
        N (int, optional): Area under the curve. If `N=1`, the function is a true probability density function. If `N>1`, the function is a number density function. Defaults to 1.

    Returns:
        float: The density at x. The density is either a probability density or number density depending on `N`.
    """
    return N * norm.pdf(x, mu, sigma)


def gaussian_cdf(x, *args, **kwargs):
    return norm.cdf(x, *args, **kwargs)


def bimodal_gaussian(x, mu1=0, sigma1=1, mu2=0, sigma2=1, g=0.5, N=1):
    """Bimodal independent Gaussian PDF.

    Parameters
    ----------
    mu1, sigma1 : float
        Parameters for Gaussian PDF.
    mu2, sigma2 : float
        Parameter for Gaussian PDF.
    g : float
        Mixing parameter. If `N=1`, `g` is bound by [0, 1]. If `N>1`, `g` is bound by [0, N].
    N (int, optional): Area under the curve. If `N=1`, the function is a true probability density function. If `N>1`, the function is a number density function. Defaults to 1.

    Returns:
        float: The density at x. The density is either a probability density or number density depending on `N`.
    """
    return g * gaussian(x, mu1, sigma1) + (N - g) * gaussian(x, mu2, sigma2)


def bimodal_gaussian_cdf(x, mu1=0, sigma1=1, mu2=0, sigma2=1, g=0.5, N=1):
    return g * norm.cdf(x, mu1, sigma1) + (N - g) * norm.cdf(x, mu2, sigma2)


def unimodal_fit(x, guess_params):
    """Un-normalized unimodial Gaussian fit on some data `x`"""

    def loss(params):
        mu, sigma = params
        N = len(x)
        prob = N * gaussian(x, mu, sigma)
        lnl = sum(np.log(prob))
        return -lnl

    result = minimize(loss, guess_params, method="Nelder-Mead")
    print(result.message)
    return result


def bimodal_fit(x, guess_params):
    """Un-normalized bimodial Gaussian fit on some data `x`"""

    def loss(params):
        mu1, sigma1, mu2, sigma2, n1 = params
        N = len(x)
        prob1 = gaussian(x, mu1, sigma1)
        prob2 = gaussian(x, mu2, sigma2)
        prob = n1 * prob1 + (N - n1) * prob2
        lnl = sum(np.log(prob))
        return -lnl

    result = minimize(loss, guess_params, method="Nelder-Mead")
    print(result.message)
    return result


def binned_fit(x, guess_params, pdf, bins=30, **kwargs):
    # Bin data
    N = len(x)
    y, bins = np.histogram(x, bins)
    bin_width = np.mean(bins[1:] - bins[:-1])
    bin_midpoints = (bins[1:] + bins[:-1]) / 2

    # Predict and calculate loss
    def loss(params):
        """L2 loss"""
        y_pred = bin_width * pdf(bin_midpoints, *params)
        return np.sum((y - y_pred) ** 2)

    result = minimize(loss, guess_params, method="Nelder-Mead", **kwargs)
    print(result.message)
    return result


if __name__ == "__main__":
    from dataloader import import_kaepora

    PROJECT_PATH = Path(__file__).resolve().parent
    RESULTS_PATH = PROJECT_PATH / "results"
    np.random.seed(822)

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

    print("\nBinned Unimodal")
    bins = np.arange(7000, 17001, 500)
    result = binned_fit(
        np.array(data),
        [np.mean(data), np.std(data)],
        bins=bins,
        pdf=lambda *args: gaussian(*args, N=len(data)),
    )
    fitted_params = result.x
    print(result.fun, fitted_params)
    if result.success:
        np.savetxt(
            str(RESULTS_PATH / "binned_unimodal_params.csv"), result.x, delimiter=","
        )

    print("\nBinned Bimodal")
    bins = np.arange(7000, 16001, 1000)
    result = binned_fit(
        np.array(data),
        [11000, 700, 14000, 1200, 200],
        bins=bins,
        pdf=lambda *args: bimodal_gaussian(*args, N=len(data)),
    )
    fitted_params = result.x
    fitted_params[4] = fitted_params[4] / len(data)
    print(result.fun, fitted_params)
    if result.success:
        np.savetxt(
            str(RESULTS_PATH / "binned_bimodal_params.csv"), result.x, delimiter=","
        )
