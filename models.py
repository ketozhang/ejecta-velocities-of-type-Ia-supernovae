import numpy as np
from scipy.optimize import curve_fit
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
    return g*gaussian(x, mu1, sigma1) + (1-g)*gaussian(x, mu2, sigma2)