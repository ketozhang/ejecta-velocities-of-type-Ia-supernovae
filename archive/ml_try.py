import pandas as pd
import numpy as np
import math
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

sns.set(color_codes=True, style="whitegrid")

##read in Velocity
sn = pd.read_csv(
    "snV.txt", delim_whitespace=True, names=["SN", "v", "verr"], comment="#"
)
velocity = sn["v"] / 1000.0

initial_guess1 = [11.1, 3.0]


def f1(params):
    # print(params)  # <-- you'll see that params is a NumPy array
    m1, s1 = (
        params
    )  # <-- for readability you may wish to assign names to the component variables
    totalnumber = len(velocity)
    # probtmp=1./(math.sqrt(2.*math.pi)*s1)*np.exp(-np.power((velocity - m1)/s1, 2.)/2)
    probtmp = (
        totalnumber
        * 1.0
        / (math.sqrt(2.0 * math.pi) * s1)
        * np.exp(-np.power((velocity - m1) / s1, 2.0) / 2)
    )
    # print(probtmp)
    restmp = sum(np.log(probtmp))
    # print(restmp)
    return -restmp


initial_guess2 = [200.98811799, 10.99968538, 0.69310179,12.31243812, 1.77763118]


def f2(params):
    # print(params)  # <-- you'll see that params is a NumPy array
    n1, m1, s1, m2, s2 = (
        params
    )  # <-- for readability you may wish to assign names to the component variables
    totalnumber = len(velocity)
    probtmp = n1 * 1.0 / (math.sqrt(2.0 * math.pi) * s1) * np.exp(
        -np.power((velocity - m1) / s1, 2.0) / 2
    ) + (totalnumber - n1) * 1.0 / (math.sqrt(2.0 * math.pi) * s2) * np.exp(
        -np.power((velocity - m2) / s2, 2.0) / 2
    )
    # print(probtmp)
    restmp = sum(np.log(probtmp))
    # print(restmp)
    return -restmp


initial_guess3 = [200.0, 11.0, 1.0, 50, 12.0, 1.0, 15.0, 1.0]


def f3(params, normalized=True):
    # print(params)  # <-- you'll see that params is a NumPy array
    n1, m1, s1, n2, m2, s2, m3, s3 = (
        params
    )  # <-- for readability you may wish to assign names to the component variables
    totalnumber = 1 if normalized else len(velocity)
    probtmp = (
        n1
        * 1.0
        / (math.sqrt(2.0 * math.pi) * s1)
        * np.exp(-np.power((velocity - m1) / s1, 2.0) / 2)
        + n2
        * 1.0
        / (math.sqrt(2.0 * math.pi) * s2)
        * np.exp(-np.power((velocity - m2) / s2, 2.0) / 2)
        + (totalnumber - n1 - n2)
        * 1.0
        / (math.sqrt(2.0 * math.pi) * s3)
        * np.exp(-np.power((velocity - m3) / s3, 2.0) / 2)
    )
    restmp = sum(np.log(probtmp))
    # print(params,restmp)
    return -restmp


if __name__ == "__main__":
    print("Velocity in unit of 10^3 km/s")
    result1 = optimize.minimize(f1, initial_guess1, method="Nelder-Mead")
    if result1.success:
        fitted_params = result1.x
        ml = -result1.fun
        print(
            "ML : ",
            ml,
            " Mean1:",
            fitted_params[0],
            " Sigma1:",
            fitted_params[1],
        )
    else:
        raise ValueError(result1.message)

    result2 = optimize.minimize(f2, initial_guess2, method="Nelder-Mead")
    if result2.success:
        fitted_params = result2.x
        ml = -result2.fun
        print(
            "ML : ",
            ml,
            " Mean1:",
            fitted_params[1],
            " Sigma1:",
            fitted_params[2],
            " Mean2:",
            fitted_params[3],
            " Sigma2:",
            fitted_params[4],
        )
    else:
        raise ValueError(result2.message)

    # initial_guess3 = [200.0 / len(velocity), 11.0, 1.0, 50 / len(velocity), 12.0, 1.0, 15.0, 1.0]
    # result3 = optimize.minimize(f3, initial_guess3, method="Nelder-Mead")
    # if result3.success:
    #     fitted_params = result3.x
    #     ml = 1 / f3([fitted_params[0] * len(velocity), fitted_params[1], fitted_params[2], fitted_params[3] * len(velocity), fitted_params[4], fitted_params[5], fitted_params[6], fitted_params[7]], normalized=False)
    #     print(
    #         "ML : ", ml,
    #         " Mean1:", fitted_params[1],
    #         " Sigma1:", fitted_params[2],
    #         " Mean2:", fitted_params[4],
    #         " Sigma2:", fitted_params[5],
    #         " Mean3:", fitted_params[6],
    #         " Sigma3:", fitted_params[7],
    #     )
    # else:
    #     raise ValueError(result3)

    value, bins, _ = plt.hist(velocity, bins=20, density=True, color="gray")
    xrange = np.linspace(np.min(bins), np.max(bins), 100)
    g1 = fitted_params[0] / len(velocity)
    g2 = 1 - g1
    v_dist = g1 * norm.pdf(xrange, *fitted_params[1:3]) + g2 * norm.pdf(
        xrange, *fitted_params[3:5]
    )
    plt.plot(
        xrange,
        v_dist,
        color="r",
        label=f"unbinned (maximum likelihood)\n{fitted_params[0] / len(velocity):.2f} * N({fitted_params[1]:.2f}, {fitted_params[2]:.2f}) + {1 - (fitted_params[0] / len(velocity)):.2f} * N({fitted_params[3]:.2f}, {fitted_params[4]:.2f})\nloglike: {ml:.2f}",
    )

    bimodal_params = np.loadtxt("results/bimodal_params.csv", delimiter=",")
    bimodal_params[0:4] = bimodal_params[0:4] / 1e3
    g1 = bimodal_params[4]
    g2 = 1 - g1
    v_dist = g1 * norm.pdf(xrange, *bimodal_params[0:2]) + g2 * norm.pdf(
        xrange, *bimodal_params[2:4]
    )
    lnl = np.sum(np.log(
          g1 * len(velocity) * norm.pdf(velocity, *bimodal_params[0:2])
        + g2 * len(velocity) * norm.pdf(velocity, *bimodal_params[2:4])
    ))
    plt.plot(
        xrange,
        v_dist,
        color="b",
        label=f"binned (least squares)\n{bimodal_params[4]:.2f} * N({bimodal_params[0]:.2f}, {bimodal_params[1]:.2f}) + {1-bimodal_params[4]:.2f} * N({bimodal_params[2]:.2f}, {bimodal_params[3]:.2f})\nloglike: {lnl:.2f}",
    )
    plt.legend()
    plt.show()