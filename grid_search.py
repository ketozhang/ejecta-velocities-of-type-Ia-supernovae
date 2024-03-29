import time
import sys
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm, ks_2samp
from sklearn.model_selection import ParameterGrid
from dataloader import import_kaepora

PROJECT_PATH = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_PATH / "results"
VERBOSE = False

# [mu_lv, sigma_lv, mu_hv, sigma_hv, mixing_parameter]
BIMODAL_PARAMS = np.loadtxt(str(RESULTS_PATH / "bimodal_params.csv"), delimiter=",")


def model_v2(lv_dist, theta, delta_v, **kwargs):
    """Two population model --- one spherical and one asymmetric."""
    rng = np.random.RandomState(seed=822)
    sample_size = kwargs.pop("sample_size", 10000)
    g = BIMODAL_PARAMS[4]  # mixing parameter
    lv_size = int(sample_size * g)
    hv_size = sample_size - lv_size

    # Take sample from the low velocity distribution
    lv = lv_dist.rvs(hv_size, random_state=rng)

    # Take sample of the line of sight using uniform RV to spherical transformation
    u = np.random.uniform(0, 1, hv_size)
    los = np.degrees(np.arccos(2 * u - 1))

    # hv = lv + delta_v  # constant velocity
    hv = lv + delta_v * ((theta - los) / theta)  # linear velocity

    # For each data point in hv and lv,
    # use hv's data point if line of sight is within theta otherwise use lv data point
    lv_cond = los > theta
    hv = np.choose(lv_cond, [hv, lv])

    # Now take a sample from the low velocity distribution and do nothing to it
    lv = lv_dist.rvs(lv_size, random_state=rng)

    v_sim = np.hstack((lv, hv))
    return v_sim


def model_v3(lv_dist, theta, delta_v, **kwargs):
    """One population model --- all asymmetric."""
    rng = np.random.RandomState(seed=822)
    sample_size = kwargs.pop("sample_size", 10000)

    # Take sample from the low velocity distribution
    lv = lv_dist.rvs(sample_size, random_state=rng)

    # Take sample of the line of sight using uniform RV to spherical transformation
    u = np.random.uniform(0, 1, sample_size)
    los = np.degrees(np.arccos(2 * u - 1))

    # hv = lv + delta_v  # constant velocity
    hv = lv + delta_v * ((theta - los) / theta)  # linear velocity

    # For each data point in hv and lv:
    # Use hv's data point if line of sight is within theta otherwise use lv data point
    lv_cond = los > theta
    v_sim = np.choose(lv_cond, [hv, lv])
    return v_sim


def simulate(model, v_data, lv_dist, param_grid, **kwargs):
    """Simualte and return score.

    Parameters
    ----------
    v_data : array-like
        Input velocity data for KS score.
    lv_dist : scipy.stats.rv_continuous
        Gaussian distribution for low velocity. Used for sampling via `lv_dist.rvs`.
    param_grid : dict
        delta_v: Difference in velocity from velocity > theta (called lv) and velocity at 0 deg
        theta: Angle of separation from high velocity and low velocity region

    Returns
    -------
    score  : dict
        ks -- Score from KS
        pvalue -- p-value from KS
        params -- model parameters (delta_v, theta)
        v_sim -- resultant simulated velocity data
    """
    # Expand the parameter grids to a list of dict elements
    # with each element being a single set model parameters
    # [
    #    {'theta': ..., 'delta_v': ...},
    #    {'theta': ..., 'delta_v': ...},
    #    ...
    # ]
    param_grid = ParameterGrid(param_grid)

    # The following scores will be stored for each set of parameters
    scores = {"ks": [], "pvalue": [], "params": [], "v_sim": []}
    all_scores = {"ks": [], "pvalue": [], "params": []}

    nparams = len(param_grid)
    start = time.perf_counter()
    elapsed_times = []

    for i, params in enumerate(param_grid):
        # Observe the model and return the observed (simulated) velocity data
        v_sim = model(lv_dist, params["theta"], params["delta_v"], **kwargs)

        # Compute KS and p-value
        ks, pvalue = ks_2samp(v_data, v_sim)

        # if pvalue >= 0.05:
        #     plt.hist(v_data, bins=20, density=True, alpha=0.5, label='data')
        #     plt.hist(v_sim, bins=20, density=True, alpha=0.5, label='simulation')
        #     plt.title(f"{params}\n{ks} {pvalue}")
        #     plt.legend()
        #     plt.show()

        all_scores["ks"].append(ks)
        all_scores["pvalue"].append(pvalue)
        all_scores["params"].append(params)

        # For top 10 models, keep both score and sample
        if i < 10:
            scores["ks"].append(ks)
            scores["pvalue"].append(pvalue)
            scores["params"].append(params)
            scores["v_sim"].append(v_sim)
        else:
            largest_idx = np.argmax(scores["ks"])
            if ks < scores["ks"][largest_idx]:
                # Remove the largest KS score from top 100 score
                scores["ks"][largest_idx] = ks
                scores["pvalue"][largest_idx] = pvalue
                scores["params"][largest_idx] = params
                scores["v_sim"][largest_idx] = v_sim

        # Print results and ETA
        end = time.perf_counter()
        elapsed_times.append(end - start)
        mean_step_period = np.mean(elapsed_times)
        if i % (nparams // 100) == 0 or i > nparams - 10:
            if VERBOSE:
                print(f"{params} -> {ks, pvalue}")
                print(f"Step [{i}/{nparams}] ", end="")
            print(f"ETA: {(nparams - i) * mean_step_period:.2f} secs")
        start = time.perf_counter()

    # Convert score from Python list to numpy array
    scores = {k: np.array(v) for k, v in scores.items()}
    return scores, all_scores


if __name__ == "__main__":
    if "-v" in sys.argv[1:]:
        VERBOSE = True
    # Velocity data
    v_data = import_kaepora()["v_siII"]

    # Create Gaussian distribution describing low velocity group I
    lv_params = BIMODAL_PARAMS[:2]
    lv_dist = norm(*lv_params)

    # Set the range for each parameter in the parameter space
    param_grid = {
        "theta": np.arange(0, 181, 1),
        "delta_v": np.arange(3000, 7501, 100),
    }

    # Simulation v2
    # start simulation and save scores
    scores, all_scores = simulate(
        model_v2, v_data, lv_dist, param_grid, sample_size=int(1e5)
    )
    with open(RESULTS_PATH / "scores_v2.pkl", "wb") as f:
        pkl.dump(scores, f)

    with open(RESULTS_PATH / "all_scores_v2.pkl", "wb") as f:
        pkl.dump(all_scores, f)

    # Simulation v3
    # start simulation and save scores
    scores, all_scores = simulate(
        model_v3, v_data, lv_dist, param_grid, sample_size=int(1e5)
    )
    with open(RESULTS_PATH / "scores_v3.pkl", "wb") as f:
        pkl.dump(scores, f)

    with open(RESULTS_PATH / "all_scores_v3.pkl", "wb") as f:
        pkl.dump(all_scores, f)
