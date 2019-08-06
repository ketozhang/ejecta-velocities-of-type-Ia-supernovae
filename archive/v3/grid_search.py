import time
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm, ks_2samp
from sklearn.model_selection import ParameterGrid
from dataloader import import_kaepora

PROJECT_PATH = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_PATH / 'results'
RNG = np.random.RandomState(seed=822)

# [mu_lv, sigma_lv, mu_hv, sigma_hv, mixing_parameter]
BIMODAL_PARAMS = np.loadtxt(str(RESULTS_PATH/"bimodal_params.csv"), delimiter=",")


def model(lv_dist, theta, delta_v, **kwargs):
    sample_size = kwargs.pop('sample_size', 10000)

    # Take sample from the low velocity distribution
    lv = lv_dist.rvs(sample_size, random_state=RNG)

    # Take sample of the line of sight
    los = RNG.uniform(0, 180, len(lv))

    # hv = lv + delta_v  # constant velocity
    hv = lv + delta_v*((theta - los) / theta)  # linear velocity

    # For each data point in hv and lv:
    # Use hv's data point if line of sight is within theta otherwise use lv data point
    lv_cond = los > theta
    v_sim = np.choose(lv_cond, [hv, lv])
    return v_sim


def simulate(v_data, lv_dist, param_grid, **kwargs):
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
    # Expand the parameter grids to a list of dict elements.
    # Each element is a single set model parameters
    param_grid = ParameterGrid(param_grid)

    # The following scores will be stored for each set of parameters
    all_scores = {"ks": [], "pvalue": [], "params": []}
    scores = {"ks": [], "pvalue": [], "params": [], "v_sim": []}

    nparams = len(param_grid)
    start = time.perf_counter()
    elapsed_times = []

    for i, params in enumerate(param_grid):
        # Observe the model and return the observed (simulated) velocity data
        v_sim = model(lv_dist, params['theta'], params['delta_v'], **kwargs)

        # Compute KS values
        ks, pvalue = ks_2samp(v_data, v_sim)

        # if pvalue >= 0.05:
        #     plt.hist(v_data, bins=20, density=True, alpha=0.5, label='data')
        #     plt.hist(v_sim, bins=20, density=True, alpha=0.5, label='simulation')
        #     plt.title(f"{params}\n{ks} {pvalue}")
        #     plt.legend()
        #     plt.show()

        # Keeping only the top 100 models chosen by their KS score
        all_scores["ks"].append(ks)
        all_scores["pvalue"].append(pvalue)
        all_scores["params"].append(params)
        if i < 100:
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
        if (i % (nparams//100) == 0 or i > nparams-10):
            print(f"{params} -> {ks, pvalue}")
            print(f"Step [{i}/{nparams}] ", end="")
            print(f"ETA: {(nparams - i) * mean_step_period:.2f} secs")
        start = time.perf_counter()

    # Conver score from Python list to numpy array
    scores = {k: np.array(v) for k, v in scores.items()}
    all_scores = {k: np.array(v) for k, v in scores.items()}
    return scores, all_scores


if __name__ == "__main__":
    # Velocity data
    v_data = import_kaepora()["v_siII"]

    # Create Gaussian distribution for low velocity
    lv_params = BIMODAL_PARAMS[:2]
    lv_dist = norm(*lv_params)

    # Set the range for each parameter in the parameter grid
    param_grid = {
        "theta": np.arange(0, 181, 5),
        "delta_v": np.arange(3000, 7501, 500),
    }

    # Start simulation and save scores
    scores, all_scores = simulate(v_data, lv_dist, param_grid, sample_size=int(1e5))
    with open(RESULTS_PATH/"scores.pkl", "wb") as f:
        # pkl.dump({k: v[:10] for k, v in scores.items()}, f)
        pkl.dump(scores, f)
    with open(RESULTS_PATH/"all_scores.pkl", "wb") as f:
        pkl.dump(all_scores, f)
