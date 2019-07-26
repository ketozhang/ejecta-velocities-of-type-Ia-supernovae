import time
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm, ks_2samp
from sklearn.model_selection import ParameterGrid
from dataloader import import_kaepora

BIMODAL_PARAMS = np.loadtxt("bimodal_params.csv", delimiter=",")


def model(lv_dist, theta, delta_v, **kwargs):
    size = kwargs.pop("size", 10000)
    lv = lv_dist.rvs(size)
    hv = lv + delta_v
    los = np.random.uniform(0, 180, len(lv))
    lv_cond = los > theta

    v_sim = np.choose(lv_cond, [hv, lv])
    return v_sim


def simulate(v_data, lv_dist, param_grid, **kwargs):
    param_grid = ParameterGrid(param_grid)

    scores = {"ks": [], "pvalue": [], "params": [], "v_sim": []}

    nparams = len(param_grid)
    start = time.perf_counter()
    elapsed_times = []

    for i, params in enumerate(param_grid):
        v_sim = model(lv_dist, params['theta'], params['delta_v'])
        ks, pvalue = ks_2samp(v_data, v_sim)

        # if pvalue >= 0.05:
        #     plt.hist(v_data, bins=20, density=True, alpha=0.5, label='data')
        #     plt.hist(v_sim, bins=20, density=True, alpha=0.5, label='simulation')
        #     plt.title(f"{params}\n{ks} {pvalue}")
        #     plt.legend()
        #     plt.show()

        if i < 100:
            scores["ks"].append(ks)
            scores["pvalue"].append(pvalue)
            scores["params"].append(params)
            scores["v_sim"].append(v_sim)
        else:
            lowest_idx = np.argmin(scores["pvalue"])
            if ks < scores["ks"][lowest_idx]:
                scores["ks"][lowest_idx] = ks
                scores["pvalue"][lowest_idx] = pvalue
                scores["params"][lowest_idx] = params
                scores["v_sim"][lowest_idx] = v_sim

        end = time.perf_counter()
        elapsed_times.append(end - start)
        mean_step_period = np.mean(elapsed_times)
        if (i % (nparams//100) == 0 or i > nparams-10):
            print(f"{params} -> {ks, pvalue}")
            print(f"Step [{i}/{nparams}] ", end="")
            print(f"ETA: {(nparams - i) * mean_step_period:.2f} secs")
        start = time.perf_counter()

    scores = {k: np.array(v) for k, v in scores.items()}
    return scores


if __name__ == "__main__":
    v_data = import_kaepora()["v_siII"]
    lv_params = BIMODAL_PARAMS[:2]
    lv_dist = norm(*lv_params)

    param_grid = {
        "theta": np.arange(0, 181, 5),
        "delta_v": np.arange(500, 20501, 500),
    }

    scores = simulate(v_data, lv_dist, param_grid, sample_size=int(1e5))

    with open("scores.pkl", "wb") as f:
        # pkl.dump({k: v[:10] for k, v in scores.items()}, f)
        pkl.dump(scores, f)
