import time
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm, ks_2samp
from sklearn.model_selection import ParameterGrid
from dataloader import import_all_data

PROJECT_PATH = Path(__name__).resolve().parent

def model_lv_vdist(mu_LV, sigma_LV, sample_size=10000):
    """
    Return `sample_size` samples of the LV Gausssian distribution given
    by the model paramter observed at some uniform probability LOS.
    """
    return np.random.normal(mu_LV, sigma_LV, sample_size)


def observe_velocity(theta, lv, delta_v, R):
    """

    Parameters
    ----------
    theta : float or array-like
    lv : float or array-like
    delta_v : float
    R : float
    """
    upper_v = lv - (delta_v / 90)*(theta - 90)
    lower_v = lv + (delta_v / R / 90)*(theta - 90)
    upper_cond = (0 <= theta) & (theta <= 90)

    return np.choose(upper_cond, [lower_v, upper_v])



def simulate_grid(vs, param_grid, **kwargs):
    """
    For each of the `sample_size` observations:
    1. Sample lv from Gaussian distribution.
    2. Generate a supernovae of parameters delta_v and R.
    3. Get the velocity for the given line of sight (`theta`)

    Keyword Parameters
    ------------------
    sample_size : int
        Number of observations
    """
    bimodal_params_fpath = str(PROJECT_PATH / 'bimodal_params.csv')
    bimodal_params = np.genfromtxt(bimodal_params_fpath, delimiter=',')
    lv_params = bimodal_params[0:2]
    hv_params = bimodal_params[2:4]
    mixing_param = bimodal_params[4]
    param_grid = ParameterGrid(param_grid)

    scores = {
        'lv_samp': [],
        'hvs': [],
        'ks': [],
        'pvalue': [],
        'params': []
    }

    nparams = len(param_grid)
    start = time.perf_counter()
    elapsed_times = []
    lvs = model_lv_vdist(*lv_params, **kwargs)

    for i, params in enumerate(param_grid):
        thetas = np.random.uniform(0, 180, len(lvs))
        delta_v = params['delta_v']
        R = params['R']
        hvs = observe_velocity(thetas, lvs, delta_v, R)
        # hv_samp = norm(*hv_params).rvs(len(hvs))
        lv_samp = norm(*lv_params).rvs(int(len(hvs) / (1-mixing_param)))
        ks, pvalue = ks_2samp(np.append(lv_samp, hvs), vs)
        print(f"{params} -> {ks, pvalue}")
        # if pvalue > 0.10:
        #     plt.hist(vs, label="data", bins = np.arange(8, 20.5, 0.5), density=True, alpha=0.5)
        #     plt.hist(np.append(lv_samp, hvs), label="simulation", bins=np.arange(8, 20.5, 0.5), density=True, alpha=0.5)
        #     plt.legend()
        #     plt.show()

        if i < 20:
            scores['lv_samp'].append(lv_samp)
            scores['hvs'].append(hvs)
            scores['ks'].append(ks)
            scores['pvalue'].append(pvalue)
            scores['params'].append(params)
        else:
            max_idx = np.argmax(scores['ks'])
            if ks < scores['ks'][max_idx]:
                scores['lv_samp'][max_idx] = lv_samp
                scores['hvs'][max_idx]= hvs
                scores['ks'][max_idx]= ks
                scores['pvalue'][max_idx]= pvalue
                scores['params'][max_idx]= params

        end = time.perf_counter()
        elapsed_times.append(end-start)
        mean_step_period = np.mean(elapsed_times)
        print(f"Step [{i}/{nparams}] ", end='')
        print(f"ETA: {(nparams - i) * mean_step_period:.2f} secs")
        start = time.perf_counter()

    scores = {k: np.array(v) for k, v in scores.items()}
    return scores

if __name__ == "__main__":
    sn_data = import_all_data()
    param_grid = {
        'delta_v': np.arange(1,11),
        'R': np.arange(2,11)
}

    scores = simulate_grid(sn_data['v_siII'], param_grid, sample_size=int(1e4))

    with open('scores.pkl', 'wb') as f:
        # pkl.dump({k: v[:10] for k, v in scores.items()}, f)
        pkl.dump(scores, f)
