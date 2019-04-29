import time
import numpy as np
import pickle as pkl
from scipy.stats import norm, ks_2samp
from sklearn.model_selection import ParameterGrid
from dataloader import import_sn_data

param_grid = {
    'theta': np.linspace(0, 180, 180),
    'mu_HV': np.linspace(1.3, 5, 10),
    'sigma_HV': np.linspace(0.1, 0.3, 5),
    'mu_LV': np.linspace(1., 1.3, 10),
    'sigma_LV': np.linspace(0.1, 0.5, 5),
}


# def simulate_slow():
#     vs = []
#     ks = []

#     for theta in param_grid[0]:
#         v = []
#         for i in range(N):
#             los = np.random.uniform(-np.pi, np.pi)
#             if -theta/2 < los < theta/2:
#                 v.append(sample_hv())
#             else:
#                 v.append(sample_lv())

#         v = np.array(v).flatten()
#         vs.append(v)
#         ks.append(ks_2samp(x, v))


# def simulate(param_grid, sample_size=1000):
#     vs = []
#     ks = []
#     for theta in param_grid[0]:
#         los = np.random.uniform(-np.pi, np.pi, sample_size)

#         cond = (-theta/2 < los) & (-theta/2 < los)
#         hv_size = np.sum(cond)
#         lv_size = np.sum(~cond)

#         v = np.append(
#             sample_hv(hv_size),
#             sample_lv(hv_size)
#         )
#         vs.append(v)
#         ks.append(ks_2samp(x, v)[0])

#     return vs, ks


def model_vdist(theta, mu_HV, sigma_HV, mu_LV, sigma_LV, sample_size=1000):
    los = np.random.uniform(-180, 180, sample_size)

    cond = (los > -theta/2) & (los < theta/2)
    hv_size = np.sum(cond)
    lv_size = np.sum(~cond)

    vs = np.append(
        np.random.normal(mu_HV, sigma_HV, hv_size),
        np.random.normal(mu_LV, sigma_LV, lv_size)
    )
    return vs


def simulate_grid(vs_data, param_grid, **kwargs):
    param_grid = ParameterGrid(param_grid)
    scores = {
        'vs': [],
        'ks': [],
        'pvalue': [],
        'params': []
    }

    nparams = len(param_grid)
    start = time.perf_counter()
    elapsed_times = []
    for i, params in enumerate(param_grid):
        vs = model_vdist(**params, **kwargs)
        ks, pvalue = ks_2samp(vs_data, vs)

        if i < 10:
            scores['vs'].append(vs)
            scores['ks'].append(ks)
            scores['pvalue'].append(pvalue)
            scores['params'].append(params)
        else:
            min_idx = np.argmin(scores['ks'])
            if ks < scores['ks'][min_idx]:
                scores['vs'].insert(min_idx, vs)
                scores['ks'].insert(min_idx, ks)
                scores['pvalue'].insert(min_idx, pvalue)
                scores['params'].insert(min_idx, params)

        if (i != 0) and (i % (nparams / 100) == 0):
            end = time.perf_counter()
            elapsed_times.append(end-start)
            mean_step_period = np.mean(elapsed_times) / (nparams / 100)
            print(f"Step [{i}/{nparams}] ", end='')
            print(
                # f"{params} " +
                f"KS: {ks:.3f} " +
                f"p-value: {pvalue:.2f} " +
                f"Elapsed Time: {elapsed_times[-1]:.2f} secs " +
                f"ETA: {(nparams - i) * mean_step_period / 60:.2f} mins "
            )
            start = time.perf_counter()

    return {k: np.array(v) for k, v in scores.items()}


if __name__ == "__main__":
    sn_data = import_sn_data()
    scores = simulate_grid(sn_data['v_siII'], param_grid, sample_size=10000)
    print(scores)

    # ks = scores['ks']
    # sort_idx = np.argsort(ks)
    # ks = ks[sort_idx]
    # vs = scores['vs'][sort_idx]
    # params = scores['params'][sort_idx]

    with open('scores.pkl', 'wb') as f:
        # pkl.dump({k: v[:10] for k, v in scores.items()}, f)
        pkl.dump(scores, f)
