import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns  # >= 0.9
from dataloader import import_kaepora
import pickle as pkl

# Plotting Configurations (Optional)
sns.set(context='talk', style='whitegrid', color_codes=True)
mpl.rcParams['figure.figsize'] = (10,6)
mpl.rcParams['legend.fontsize'] = 18


sn_data = import_kaepora()

with open('results/all_scores.pkl', 'rb') as f:
    scores = pkl.load(f)

ks = scores['ks']
sort_idx = np.argsort(ks)

pvalues = scores['pvalue']
pvalues = pvalues[sort_idx]
ks = ks[sort_idx]

params = scores['params'][sort_idx]
v_sim = scores['v_sim'][sort_idx]

results = pd.DataFrame([
         ks,
         pvalues
     ], index=['ks', 'pvalues']).T

results['theta'] = [param['theta'] for param in params]
results['delta_v'] = [param['delta_v'] for param in params]

def plot_histogram(nrows=2, ncols=2):
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 4.5*nrows))


    for i in range(nrows*ncols):
        ax = axs[i//ncols, i%ncols]
        bins = np.arange(7000, 17000, 500)
        param = params[i]

        # Data
        ax.hist(sn_data['v_siII'], bins, density=True, alpha=0.5, label='Data')
        ax.hist(v_sim[i], bins, density=True, alpha=0.5, label="Simulation")

        # Plot config
        ax.set_title(f"$\\theta$: {param['theta']} $\Delta v: ${param['delta_v']}\nks: {ks[i]:.1e} p: {pvalues[i]:.2f}")
        ax.set_xlim(bins.min(), bins.max())
        ax.set_yticklabels([f'{tick:.2f}' for tick in ax.get_yticks()*500])
        ax.set_xlabel('Si II Velocity (km/s)')
        ax.set_ylabel('Proportion (% of data)')
        ax.legend(loc='upper right')

    plt.tight_layout()
    # plt.savefig('results/grid_search_results.png', dpi=600, format='png')
    plt.show()


def plot_cumulative(nrows=2, ncols=2):
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 4.5*nrows))

    for i in range(nrows*ncols):
        ax = axs[i//ncols, i%ncols]

        param=params[i]

        ax.plot(np.sort(sn_data['v_siII']), np.arange(len(sn_data))/len(sn_data), label='Data', lw=3)
        ax.plot(np.sort(v_sim[i]), np.arange(len(v_sim[i]))/len(v_sim[i]), label='Simulation', lw=3)

        # Plot config
        ax.set_title(f"$\\theta$: {param['theta']} $\Delta v: ${param['delta_v']}\nks: {ks[i]:.1e} p: {pvalues[i]:.2f}")
        ax.set_xticks(np.arange(7000, 20000, 2000))
        ax.set_xlabel('Si II Velocity (km/s)')
        ax.set_ylabel('Proportion')
        ax.legend(loc='lower right')

    plt.tight_layout()
    # plt.savefig('results/grid_search_cumulative_results.png', dpi=600, format='png')
    plt.show()

def plot_constant_delta_v(delta_v=4000):
    _ = results.loc[results['delta_v'] == delta_v, :]
    plt.scatter(_['theta'], _['pvalues'])
    plt.xlabel(r'$\theta$ [deg]')
    plt.tight_layout()
    plt.show()


def plot_constant_theta(theta=120):
    _ = results.loc[results['theta'] == theta, :]
    plt.scatter(_['delta_v'], _['pvalues'])
    plt.xlabel(r'$\Delta v$ [km/s]')
    plt.tight_layout()
    plt.show()