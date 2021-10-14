[![arXiv](https://img.shields.io/badge/ads-2020MNRAS.499.5325Z-blue)](https://ui.adsabs.harvard.edu/#abs/2020MNRAS.499.5325Z/abstract)
[![DOI](https://zenodo.org/badge/183695689.svg)](https://zenodo.org/badge/latestdoi/183695689)

 # Distribution of Si II λ6355 velocities of Type Ia supernovae and implications for asymmetric explosions 
  Keto D. Zhang, WeiKang Zheng Thomas de Jaeger, Benjamin E. Stahl, Thomas G. Brink, Xuhui Han, Daniel Kasen, Ken J. Shen, Kevin Tang, Alexei V. Filippenko
> The ejecta velocity is a very important parameter in studying the structure and properties of Type Ia supernovae (SNe Ia) and is a candidate key parameter in improving the utility of SNe Ia for cosmological distance determinations. Here, we study the velocity distribution of a sample of 311 SNe Ia from the kaepora data base. The velocities are derived from the Si II λ6355 absorption line in optical spectra measured at (or extrapolated to) the time of peak brightness. We statistically show that the observed velocity has a bimodal Gaussian distribution (population ratio 201:110 or 65 per cent:35 per cent) consisting of two groups of SNe Ia: Group I with a lower but narrower scatter ( 11000±700kms−1 ), and Group II with a higher but broader scatter ( 12300±1800kms−1 ). The true origin of the two components is unknown. Naturally, there could exist two intrinsic velocity distributions observed. However, we try to use asymmetric geometric models through statistical simulations to reproduce the observed distribution assuming that all SNe Ia share the same intrinsic distribution. In the two cases we consider, 35 per cent of SNe Ia are considered to be asymmetric in Case 1, and all SNe Ia are asymmetric in Case 2. Simulations for both cases can reproduce the observed velocity distribution but require a significantly large portion ( >35 per cent ) of SNe Ia to be asymmetric. In addition, the Case 1 result is consistent with recent SNe Ia polarization observations that higher Si II λ6355 velocities tend to be more polarized. 

## Repository
All analysis with the paper is organized in a Jupyter notebook. Please open the [report.ipynb](report.ipynb) on GitHub or your own Jupyter instance. All other useful files are described below.

- `report.ipynb`
  Full report and code summarized as a Jupyter notebook.
- `models.py`
  Maximum likelihood fitting of data with Normal family likelihood functions. Running the file outputs the results as model paramters.
- `grid_search.py`
  Best parameter estimate of Two Asymmetry Models by grid search method.
- `dataloader.py`
  Use of DataFrame to import raw data and data cleaning. Running the file outputs the size comparison of each dataset.
- `data/`
  Data directory. Due to size limitations, KAEPORA database is purposely absent. `kaepora_v1.0.db` was used and can be downloaded at [https://msiebert1.github.io/kaepora/](https://msiebert1.github.io/kaepora/).
- `results/`
  Output directory for many programs above in additon to all plots and figures in EPS format.

## Supplementary

Other content that may or may not be discussed in the paper that is considered not too crucial for the appendix.

- `selection_bias_analysis.ipynb`
- `binned_fitting.ipynb`
- `bayes_decision_analysis.ipynb`
  Develops Bayes decision rule of assigning group I and group II SNe Ia to the data.
