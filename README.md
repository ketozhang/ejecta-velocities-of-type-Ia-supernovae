All analysis with the paper is organized in a Jupyter notebook. Please open the [report.ipynb](report.ipynb) on GitHub or your own Jupyter instance. All other useful files are described below.

* `report.ipynb`
	Full report and code summarized as a Jupyter notebook.
* `models.py`
	Maximum likelihood fitting of data with Normal family likelihood functions. Running the file outputs the results as model paramters.
* `grid_search.py`
	Best parameter estimate of Two Asymmetry Models by grid search method.
* `dataloader.py`
	Use of DataFrame to import raw data and data cleaning. Running the file outputs the size comparison of each dataset.
* `data/`
	Data directory. Due to size limitations, KAEPORA database is purposely absent. `kaepora_v1.0.db` was used and can be downloaded at [https://msiebert1.github.io/kaepora/](https://msiebert1.github.io/kaepora/).
* `results/`
	Output directory for many programs above in additon to all plots and figures in EPS format.

## Supplementary
Other content that  may or may not be discussed in the paper that is considered not too crucial for the appendix.

* `selection_bias_analysis.ipynb`
* `binned_fitting.ipynb`
* `bayes_decision_analysis.ipynb`
	Develops Bayes decision rule of assigning group I and group II SNe Ia to the data.