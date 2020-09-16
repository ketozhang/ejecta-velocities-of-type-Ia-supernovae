* **`all_scores*.pkl`**
  * `params`: List of parameter dictionary where each dict has has keys `"delta_v"`, `"theta"`
  * `ks`: Kolmogorov-Smirnov score
  * `pvalue`: p-value of the associated KS score
* **`scores*.pkl`**
  Only contains top 10 KS performing scores along with the sampleed Si II velocity distribution used.
  * `v_sim`: Simulated SNe Ia Si II velocity sampled from a Gaussian distribution fitted on the empirical velocities as described in the paper
  * `params`, `ks`, `pvalue` as described above
* **`*bimodal_params.csv`**
  Stores the Gaussian mixture parameters (mean1, sd1, mean2, sd2, p) for maximum likelihood fitting and minimum chi^2 fitting.
* **`*unmodal_params.csv`**
  Stores the Gaussian parameters (mean1, sd1) for maximum likelihood fitting.