#%%
"""
Checks of the model with T_d free against the paper's.
======================================================

    python check_model.py

For every mask, mode and dataset, at the paper's maximum-posterior sample
(results/wishart_hmdm/pl_wishart_*.npz, or results/wishart/ with the
simulation noise) with T_d = 19.6 K:

  1. the model matrix equals wishart_likelihood.build_model_matrix (largest
     relative difference; the colour-correction surface differs from the
     paper's polynomial by < 1e-7);
  2. the deviance equals wishart_likelihood.wishart_deviance, and the paper's
     chi2_red is recovered;
  3. the time of one evaluation of the posterior, paper's and this one;
  4. moving T_d changes the deviance (the T_d dependence is wired in);
  5. td_model.mbb equals functions.mbb_scaling_KRJ over the prior of beta_d and T_d.
"""

import os
import time

import tdconfig as C

import numpy as np  # noqa: E402

import functions as F  # noqa: E402
import td_model as TD  # noqa: E402
import wishart_likelihood as W  # noqa: E402
from config import DATASETS, MASKS, wishart_path  # noqa: E402

worst_M, worst_dev = 0.0, 0.0
for m in MASKS:
    for dataset, (bands, tag) in DATASETS.items():
        for mode in ('EE', 'BB'):
            wd = TD.load_wishart(m['key'], mode, bands)
            data = TD.Data(wd)
            z = np.load(wishart_path('pl', m['key'], mode, 'dl20', tag))
            theta = np.asarray(z['theta_map'], float)
            x = np.insert(theta, 6, TD.W.T_DUST)                 # T_d after beta_d
            model = TD.power_law_model()
            M_paper = W.build_model_matrix(theta, wd)
            M_new = model.foreground(model.params(x), data) * data.tt + data.base
            dM = float(np.max(np.abs(M_new - M_paper) / np.abs(M_paper).max(axis=(1, 2))[:, None, None]))
            dev_paper = W.wishart_deviance(wd.D_hat, M_paper, wd.nu_dof, wd.logdet_D)
            dev_new = model.deviance(x, data)
            chi2 = dev_new / (data.n_data - 7)
            worst_M, worst_dev = max(worst_M, dM), max(worst_dev, abs(dev_new - dev_paper))
            dev_hot = model.deviance(np.insert(theta, 6, 25.0), data)
            print(f"{m['key']:9s} {dataset:8s} {mode}: max|dM|/max|M| {dM:.1e}, deviance paper "
                  f"{dev_paper:.6f} new {dev_new:.6f}, chi2_red {chi2:.4f} (paper file "
                  f"{float(z['chi2_reduced']):.4f}); deviance at T_d = 25 K: {dev_hot:.2f}")
print(f'largest relative difference of the model matrices {worst_M:.1e}, of the deviances {worst_dev:.1e}')

# timing, galcut10 BB, QJ+WP+Pl
wd = TD.load_wishart('galcut10', 'BB', DATASETS['QJ+WP+Pl'][0])
data = TD.Data(wd)
theta = np.asarray(np.load(wishart_path('pl', 'galcut10', 'BB'))['theta_map'], float)
x = np.insert(theta, 6, TD.W.T_DUST)
model = TD.power_law_model()
n = 3000
for name, f in (('paper', lambda: W.log_posterior(theta, wd)),
                ('t_dust', lambda: model.log_posterior(x, data))):
    f()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    print(f'{name:7s} log posterior: {(time.perf_counter() - t0) / n * 1e6:.0f} us per call')

# MBB against the paper's function
worst = 0.0
for beta_d in np.linspace(0.0, 6.0, 25):
    for T_d in np.linspace(10.0, 40.0, 31):
        ref = F.mbb_scaling_KRJ(data.freqs, TD.NU_REF_DUST, beta_d, T_d)
        worst = max(worst, float(np.max(np.abs(TD.mbb(beta_d, T_d, data) / ref - 1))))
print(f'td_model.mbb / functions.mbb_scaling_KRJ - 1: largest {worst:.1e} '
      '(0 <= beta_d <= 6, 10 <= T_d <= 40 K)')
