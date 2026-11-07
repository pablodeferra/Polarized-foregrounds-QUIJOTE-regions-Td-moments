#%%
"""
Bin-to-bin fit with the Gaussian likelihood (not shown in the paper).
=====================================================================

    python run_mcmc_bin_to_bin.py [--masks all] [--nproc 6]

Each of the nine dl20 bandpowers of the QJ+WP+Pl dataset fitted on its own
for A_s, beta_s, A_d, beta_d and rho, with a Gaussian prior beta_s ~
N(-3.1, 0.3^2), for EE and BB: 100 walkers, 5000 steps, the first half
discarded.  Writes

    results/gaussian_hmdm/btb_gauss_<mask>_<mode>_dl20.npz

with, per bandpower, the 15.865/50/84.135 percentiles of every parameter,
the maximum-posterior sample, chi2_red there and the autocorrelation times
(amplitudes in mK^2_RJ).
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import gaussian_likelihood as G  # noqa: E402
from config import BANDS, GAUSS_DIR, corrected_path, gauss_path, mask_names  # noqa: E402

NWALKERS, NSTEPS, DISCARD = 100, 5000, 0.5
PERCENTILES = [15.865, 50.0, 84.135]
SEED = 42

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--masks', default='all', help='comma-separated mask keys (default: all)')
ap.add_argument('--nproc', type=int, default=6, help='emcee worker processes')
args = ap.parse_args()
os.makedirs(GAUSS_DIR, exist_ok=True)

for key in mask_names(args.masks):
    for mode in ('EE', 'BB'):
        fit = G.load_fit_data(corrected_path(key), BANDS, mode)
        samplers, samples, chi2_red = G.run_bin_to_bin(fit, NWALKERS, NSTEPS, DISCARD,
                                                       n_processes=args.nproc, seed=SEED)
        discard = int(NSTEPS * DISCARD)
        out = gauss_path('btb', key, mode)
        np.savez(out, params=np.array(G.PARAMS_BTB),
                 percentiles=np.array([np.percentile(s, PERCENTILES, axis=0) for s in samples]),
                 theta_map=np.array([s[int(np.argmax(sp.get_log_prob(discard=discard, flat=True)))]
                                     for s, sp in zip(samples, samplers)]),
                 chi2_reduced=np.array(chi2_red, float), ell_eff=fit['ell_eff'],
                 tau=np.array([sp.get_autocorr_time(quiet=True) for sp in samplers]))
        print(f'[{key} {mode}] chi2_red per bandpower '
              + ' '.join(f'{c:.2f}' for c in chi2_red) + f'\n  -> {out}', flush=True)
