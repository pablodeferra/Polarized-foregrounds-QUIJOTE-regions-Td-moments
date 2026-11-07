#%%
"""
Global power-law fit with the Gaussian likelihood (Appendix A).
===============================================================

    python run_mcmc_power_law.py [--masks all] [--nproc 6]

The four fits of every mask: WP+Pl and QJ+WP+Pl, EE and BB, all
auto- and cross-spectra of the nine dl20 bandpowers, with the corrected
spectra of correct_spectra.py: 100 walkers, 1000 burn-in steps, then 5000
steps of which the first half is discarded.  Writes, per fit,

    results/gaussian_hmdm/pl_gauss_<mask>_<mode><tag>_dl20.npz

with the 15.865/50/84.135 percentiles of every parameter, the
maximum-posterior sample, chi2_red there, the autocorrelation times and
20000 samples (amplitudes in mK^2_RJ).
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import gaussian_likelihood as G  # noqa: E402
from config import DATASETS, GAUSS_DIR, corrected_path, gauss_path, mask_names  # noqa: E402

NWALKERS, NSTEPS, DISCARD = 100, 5000, 0.5
PERCENTILES = [15.865, 50.0, 84.135]
N_KEEP = 20000             # samples kept for the B-to-E ratios
SEED = 42

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--masks', default='all', help='comma-separated mask keys (default: all)')
ap.add_argument('--nproc', type=int, default=6, help='emcee worker processes')
args = ap.parse_args()
os.makedirs(GAUSS_DIR, exist_ok=True)

for key in mask_names(args.masks):
    for dataset, (bands, tag) in DATASETS.items():
        for mode in ('EE', 'BB'):
            fit = G.load_fit_data(corrected_path(key), bands, mode)
            sampler, samples, chi2_red = G.run_power_law(fit, NWALKERS, NSTEPS, DISCARD,
                                                         n_processes=args.nproc, seed=SEED)
            logp = sampler.get_log_prob(discard=int(NSTEPS * DISCARD), flat=True)
            keep = np.random.default_rng(SEED).choice(len(samples), size=N_KEEP, replace=False)
            out = gauss_path('pl', key, mode, tag)
            np.savez(out, params=np.array(G.PARAMS_PL),
                     percentiles=np.percentile(samples, PERCENTILES, axis=0),
                     theta_map=samples[int(np.argmax(logp))], chi2_reduced=float(chi2_red),
                     n_data=len(fit['y_all']), ell_eff=fit['ell_eff'],
                     tau=sampler.get_autocorr_time(quiet=True),
                     samples=samples[keep].astype(np.float32),
                     A_s_samples=samples[keep, 0], A_d_samples=samples[keep, 3])
            print(f'[{key} {dataset} {mode}] chi2_red = {chi2_red:.3f}, max tau = '
                  f'{np.nanmax(sampler.get_autocorr_time(quiet=True)):.0f} steps\n  -> {out}',
                  flush=True)
