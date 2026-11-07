#%%
"""
Global power-law fit with the Wishart likelihood.
=================================================

    python run_mcmc_power_law.py [--binning dl20] [--masks all] [--nproc 5]

The seven parameters are fitted to all the bandpowers at once, for the EE and
BB spectra and for the WP+Pl (12 bands) and QJ+WP+Pl (14 bands) datasets:
64 walkers, 8000 steps, the first half discarded.  Reads the data spectra of
compute_spectra.py and the noise N_l of config.noise_path (the HMDM spectra or,
with SPECIFIC_REGIONS_NOISE=sims, the mean of the noise simulations, results
then in results/wishart/) and writes, per fit,

    results/wishart_hmdm/pl_wishart_<mask>_<mode><tag>_<binning>.npz

(tag '' for QJ+WP+Pl, '_WPPl' for WP+Pl) with the samples (every THIN-th
step of each walker, float32), the maximum-posterior sample (of the full chain)
and chi2_red there.  Amplitudes in uK^2_RJ.
"""

import os

# emcee runs one process per CPU: no extra threads inside each
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import wishart_likelihood as W  # noqa: E402
from config import BINNINGS, DATASETS, MASK, WISHART_DIR, mask_names, noise_path, spectra_path, wishart_path  # noqa: E402

NWALKERS, NSTEPS, DISCARD = 64, 8000, 0.5
# every THIN-th step of each walker is kept, in float32: the autocorrelation time
# of these chains is ~70 steps, so the medians and intervals change by < 0.01 sigma
THIN = 4

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--binning', default='dl20', choices=list(BINNINGS))
ap.add_argument('--masks', default='all', help='comma-separated mask keys (default: all)')
ap.add_argument('--nproc', type=int, default=5, help='emcee worker processes')
args = ap.parse_args()
os.makedirs(WISHART_DIR, exist_ok=True)

for key in mask_names(args.masks):
    f_sky = MASK[key]['fsky']
    for dataset, (bands, tag) in DATASETS.items():
        for mode in ('EE', 'BB'):
            wd = W.load_wishart_data(spectra_path(key, args.binning),
                                     noise_path(key, args.binning), bands, mode, f_sky)
            wd.check_positive_definite()
            samples, info = W.run_wishart_mcmc(wd, NWALKERS, NSTEPS, DISCARD, n_processes=args.nproc)
            out = wishart_path('pl', key, mode, args.binning, tag)
            samples = samples.reshape(-1, NWALKERS, samples.shape[1])[::THIN].reshape(-1, samples.shape[1])
            np.savez_compressed(out, samples=samples.astype(np.float32), theta_map=info['theta_map'],
                                chi2_reduced=info['chi2_reduced'], n_data=info['n_data'],
                                ell_eff=wd.ell_eff, nu_dof=wd.nu_dof, bands=np.array(bands),
                                f_sky=f_sky)
            med = np.median(samples, axis=0)
            print(f"[{key} {dataset} {mode} {args.binning}] chi2_red = {info['chi2_reduced']:.3f}, "
                  'median ' + ' '.join(f'{n}={v:.4g}' for n, v in zip(W.PARAM_NAMES, med))
                  + f'\n  -> {out}', flush=True)
