#%%
"""
Bin-to-bin fit with the Wishart likelihood.
===========================================

    python run_mcmc_bin_to_bin.py [--binning dl20] [--masks all] [--nproc 5]

Each bandpower of the QJ+WP+Pl dataset is fitted on its own, for EE and BB,
with the multipole slopes fixed to alpha_s = alpha_d = 0 (so that A_s and A_d
are the power in that bandpower) and a Gaussian prior beta_s ~ N(-3.1, 0.3^2):
64 walkers, 5000 steps, the first half discarded.  Writes

    results/wishart_hmdm/btb_wishart_<mask>_<mode>_<binning>.npz

with the 15.865, 50 and 84.135 percentiles of the samples of every bandpower
(percentiles: bandpower x 3 x 7 parameters, slopes constant; the median and
68 per cent interval quoted in the paper) and chi2_red per bandpower.
The chains themselves are not kept.
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import wishart_likelihood as W  # noqa: E402
from config import BANDS, BINNINGS, MASK, WISHART_DIR, mask_names, noise_path, spectra_path, wishart_path  # noqa: E402

NWALKERS, NSTEPS, DISCARD = 64, 5000, 0.5
SEED = 42

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--binning', default='dl20', choices=list(BINNINGS))
ap.add_argument('--masks', default='all', help='comma-separated mask keys (default: all)')
ap.add_argument('--nproc', type=int, default=5, help='emcee worker processes')
args = ap.parse_args()
os.makedirs(WISHART_DIR, exist_ok=True)

for key in mask_names(args.masks):
    f_sky = MASK[key]['fsky']
    for mode in ('EE', 'BB'):
        wd = W.load_wishart_data(spectra_path(key, args.binning),
                                 noise_path(key, args.binning), BANDS, mode, f_sky)
        wd.check_positive_definite()
        chains, chi2_reduced = [], []
        for b in range(wd.n_bins):
            samples, info = W.run_wishart_mcmc(
                wd.select_bins(b), NWALKERS, NSTEPS, DISCARD, fixed=W.BINBIN_FIXED,
                gaussian_priors=W.BINBIN_PRIORS, n_processes=args.nproc, seed=SEED + b)
            chains.append(samples)
            chi2_reduced.append(info['chi2_reduced'])
            s = W.summarize_samples(samples)
            print(f"[{key} {mode} {args.binning}] l_eff = {wd.ell_eff[b]:5.1f}  "
                  f"A_s = {s['A_s'][0] * 1e3:9.3f}  beta_s = {s['beta_s'][0]:+.3f}  "
                  f"beta_d = {s['beta_d'][0]:.3f}  rho = {s['rho'][0]:+.3f}  "
                  f"chi2_red = {info['chi2_reduced']:.3f}", flush=True)
        out = wishart_path('btb', key, mode, args.binning)
        np.savez_compressed(out, percentiles=np.array([np.percentile(c, W.PERCENTILES, axis=0) for c in chains]),
                            percentile_levels=np.array(W.PERCENTILES), chi2_reduced=np.array(chi2_reduced),
                            ell_eff=wd.ell_eff, ell1=wd.ell1, ell2=wd.ell2,
                            bands=np.array(BANDS), f_sky=f_sky)
        print(f'  -> {out}', flush=True)
