#%%
"""
Global power-law fit with the Wishart likelihood and T_d free.
==============================================================

    python run_mcmc_power_law.py [--masks all] [--datasets all] [--modes EE,BB] [--nproc 1] [--nsteps N]

The paper's global power-law fit (python/wishart/run_mcmc_power_law.py) with
the dust temperature as an eighth parameter (td_model.py), for the EE and BB
spectra and the WP+Pl and QJ+WP+Pl datasets, binning dl20: 64 walkers, NSTEPS
steps, the first half discarded.  Writes, per fit,

    t_dust/results/wishart_hmdm/pl_wishart_td_<mask>_<mode><tag>_dl20.npz

(tag '' for QJ+WP+Pl, '_WPPl' for WP+Pl) with the samples and their log
posterior (every THIN-th step of each walker, float32), the maximum-posterior
sample and chi2_red there (as the paper), the maximum of the likelihood with
its deviance, AIC and BIC, the autocorrelation times and the acceptance
fraction.  Amplitudes in uK^2_RJ.  The noise term is that of tdconfig (HMDM
unless SPECIFIC_REGIONS_NOISE says otherwise).

The search for the maximum of the posterior starts from the paper's solution
(its maximum-posterior sample with T_d = 19.6 K) when its file exists.  The
autocorrelation time of these chains is ~80-150 steps, hence NSTEPS: the kept
half is > 50 tau long.
"""

import os

# emcee runs one process per CPU: no extra threads inside each
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse  # noqa: E402

import tdconfig as C  # noqa: E402

import numpy as np  # noqa: E402

import td_model as TD  # noqa: E402
from config import DATASETS, MASK, mask_names, wishart_path  # noqa: E402

NWALKERS, NSTEPS, DISCARD = 64, 16000, 0.5
# every THIN-th step of each walker is kept, in float32, as in the paper
THIN = 4
SEED = 42

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--masks', default='all', help='comma-separated mask keys (default: all)')
ap.add_argument('--datasets', default='all', help="'QJ+WP+Pl', 'WP+Pl' or 'all'")
ap.add_argument('--modes', default='EE,BB', help="'EE', 'BB' or 'EE,BB'")
ap.add_argument('--nproc', type=int, default=1, help='emcee worker processes')
ap.add_argument('--nsteps', type=int, default=NSTEPS)
args = ap.parse_args()
datasets = list(DATASETS) if args.datasets == 'all' else args.datasets.split(',')
os.makedirs(C.RESULTS_DIR, exist_ok=True)

model = TD.power_law_model()
for key in mask_names(args.masks):
    for dataset in datasets:
        bands, tag = DATASETS[dataset]
        for mode in args.modes.split(','):
            data = TD.Data(TD.load_wishart(key, mode, bands))
            paper = wishart_path('pl', key, mode, 'dl20', tag)
            x_start = TD.from_paper(model, np.load(paper)['theta_map']) if os.path.exists(paper) else None
            r = TD.run_mcmc(model, data, NWALKERS, args.nsteps, DISCARD, seed=SEED,
                            n_processes=args.nproc, x_start=x_start)
            npar = len(model.names)
            samples = r['samples'].reshape(-1, NWALKERS, npar)[::THIN].reshape(-1, npar)
            log_prob = r['log_prob'].reshape(-1, NWALKERS)[::THIN].ravel()
            out = C.result_path('pl', key, mode, 'dl20', tag)
            np.savez_compressed(
                out, samples=samples.astype(np.float32), log_prob=log_prob.astype(np.float32),
                param_names=np.array(model.names), free=np.array(model.free),
                theta_map=r['theta_map'], chi2_reduced=r['chi2_reduced'],
                theta_ml=r['theta_ml'], deviance_ml=r['deviance_ml'],
                deviance_min_sample=r['deviance_min_sample'], aic=r['aic'], bic=r['bic'],
                n_par=r['n_par'], n_data=r['n_data'], tau=r['tau'], acceptance=r['acceptance'],
                nsteps=args.nsteps, nwalkers=NWALKERS, discard=DISCARD, thin=THIN,
                ell_eff=data.wd.ell_eff, nu_dof=data.wd.nu_dof, bands=np.array(bands),
                f_sky=MASK[key]['fsky'], noise=C.config.NOISE)
            s = TD.summarize(r['samples'])
            print(f"[{key} {dataset} {mode}] chi2_red = {r['chi2_reduced']:.3f}, "
                  f"D_ml = {r['deviance_ml']:.2f}, tau max = {r['tau'].max():.0f} steps "
                  f"(kept steps/tau = {args.nsteps * (1 - DISCARD) / r['tau'].max():.0f}), "
                  f"acceptance {r['acceptance']:.2f}\n  "
                  + '  '.join(f'{n}={v[0]:.4g}+{v[1]:.2g}-{v[2]:.2g}' for n, v in s.items())
                  + f'\n  -> {out}', flush=True)
