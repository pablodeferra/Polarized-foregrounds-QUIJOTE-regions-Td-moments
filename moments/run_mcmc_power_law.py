#%%
"""
Global power-law fit with the Wishart likelihood, T_d free and first-order moments.
===================================================================================

    python run_mcmc_power_law.py [--moments all] [--masks all] [--datasets all] [--modes EE,BB] [--nproc 1] [--nsteps N]

The power-law fit of t_dust/ (eight parameters, T_d free) with the first-order
moment terms of moments_model.py: 'sync' (w_s_bb, 9 parameters), 'dust'
(w_d_bb, w_d_TT, w_d_bT, 11) or 'both' (12).  Each moment spectrum is its
component's amplitude spectrum A_c (l/80)^alpha_c times a constant w.  EE and
BB, WP+Pl and QJ+WP+Pl, binning dl20: 64 walkers, NSTEPS steps, the first half
discarded.  Writes, per fit,

    moments/results/wishart_hmdm/pl_wishart_mom_<moments>_<mask>_<mode><tag>_dl20.npz

with the same contents as t_dust/run_mcmc_power_law.py.  The search for the
maximum of the posterior starts from the T_d-free solution of t_dust/ (or the
paper's, with T_d = 19.6 K) with every w = 0.
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse  # noqa: E402

import mconfig as MC  # noqa: E402

import numpy as np  # noqa: E402

import moments_model as MM  # noqa: E402
import tdconfig  # noqa: E402
import td_model as TD  # noqa: E402
from config import DATASETS, MASK, mask_names, wishart_path  # noqa: E402

NWALKERS, NSTEPS, DISCARD = 64, 24000, 0.5
THIN = 4
SEED = 42

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--moments', default='all', help="'sync', 'dust', 'both' or 'all' (comma-separated)")
ap.add_argument('--masks', default='all', help='comma-separated mask keys (default: all)')
ap.add_argument('--datasets', default='all', help="'QJ+WP+Pl', 'WP+Pl' or 'all'")
ap.add_argument('--modes', default='EE,BB', help="'EE', 'BB' or 'EE,BB'")
ap.add_argument('--nproc', type=int, default=1, help='emcee worker processes')
ap.add_argument('--nsteps', type=int, default=NSTEPS)
args = ap.parse_args()
moment_sets = list(MC.MOMENT_SETS) if args.moments == 'all' else args.moments.split(',')
datasets = list(DATASETS) if args.datasets == 'all' else args.datasets.split(',')
os.makedirs(MC.RESULTS_DIR, exist_ok=True)


def start(model, key, mode, tag):
    """The T_d-free maximum of the posterior of t_dust/ (else the paper's), moments at 0."""
    f = tdconfig.result_path('pl', key, mode, 'dl20', tag)
    if os.path.exists(f):
        return MM.start_from(model, np.load(f)['theta_map'])
    f = wishart_path('pl', key, mode, 'dl20', tag)
    if os.path.exists(f):
        return MM.start_from(model, TD.from_paper(TD.power_law_model(), np.load(f)['theta_map']))
    return None


for mom in moment_sets:
    model = MM.power_law_model(mom)
    for key in mask_names(args.masks):
        for dataset in datasets:
            bands, tag = DATASETS[dataset]
            for mode in args.modes.split(','):
                data = TD.Data(TD.load_wishart(key, mode, bands))
                r = TD.run_mcmc(model, data, NWALKERS, args.nsteps, DISCARD, seed=SEED,
                                n_processes=args.nproc, x_start=start(model, key, mode, tag))
                npar = len(model.names)
                samples = r['samples'].reshape(-1, NWALKERS, npar)[::THIN].reshape(-1, npar)
                log_prob = r['log_prob'].reshape(-1, NWALKERS)[::THIN].ravel()
                out = MC.result_path('pl', key, mode, mom, 'dl20', tag)
                np.savez_compressed(
                    out, samples=samples.astype(np.float32), log_prob=log_prob.astype(np.float32),
                    param_names=np.array(model.names), free=np.array(model.free), moments=mom,
                    theta_map=r['theta_map'], chi2_reduced=r['chi2_reduced'],
                    theta_ml=r['theta_ml'], deviance_ml=r['deviance_ml'],
                    deviance_min_sample=r['deviance_min_sample'], aic=r['aic'], bic=r['bic'],
                    n_par=r['n_par'], n_data=r['n_data'], tau=r['tau'], acceptance=r['acceptance'],
                    nsteps=args.nsteps, nwalkers=NWALKERS, discard=DISCARD, thin=THIN,
                    ell_eff=data.wd.ell_eff, nu_dof=data.wd.nu_dof, bands=np.array(bands),
                    f_sky=MASK[key]['fsky'], noise=MC.config.NOISE)
                s = TD.summarize(r['samples'], model.names)
                print(f"[{mom} {key} {dataset} {mode}] chi2_red = {r['chi2_reduced']:.3f}, "
                      f"D_ml = {r['deviance_ml']:.2f}, tau max = {r['tau'].max():.0f} steps "
                      f"(kept steps/tau = {args.nsteps * (1 - DISCARD) / r['tau'].max():.0f}), "
                      f"acceptance {r['acceptance']:.2f}\n  "
                      + '  '.join(f'{n}={v[0]:.4g}+{v[1]:.2g}-{v[2]:.2g}' for n, v in s.items())
                      + f'\n  -> {out}', flush=True)
