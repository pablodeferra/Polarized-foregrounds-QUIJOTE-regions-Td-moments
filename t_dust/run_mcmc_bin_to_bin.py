#%%
"""
Bin-to-bin fit with the Wishart likelihood and T_d free.
========================================================

    python run_mcmc_bin_to_bin.py [--masks all] [--modes EE,BB] [--nproc 1] [--nsteps N] [--rerun-below R]

The paper's bin-to-bin fit (python/wishart/run_mcmc_bin_to_bin.py) with the
dust temperature free (td_model.py): each bandpower of the QJ+WP+Pl dataset,
binning dl20, fitted on its own for EE and BB, with alpha_s = alpha_d = 0, the
Gaussian prior beta_s ~ N(-3.1, 0.3^2) and 10 <= T_d <= 40 K: 64 walkers,
NSTEPS steps, the first half discarded.  Writes

    t_dust/results/wishart_hmdm/btb_wishart_td_<mask>_<mode>_dl20.npz

with, per bandpower, the PERCENTILES of the samples (bandpower x 5 x 8
parameters: 2.275, 15.865, 50, 84.135, 97.725), chi2_red at the
maximum-posterior sample (as the paper), the maximum of the likelihood with
its deviance, AIC and BIC, the autocorrelation times and the acceptance
fraction.  The chains themselves are not kept.

The search for the maximum of the posterior starts from the paper's medians
of that bandpower with T_d = 19.6 K, when its file exists.  The
autocorrelation time of these chains is up to ~110 steps, hence NSTEPS: the
kept half is > 50 tau long.  With --rerun-below R, only the bandpowers of the
existing file whose kept half is shorter than R tau are run again, with
--nsteps steps; the file stores the steps of each bandpower.
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse  # noqa: E402

import tdconfig as C  # noqa: E402

import numpy as np  # noqa: E402

import td_model as TD  # noqa: E402
from config import BANDS, MASK, mask_names, wishart_path  # noqa: E402

NWALKERS, NSTEPS, DISCARD = 64, 12000, 0.5
SEED = 42

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--masks', default='all', help='comma-separated mask keys (default: all)')
ap.add_argument('--modes', default='EE,BB', help="'EE', 'BB' or 'EE,BB'")
ap.add_argument('--nproc', type=int, default=1, help='emcee worker processes')
ap.add_argument('--nsteps', type=int, default=NSTEPS)
ap.add_argument('--rerun-below', type=float, default=None,
                help='run again only the bandpowers of the existing file with fewer kept steps than '
                     'this many autocorrelation times')
args = ap.parse_args()
os.makedirs(C.RESULTS_DIR, exist_ok=True)

model = TD.bin_to_bin_model()
for key in mask_names(args.masks):
    for mode in args.modes.split(','):
        wd = TD.load_wishart(key, mode, BANDS)
        paper = wishart_path('btb', key, mode)
        paper = np.load(paper)['percentiles'][:, 1] if os.path.exists(paper) else None
        out = C.result_path('btb', key, mode)
        done = TD.converged_bandpowers(out, args.rerun_below)
        if len(done) == wd.n_bins:
            continue
        fits = []
        for b in range(wd.n_bins):
            if b in done:
                fits.append(done[b])
                continue
            data = TD.Data(wd.select_bins(b))
            x_start = None if paper is None else TD.from_paper(model, paper[b])
            r = TD.run_mcmc(model, data, NWALKERS, args.nsteps, DISCARD, seed=SEED + b,
                            n_processes=args.nproc, x_start=x_start)
            r['percentiles'] = np.percentile(r.pop('samples'), TD.PERCENTILES, axis=0)
            r.pop('log_prob')
            r['nsteps'] = args.nsteps
            fits.append(r)
            p = r['percentiles']
            print(f"[{key} {mode}] l_eff = {wd.ell_eff[b]:5.1f}  beta_d = {p[2, 5]:.3f} "
                  f"(+{p[3, 5] - p[2, 5]:.3f} -{p[2, 5] - p[1, 5]:.3f})  T_d = {p[2, 6]:.1f} "
                  f"[{p[0, 6]:.1f}, {p[4, 6]:.1f}] K (95%)  chi2_red = {r['chi2_reduced']:.3f}  "
                  f"D_ml = {r['deviance_ml']:.2f}  tau max = {r['tau'].max():.0f}", flush=True)
        stack = lambda name: np.array([f[name] for f in fits])            # noqa: E731
        np.savez_compressed(
            out, percentiles=stack('percentiles'), percentile_levels=np.array(TD.PERCENTILES),
            param_names=np.array(model.names), free=np.array(model.free),
            theta_map=stack('theta_map'), chi2_reduced=stack('chi2_reduced'),
            theta_ml=stack('theta_ml'), deviance_ml=stack('deviance_ml'),
            deviance_min_sample=stack('deviance_min_sample'), aic=stack('aic'), bic=stack('bic'),
            n_par=model.n_par, n_data=fits[0]['n_data'], tau=stack('tau'),
            acceptance=stack('acceptance'), nsteps=stack('nsteps'), nwalkers=NWALKERS, discard=DISCARD,
            ell_eff=wd.ell_eff, ell1=wd.ell1, ell2=wd.ell2, bands=np.array(BANDS),
            f_sky=MASK[key]['fsky'], noise=C.config.NOISE)
        print(f'  -> {out}', flush=True)
