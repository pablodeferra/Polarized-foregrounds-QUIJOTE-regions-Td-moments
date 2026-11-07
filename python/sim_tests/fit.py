#%%
"""
Fits of the PySM sky plus noise simulation k (simulation tests).
================================================================

    python fit.py wishart  --kind pl  --masks nps --first 1 --last 10 [--nproc 5]
    python fit.py gaussian --kind pl  --masks nps --first 1 --last 10 [--nproc 6]
    python fit.py wishart  --kind btb --masks all --first 1 --last 1

The spectra of spectra.py, realization by realization, fitted as the data of
the paper (14 bands, dl20, EE and BB): the Wishart likelihood with N_l the
mean of the noise simulations; the Gaussian one with the spectra corrected by
gaussian/correct_spectra.py (noise mean, errors from the simulations).  Same
walkers, steps, priors and seeds as the paper, but with colour corrections 1
and the exact K_CMB -> K_RJ at nu0 (simlib.monochromatic: the PySM maps are
monochromatic).

Power law: the summaries are also kept as if the chains had been shorter
(LENGTHS: the first L steps, the first half of them discarded; the first L
steps of a chain are those of a run of L steps), to choose the chain length.
Wishart: L total steps.  Gaussian: L steps after its burn-in of 1000.

Writes, after every realization (a failed job keeps what it finished),

    results/sim_tests/fits/<kind>_<likelihood>_<mask>_k<first>-<last>.npz

with k, params, modes and, per realization and mode, the 15.865/50/84.135
percentiles, mean and standard deviation of every parameter (pl: per
length; btb: per bandpower), the maximum-posterior sample and chi2_red there.
Amplitudes in uK^2_RJ (Wishart) or mK^2_RJ (Gaussian), as the paper's fits.
No chains are kept.  A realization whose data matrix is not positive definite
cannot be fitted with the Wishart likelihood: its mode is written as NaN
(chi2_red NaN) and the job goes on.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simlib as S  # noqa: E402  (sets the paths)
import numpy as np  # noqa: E402
import gaussian_likelihood as G  # noqa: E402
import wishart_likelihood as W  # noqa: E402
from config import BANDS, MASK, N_SIM, mask_names, sims_path  # noqa: E402

PERCENTILES = [15.865, 50.0, 84.135]
SETTINGS = {   # walkers, steps, seed: as the paper's run_mcmc_*.py
    ('wishart', 'pl'): dict(nwalkers=64, nsteps=8000, lengths=[2000, 4000, 6000, 8000]),
    ('gaussian', 'pl'): dict(nwalkers=100, nsteps=5000, lengths=[1250, 2500, 3750, 5000]),
    ('wishart', 'btb'): dict(nwalkers=64, nsteps=5000),
    ('gaussian', 'btb'): dict(nwalkers=100, nsteps=5000),
}
SEED = 42


def summary(samples):
    """percentiles (3, n_par), mean and std (n_par) of flat samples."""
    return np.percentile(samples, PERCENTILES, axis=0), samples.mean(axis=0), samples.std(axis=0)


def fit_pl(likelihood, key, raw, corrected, mode, cfg, nproc):
    """Summaries per length (n_L, ...), maximum-posterior sample, chi2_red."""
    if likelihood == 'wishart':
        wd = W.load_wishart_data(raw, sims_path(key, 'noise', S.BINNING), BANDS, mode, MASK[key]['fsky'])
        wd.check_positive_definite()
        samples, info = W.run_wishart_mcmc(wd, cfg['nwalkers'], cfg['nsteps'], discard_fraction=0.0,
                                           n_processes=nproc, seed=SEED)
        chain = samples.reshape(cfg['nsteps'], cfg['nwalkers'], -1)
        theta_map, chi2 = info['theta_map'], info['chi2_reduced']
    else:
        fit = G.load_fit_data(corrected, BANDS, mode)
        sampler, _, chi2 = G.run_power_law(fit, cfg['nwalkers'], cfg['nsteps'], 0.5, n_processes=nproc,
                                           seed=SEED)
        chain = sampler.get_chain()
        half = cfg['nsteps'] // 2
        flat = sampler.get_chain(discard=half, flat=True)
        theta_map = flat[int(np.argmax(sampler.get_log_prob(discard=half, flat=True)))]
    out = [summary(chain[L // 2:L].reshape(-1, chain.shape[-1])) for L in cfg['lengths']]
    return [np.array(x) for x in zip(*out)], theta_map, chi2


def fit_btb(likelihood, key, raw, corrected, mode, cfg, nproc):
    """Summaries per bandpower (n_bins, ...), maximum-posterior samples, chi2_red."""
    res = []
    if likelihood == 'wishart':
        wd = W.load_wishart_data(raw, sims_path(key, 'noise', S.BINNING), BANDS, mode, MASK[key]['fsky'])
        wd.check_positive_definite()
        for b in range(wd.n_bins):
            samples, info = W.run_wishart_mcmc(wd.select_bins(b), cfg['nwalkers'], cfg['nsteps'], 0.5,
                                               fixed=W.BINBIN_FIXED, gaussian_priors=W.BINBIN_PRIORS,
                                               n_processes=nproc, seed=SEED + b)
            res.append((summary(samples), info['theta_map'], info['chi2_reduced']))
    else:
        fit = G.load_fit_data(corrected, BANDS, mode)
        samplers, samples, chi2 = G.run_bin_to_bin(fit, cfg['nwalkers'], cfg['nsteps'], 0.5,
                                                   n_processes=nproc, seed=SEED)
        half = cfg['nsteps'] // 2
        for sp, s, c in zip(samplers, samples, chi2):
            res.append((summary(s), s[int(np.argmax(sp.get_log_prob(discard=half, flat=True)))], c))
    summaries, theta_map, chi2 = zip(*res)
    return [np.array(x) for x in zip(*summaries)], np.array(theta_map), np.array(chi2)


def nan_result(kind, n_par, cfg, n_bins):
    """The result of a fit that could not be done, with the shapes of fit_pl / fit_btb."""
    n = len(cfg['lengths']) if kind == 'pl' else n_bins
    nan = lambda *shape: np.full(shape, np.nan)  # noqa: E731
    if kind == 'pl':
        return [nan(n, 3, n_par), nan(n, n_par), nan(n, n_par)], nan(n_par), np.nan
    return [nan(n, 3, n_par), nan(n, n_par), nan(n, n_par)], nan(n, n_par), nan(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('likelihood', choices=['wishart', 'gaussian'])
    ap.add_argument('--kind', choices=['pl', 'btb'], default='pl')
    ap.add_argument('--masks', default='galcut10,nps', help='comma-separated mask keys')
    ap.add_argument('--first', type=int, default=1)
    ap.add_argument('--last', type=int, default=N_SIM)
    ap.add_argument('--nproc', type=int, default=5, help='emcee worker processes')
    ap.add_argument('--nsteps', type=int, default=None, help='steps (only for quick tests; default: the paper)')
    args = ap.parse_args()
    cfg = dict(SETTINGS[(args.likelihood, args.kind)])
    if args.nsteps:
        if 'lengths' in cfg:
            cfg['lengths'] = [L * args.nsteps // cfg['nsteps'] for L in cfg['lengths']]
        cfg['nsteps'] = args.nsteps
    S.monochromatic()
    fitter = fit_pl if args.kind == 'pl' else fit_btb
    params = W.PARAM_NAMES if (args.likelihood == 'wishart' or args.kind == 'pl') else G.PARAMS_BTB

    for key in mask_names(args.masks):
        out = os.path.join(S.FITS_DIR, f'{args.kind}_{args.likelihood}_{key}_k{args.first:03d}-{args.last:03d}.npz')
        os.makedirs(S.FITS_DIR, exist_ok=True)
        rows = []
        for k in range(args.first, args.last + 1):
            t0 = time.time()
            per_mode = []
            with S.realization_fits(key, k) as (raw, corrected):
                if args.likelihood == 'gaussian':
                    S.correct_spectra(key, raw, corrected)
                for mode in S.MODES:
                    try:
                        per_mode.append(fitter(args.likelihood, key, raw, corrected, mode, cfg, args.nproc))
                    except ValueError as err:
                        if 'positive definite' not in str(err):
                            raise
                        print(f'[{key} {args.likelihood} {args.kind}] k = {k:3d} {mode} not fitted: {err}', flush=True)
                        per_mode.append(nan_result(args.kind, len(params), cfg, S.dl20().get_n_bands()))
            rows.append(per_mode)
            summaries = [np.array([[m[0][i] for m in r] for r in rows]) for i in range(3)]
            np.savez(out, k=np.arange(args.first, k + 1), params=np.array(params), modes=np.array(S.MODES),
                     percentiles=summaries[0], percentile_levels=np.array(PERCENTILES),
                     mean=summaries[1], std=summaries[2],
                     theta_map=np.array([[m[1] for m in r] for r in rows]),
                     chi2_reduced=np.array([[m[2] for m in r] for r in rows]),
                     lengths=np.array(cfg.get('lengths', [cfg['nsteps']])), nwalkers=cfg['nwalkers'],
                     nsteps=cfg['nsteps'], likelihood=args.likelihood, kind=args.kind,
                     amplitude_unit='uK^2_RJ' if args.likelihood == 'wishart' else 'mK^2_RJ')
            print(f'[{key} {args.likelihood} {args.kind}] k = {k:3d} in {time.time() - t0:.0f} s; '
                  f'chi2_red EE/BB = {np.mean(per_mode[0][2]):.3f}/{np.mean(per_mode[1][2]):.3f}', flush=True)
        print(f'  -> {out}', flush=True)


if __name__ == '__main__':
    main()
