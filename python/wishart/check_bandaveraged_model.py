#%%
"""
Sect. 3.2: is the binning dependence of A and alpha caused by evaluating the
model at l_eff?
=============================================================================

    python check_bandaveraged_model.py [--nproc 12]     (about 1 min with 12 processes)

Refits the QJ+WP+Pl data (maximum of the posterior only) of every mask, mode
and binning with three versions of the model:

  eff       the model of the paper: power law, beams and CMB at l_eff
  bavg_fg   foreground power law averaged over the integer multipoles of the
            bandpower (uniform weights, as the binning); beams and CMB at l_eff
  bavg_all  W_l^i W_l^j (C_l^fg/(u_i u_j) + C_l^CMB) averaged over the bandpower

and prints the shifts, in units of the dl20 posterior uncertainty.  The fits are
saved to results/wishart_hmdm/check_bandaveraged_model.json.
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import BANDS, BINNINGS, MASK, MASKS, WISHART_DIR, noise_path, spectra_path, wishart_path  # noqa: E402

PARAMS = ['A_s', 'alpha_s', 'beta_s', 'A_d', 'alpha_d', 'beta_d', 'rho']
VARIANTS = ['eff', 'bavg_fg', 'bavg_all']


def job(args):
    key, mode, binning, variant = args
    import functions as F
    import wishart_likelihood as W

    # The process is reused between jobs: start from the paper's model every time
    if not hasattr(W, '_paper_model'):
        W._paper_model = W.build_model_matrix
    W.build_model_matrix = W._paper_model

    wd = W.load_wishart_data(spectra_path(key, binning), noise_path(key, binning),
                             BANDS, mode, MASK[key]['fsky'])
    theta_chain = np.asarray(np.load(wishart_path('pl', key, mode, binning))['theta_map'], float)

    if variant != 'eff':
        ells = [np.arange(int(round(a)), int(round(b)) + 1) for a, b in zip(wd.ell1, wd.ell2)]
        L = np.concatenate(ells).astype(float)
        bidx = np.concatenate([np.full(len(e), k) for k, e in enumerate(ells)])
        wts = np.concatenate([np.full(len(e), 1.0 / len(e)) for e in ells])
        if variant == 'bavg_all':
            Wl = W.transfer_functions(wd.bands, mode, L)
            cmb = np.asarray(F.cmb_spectrum(L, mode), float) * 1e12
        else:
            Wl = wd.W[bidx]
            cmb = wd.C_cmb[bidx]
        slices = [np.where(bidx == k)[0] for k in range(len(ells))]
        x = L / W.ELL_REF

        def model(theta, wd_):
            """The model of the paper averaged over the multipoles of each bandpower."""
            A_s, a_s, b_s, A_d, a_d, b_d, rho = [float(t) for t in theta[:7]]
            s = W.sed_synchrotron(wd_.freqs, b_s, wd_.bands, wd_.cc_dict)
            d = W.sed_dust(wd_.freqs, b_d, W.T_DUST, wd_.bands, wd_.cc_dict)
            Fs = wts * A_s * x ** a_s
            Fd = wts * A_d * x ** a_d
            Fx = wts * rho * np.sqrt(max(A_s * A_d, 0.0)) * x ** (0.5 * (a_s + a_d))
            T = Wl / wd_.unit[None, :]
            As, Ad = T * s[None, :], T * d[None, :]
            M = np.empty((len(slices), len(s), len(s)))
            for k, sl in enumerate(slices):
                sd = np.einsum('l,li,lj->ij', Fx[sl], As[sl], Ad[sl])
                M[k] = (np.einsum('l,li,lj->ij', Fs[sl], As[sl], As[sl])
                        + np.einsum('l,li,lj->ij', Fd[sl], Ad[sl], Ad[sl])
                        + sd + sd.T)
                M[k] += np.einsum('l,li,lj->ij', wts[sl] * cmb[sl], Wl[sl], Wl[sl])
            return M + wd_.N

        W.build_model_matrix = model

    t0 = time.time()
    theta, dev = W.maximize_posterior(wd, theta_start=theta_chain, n_restarts=3, seed=1)
    dof = W.n_data_points(wd) - len(PARAMS)
    return dict(region=key, mode=mode, scheme=binning, variant=variant,
                theta=[float(t) for t in theta], dev=float(dev),
                chi2=float(dev / dof), nbins=int(wd.n_bins),
                theta_chain=theta_chain.tolist(), sec=time.time() - t0)


def summary(out):
    """Shifts in units of the dl20 posterior uncertainty."""
    res = {(r['region'], r['mode'], r['scheme'], r['variant']): np.array(r['theta']) for r in out}
    sig = {}
    for m in MASKS:
        for mode in ('EE', 'BB'):
            s = np.load(wishart_path('pl', m['key'], mode))['samples']
            lo, hi = np.percentile(s, [15.865, 84.135], axis=0)
            sig[(m['key'], mode)] = 0.5 * (hi - lo)
    for kind in ('galcut', 'region'):
        keys = [(m['key'], mode) for m in MASKS if m['kind'] == kind for mode in ('EE', 'BB')]
        for var in VARIANTS:
            for alt in ('dl30', 'eqmode'):
                z = np.array([(res[(r, mo, alt, var)] - res[(r, mo, 'dl20', var)]) / sig[(r, mo)]
                              for r, mo in keys])
                k = np.abs(z).argmax(axis=0)
                print(f'{kind:6s} {var:8s} {alt:6s} vs dl20, max: ' + '  '.join(
                    f'{n} {z[k[i], i]:+.2f} ({keys[k[i]][0]} {keys[k[i]][1]})'
                    for i, n in enumerate(PARAMS)))
        z = np.array([(res[(r, mo, 'dl20', 'bavg_all')] - res[(r, mo, 'dl20', 'eff')]) / sig[(r, mo)]
                      for r, mo in keys])
        print(f'{kind:6s} dl20: bavg_all - eff, max |shift|: '
              + '  '.join(f'{n} {np.abs(z[:, i]).max():.2f}' for i, n in enumerate(PARAMS)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=12)
    nproc = ap.parse_args().nproc
    jobs = [(m['key'], mode, b, v) for m in MASKS for mode in ('EE', 'BB')
            for b in BINNINGS for v in VARIANTS]
    out = []
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for res in ex.map(job, jobs):
            out.append(res)
            print(f"{res['region']:9s} {res['mode']} {res['scheme']:6s} "
                  f"{res['variant']:8s} chi2={res['chi2']:.3f} "
                  f"alpha_s={res['theta'][1]:+.3f} alpha_d={res['theta'][4]:+.3f} "
                  f"({res['sec']:.0f}s)", flush=True)
    with open(os.path.join(WISHART_DIR, 'check_bandaveraged_model.json'), 'w') as f:
        json.dump(out, f, indent=1)
    summary(out)
