#%%
"""
Is T_d worth a free parameter?  AIC and BIC of T_d fixed and T_d free.
======================================================================

    python compare_models.py

For every fit of run_mcmc_power_law.py (mask x dataset x mode) and of
run_mcmc_bin_to_bin.py (mask x mode x bandpower), the maximum of the
likelihood within the uniform priors (deviance D_ml) of

  fixed   the paper's model, T_d = 19.6 K: k = 7 (power law), 5 (bin to bin),
          maximized here from the T_d-free maximum and from the paper's solution
  free    T_d free: k = 8 and 6, from the result files

and the information criteria AIC = D_ml + 2k and BIC = D_ml + k ln N, with
N = n_data the independent entries of the data matrices (945 for the power
law, 105 per bandpower).  With dD = D_fixed - D_free (>= 0, nested models):

    dAIC = AIC_free - AIC_fixed = 2 - dD,     dBIC = BIC_free - BIC_fixed = ln N - dD,

negative values favouring T_d free.  Usual reading of |dAIC| or |dBIC|: < 2
no real preference, 2-6 positive, 6-10 strong, > 10 very strong (Kass &
Raftery 1995 for BIC; Burnham & Anderson 2002 for AIC).  For the bin-to-bin
fits the nine bandpowers are also combined: dAIC = sum over bandpowers,
dBIC = 9 ln N_total - sum dD with N_total = 945, the BIC of the nine fits
taken as one model.

Since the models are nested, D_fixed >= D_free must hold.  When the
maximization of the free model ended above the fixed one, it is redone from
the fixed solution and the better maximum is kept.

Writes t_dust/results/wishart_hmdm/model_comparison_td_dl20.npz (one row per
fit) and prints a summary.
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import tdconfig as C  # noqa: E402

import numpy as np  # noqa: E402

import td_model as TD  # noqa: E402
from config import BANDS, DATASETS, MASKS, wishart_path  # noqa: E402


def free_vector(model, theta, names=TD.PARAM_NAMES):
    """The free parameters of `model` from a full vector `theta` over `names`."""
    p = dict(zip(names, map(float, theta)))
    return np.array([p[n] for n in model.free])


def compare(fixed, free, data, theta_free, dev_free, paper_start):
    """
    Maximum likelihood of the fixed model (from the free maximum and from the
    paper's solution) and, if needed, the free one redone from it.
    Returns (theta_fixed, D_fixed, theta_free, D_free), full 8-vectors.
    """
    best = None
    for start in (free_vector(fixed, theta_free), paper_start):
        if start is None:
            continue
        x, d = TD.maximize(fixed, data, start, gaussian=False, seed=1)
        if best is None or d < best[1]:
            best = (x, d)
    x_fixed, d_fixed = best
    theta_fixed = fixed.full(x_fixed)[0]
    if d_fixed < dev_free - 1e-6:
        x, d = TD.maximize(free, data, free_vector(free, theta_fixed), gaussian=False, seed=2)
        if d < dev_free:
            theta_free, dev_free = free.full(x)[0], d
    return theta_fixed, d_fixed, np.asarray(theta_free, float), float(dev_free)


rows = []
pl_fixed, pl_free = TD.power_law_model(TD.T_D_FIXED), TD.power_law_model()
for m in MASKS:
    for dataset, (bands, tag) in DATASETS.items():
        for mode in ('EE', 'BB'):
            z = np.load(C.result_path('pl', m['key'], mode, 'dl20', tag))
            data = TD.Data(TD.load_wishart(m['key'], mode, bands))
            paper = wishart_path('pl', m['key'], mode, 'dl20', tag)
            start = TD.from_paper(pl_fixed, np.load(paper)['theta_map']) if os.path.exists(paper) else None
            tf, df, tF, dF = compare(pl_fixed, pl_free, data, z['theta_ml'], float(z['deviance_ml']), start)
            rows.append(dict(kind='pl', mask=m['key'], dataset=dataset, mode=mode, bin=-1,
                             ell_eff=np.nan, n_data=data.n_data, k_fixed=pl_fixed.n_par,
                             k_free=pl_free.n_par, dev_fixed=df, dev_free=dF,
                             theta_fixed=tf, theta_free=tF))
            print(f"pl  {m['key']:9s} {dataset:8s} {mode}: D_fixed {df:9.2f}  D_free {dF:9.2f}  "
                  f"dD {df - dF:6.2f}  dAIC {2 - (df - dF):+7.2f}  "
                  f"dBIC {np.log(data.n_data) - (df - dF):+7.2f}  T_d,ml {tF[6]:5.1f} K", flush=True)

btb_fixed, btb_free = TD.bin_to_bin_model(TD.T_D_FIXED), TD.bin_to_bin_model()
for m in MASKS:
    for mode in ('EE', 'BB'):
        z = np.load(C.result_path('btb', m['key'], mode))
        wd = TD.load_wishart(m['key'], mode, BANDS)
        paper = wishart_path('btb', m['key'], mode)
        paper = np.load(paper)['percentiles'][:, 1] if os.path.exists(paper) else None
        for b in range(wd.n_bins):
            data = TD.Data(wd.select_bins(b))
            start = None if paper is None else TD.from_paper(btb_fixed, paper[b])
            tf, df, tF, dF = compare(btb_fixed, btb_free, data, z['theta_ml'][b],
                                     float(z['deviance_ml'][b]), start)
            rows.append(dict(kind='btb', mask=m['key'], dataset='QJ+WP+Pl', mode=mode, bin=b,
                             ell_eff=float(wd.ell_eff[b]), n_data=data.n_data,
                             k_fixed=btb_fixed.n_par, k_free=btb_free.n_par, dev_fixed=df,
                             dev_free=dF, theta_fixed=tf, theta_free=tF))
        r = [x for x in rows if x['kind'] == 'btb' and x['mask'] == m['key'] and x['mode'] == mode]
        dD = np.array([x['dev_fixed'] - x['dev_free'] for x in r])
        print(f"btb {m['key']:9s} {mode}: dD per bandpower {np.round(dD, 1)}; combined dAIC "
              f"{2 * len(dD) - dD.sum():+.1f}, dBIC {len(dD) * np.log(len(dD) * r[0]['n_data']) - dD.sum():+.1f}",
              flush=True)

out = C.comparison_path()
cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
cols['dAIC'] = 2.0 * (cols['k_free'] - cols['k_fixed']) - (cols['dev_fixed'] - cols['dev_free'])
cols['dBIC'] = (cols['k_free'] - cols['k_fixed']) * np.log(cols['n_data']) - (cols['dev_fixed'] - cols['dev_free'])
np.savez(out, param_names=np.array(TD.PARAM_NAMES), **cols)
print(f'  -> {out}')
