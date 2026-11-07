#%%
"""
Are the moments worth their parameters?  AIC and BIC of the five models.
========================================================================

    python compare_models.py

For every power-law fit (mask x dataset x mode), the maximum of the likelihood
within the uniform priors (deviance D_ml) of the five nested models

  paper   T_d = 19.6 K, no moments        k = 7   } t_dust/compare_models.py
  td      T_d free                        k = 8   }
  sync    td + synchrotron moment         k = 9   }
  dust    td + three dust moments         k = 11  } moments/run_mcmc_power_law.py
  both    td + the four moments           k = 12  }

and AIC = D_ml + 2k, BIC = D_ml + k ln N (N = n_data: 945 for QJ+WP+Pl).
t_dust/README.md explains the reading: lower is better, and differences beyond
2 (positive), 6 (strong), 10 (very strong) matter.

The models are nested (paper in td, td in sync and dust, both of these in
both), so a larger model can never have a higher D_ml.  When its maximization
ended above a smaller model, it is redone from the smaller model's maximum
(the extra w at 0) and the better maximum is kept.

Needs t_dust/compare_models.py first.  Writes
moments/results/wishart_hmdm/model_comparison_moments_dl20.npz: one row per
fit with dev_, k_, aic_, bic_ and theta_<model> (the maximum, over
param_names; NaN for the parameters a model lacks), and, for the rows in order,
pl_devbin (row x model x bandpower: the deviance of each bandpower at the
maximum of each model) and pl_ell_eff.
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import mconfig as MC  # noqa: E402

import numpy as np  # noqa: E402

import moments_model as MM  # noqa: E402
import tdconfig  # noqa: E402
import td_model as TD  # noqa: E402
from config import DATASETS, MASKS  # noqa: E402

MODELS = ('paper', 'td', 'sync', 'dust', 'both')
# a model and the smaller models nested in it
NESTED = {'sync': ('td',), 'dust': ('td',), 'both': ('sync', 'dust')}
ALL_NAMES = MM.param_names('both')


def improve(model, data, dev, theta, names, candidates):
    """
    The maximum of `model` (deviance dev at theta over names), redone from each
    nested maximum in `candidates` [(dev, theta, names)] that is better.
    """
    for d_small, t_small, n_small in candidates:
        if d_small < dev - 1e-6:
            x, d = TD.maximize(model, data, MM.start_from(model, t_small, n_small), gaussian=False, seed=3)
            if d < dev:
                dev, theta, names = d, model.full(x)[0], model.names
    return dev, theta, names


td_cmp = np.load(tdconfig.comparison_path())


def td_row(kind, key, dataset, mode, b):
    sel = np.flatnonzero((td_cmp['kind'] == kind) & (td_cmp['mask'] == key) & (td_cmp['dataset'] == dataset)
                         & (td_cmp['mode'] == mode) & (td_cmp['bin'] == b))[0]
    return {k: td_cmp[k][sel] for k in td_cmp.files if k != 'param_names'}


def compare(kind, key, dataset, mode, b, data, results, factory):
    """D_ml and k of the five models for one fit; results: {moments: (theta_ml, D_ml)}."""
    t = td_row(kind, key, dataset, mode, b)
    best = {'paper': (float(t['dev_fixed']), t['theta_fixed'], TD.PARAM_NAMES),
            'td': (float(t['dev_free']), t['theta_free'], TD.PARAM_NAMES)}
    k = {'paper': int(t['k_fixed']), 'td': int(t['k_free'])}
    for mom in ('sync', 'dust', 'both'):
        model = factory(mom)
        theta, dev = results[mom]
        best[mom] = improve(model, data, dev, theta, model.names, [best[s] for s in NESTED[mom]])
        k[mom] = model.n_par
    row = dict(kind=kind, mask=key, dataset=dataset, mode=mode, bin=b,
               ell_eff=np.nan if b < 0 else float(data.wd.ell_eff[0]), n_data=data.n_data)
    for m in MODELS:
        row[f'dev_{m}'], row[f'k_{m}'] = best[m][0], k[m]
        p = dict(zip(best[m][2], map(float, best[m][1])))
        row[f'theta_{m}'] = np.array([p.get(n, np.nan) for n in ALL_NAMES])
    return row


def deviance_per_bin(name, row, data):
    """The deviance of each bandpower at the maximum of the power-law model `name`."""
    model = TD.power_law_model() if name in ('paper', 'td') else MM.power_law_model(name)
    p = dict(zip(ALL_NAMES, row[f'theta_{name}']))
    return model.deviance(np.array([p[n] for n in model.free]), data, per_bin=True)


rows, pl_devbin, pl_ell = [], [], []
for m in MASKS:
    for dataset, (bands, tag) in DATASETS.items():
        for mode in ('EE', 'BB'):
            data = TD.Data(TD.load_wishart(m['key'], mode, bands))
            res = {}
            for mom in ('sync', 'dust', 'both'):
                z = np.load(MC.result_path('pl', m['key'], mode, mom, 'dl20', tag))
                res[mom] = (z['theta_ml'], float(z['deviance_ml']))
            r = compare('pl', m['key'], dataset, mode, -1, data, res, MM.power_law_model)
            rows.append(r)
            pl_devbin.append([deviance_per_bin(x, r, data) for x in MODELS])
            pl_ell.append(data.wd.ell_eff)
            assert np.allclose(np.sum(pl_devbin[-1], axis=1), [r[f'dev_{x}'] for x in MODELS], atol=1e-6)
            print(f"pl  {m['key']:9s} {dataset:8s} {mode}: D_ml "
                  + '  '.join(f"{x} {r[f'dev_{x}']:.2f}" for x in MODELS), flush=True)

cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
for m in MODELS:
    cols[f'aic_{m}'] = cols[f'dev_{m}'] + 2.0 * cols[f'k_{m}']
    cols[f'bic_{m}'] = cols[f'dev_{m}'] + cols[f'k_{m}'] * np.log(cols['n_data'])
out = MC.comparison_path()
np.savez(out, models=np.array(MODELS), param_names=np.array(ALL_NAMES),
         pl_devbin=np.array(pl_devbin), pl_ell_eff=np.array(pl_ell), **cols)
print(f'  -> {out}')
