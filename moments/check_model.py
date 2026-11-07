#%%
"""
Checks of the moment models.
============================

    python check_model.py

  1. With every w = 0 the model matrix of each moment set equals that of the
     model with T_d free (t_dust/td_model.py), for every mask, mode and dataset.
  2. The analytic derivatives of the colour-corrected SEDs (colour-correction
     derivatives included) equal central finite differences, over a grid of
     beta_s, beta_d and T_d.
  3. A moment term changes the model matrix as expected: with w_d_bb > 0 the
     auto-spectra grow by W_d w (de_d/dbeta_d)^2.
  4. Time of one evaluation of the posterior.
"""

import time

import mconfig as MC  # noqa: F401

import numpy as np  # noqa: E402

import moments_model as MM  # noqa: E402
import td_model as TD  # noqa: E402
from config import DATASETS, MASKS  # noqa: E402

# 1. w = 0 is the model with T_d free
x8 = np.array([1.3e-3, -3.0, -3.1, 2.4e-3, -2.4, 1.55, 19.6, 0.15])
worst = 0.0
for m in MASKS:
    for dataset, (bands, _) in DATASETS.items():
        for mode in ('EE', 'BB'):
            data = TD.Data(TD.load_wishart(m['key'], mode, bands))
            M0 = TD.power_law_model().foreground(TD.power_law_model().params(x8), data)
            for mom in MC.MOMENT_SETS:
                model = MM.power_law_model(mom)
                M = model.foreground(model.params(MM.start_from(model, x8)), data)
                worst = max(worst, float(np.max(np.abs(M - M0))))
print(f'1. w = 0: largest |C_fg(moments) - C_fg(T_d free)| = {worst:.1e} uK^2')

# 2. derivatives against finite differences
data = TD.Data(TD.load_wishart('galcut10', 'BB', DATASETS['QJ+WP+Pl'][0]))
err_s, err_b, err_T = 0.0, 0.0, 0.0
for beta_s in (-3.6, -3.1, -2.6):
    hs = 1e-5
    fd = (TD.sed_synchrotron(beta_s + hs, data) - TD.sed_synchrotron(beta_s - hs, data)) / (2 * hs)
    an = MM.sed_derivatives_synchrotron(beta_s, data)
    err_s = max(err_s, float(np.max(np.abs(an - fd) / np.abs(fd).max())))
for beta_d in (0.8, 1.2, 1.55, 2.0, 2.6):
    for T_d in (11.0, 15.0, 19.6, 25.0, 38.0):
        hb, hT = 1e-5, 1e-4
        fdb = (TD.sed_dust(beta_d + hb, T_d, data) - TD.sed_dust(beta_d - hb, T_d, data)) / (2 * hb)
        fdT = (TD.sed_dust(beta_d, T_d + hT, data) - TD.sed_dust(beta_d, T_d - hT, data)) / (2 * hT)
        anb, anT = MM.sed_derivatives_dust(beta_d, T_d, data)
        err_b = max(err_b, float(np.max(np.abs(anb - fdb) / np.abs(fdb).max())))
        err_T = max(err_T, float(np.max(np.abs(anT - fdT) / np.abs(fdT).max())))
print(f'2. analytic - finite-difference derivatives (relative to the largest): d/dbeta_s {err_s:.1e}, '
      f'd/dbeta_d {err_b:.1e}, d/dT_d {err_T:.1e}')
db, dT = MM.sed_derivatives_dust(1.55, 19.6, data)
print('   de_d/dbeta_d at 1.55, 19.6 K:', ' '.join(f'{b}:{v:+.3f}' for b, v in zip(data.wd.bands, db)))
print('   de_d/dT_d [1/K]            :', ' '.join(f'{b}:{v:+.4f}' for b, v in zip(data.wd.bands, dT)))

# 3. one moment term
model = MM.power_law_model('dust')
x = MM.start_from(model, x8)
x[model.free.index('w_d_bb')] = 0.04
dM = model.foreground(model.params(x), data) - TD.power_law_model().foreground(TD.power_law_model().params(x8), data)
expect = (x8[3] * data.ell ** x8[4])[:, None] * 0.04 * db[None, :] ** 2
print(f"3. w_d_bb = 0.04: auto-spectra change {np.max(np.abs(np.einsum('bii->bi', dM) - expect)):.1e} "
      f'from W_d w (de/dbeta)^2 (largest term {expect.max():.2e} uK^2)')

# 4. timing
for mom in ('sync', 'dust', 'both'):
    model = MM.power_law_model(mom)
    x = MM.start_from(model, x8)
    model.log_posterior(x, data)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        model.log_posterior(x, data)
    print(f'4. {mom:4s}: {(time.perf_counter() - t0) / n * 1e6:.0f} us per log posterior')
