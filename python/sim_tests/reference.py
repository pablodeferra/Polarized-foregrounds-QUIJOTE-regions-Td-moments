#%%
"""
Reference parameters of the simulation tests, from noiseless single-component maps.
===================================================================================

    python reference.py [--masks all]

The components of the paper's PySM maps alone, at N_side 512 with neither
beams nor noise, in uK_RJ: synchrotron (s1) at 11.1, 23.0 and 28.4 GHz and
thermal dust (d1) at 143, 217 and 353 GHz (the bands where each dominates;
23.0 GHz rather than the 22.8 GHz of WMAP K because A_s is defined there, and
the s1 template is the 23 GHz map).  Their EE and BB spectra in each mask,
with E/B purification and the dl20 bandpowers as the data, divided by the
pixel window squared at l_eff (the likelihood model includes it in W_b).
From them, separately for EE and BB, all in log C (every bandpower and pair
with the same weight; the spectra have no noise, but they are not exact power
laws):

  power law   A_s, alpha_s   straight line log C(23 x 23) = log A_s + alpha_s log(l/80)
              A_d, alpha_d   the same with C(353 x 353)
              beta_s         from the six spectra of the three synchrotron maps, A_s
                             and alpha_s fixed: log C(ij) - log[A_s (l/80)^alpha_s]
                             = beta_s log(nu_i nu_j / 23^2)
              beta_d         the same with the dust maps and the modified black body
                             of the model (T_d = 19.6 K), A_d and alpha_d fixed
              rho            mean over the bandpowers of C(23 synch x 353 dust)
                             / [sqrt(A_s A_d) (l/80)^((alpha_s + alpha_d)/2)]
  bin to bin  per bandpower: A_s = C(23 x 23), A_d = C(353 x 353), beta_s and beta_d
              as above with that bandpower only, rho = C(sd) / sqrt(C(ss) C(dd));
              alpha_s = alpha_d = 0, as in the fits.

PySM s1 is a power law in every pixel (beta_s varies across the sky); d1 is
a modified black body with its own beta_d and T_d in every pixel (T_d of
14-31 K), so beta_d is the effective index of the model (T_d = 19.6 K) over
143-353 GHz.  Writes results/sim_tests/reference.npz (amplitudes in uK^2_RJ,
as the Wishart fits): pl (n_masks, 2, 7), btb (n_masks, 2, n_bins, 7), the
spectra, and the rms of the log residuals of each fit.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simlib as S  # noqa: E402  (sets the paths)
import healpy as hp  # noqa: E402
import numpy as np  # noqa: E402
import pymaster as nmt  # noqa: E402
import functions as F  # noqa: E402
import wishart_likelihood as W  # noqa: E402
from config import NSIDE, mask_names, mask_path  # noqa: E402

SYNC_FREQS = (11.1, 23.0, 28.4)       # GHz
DUST_FREQS = (143.0, 217.0, 353.0)
NU_S, NU_D, ELL_REF, T_D = W.NU_REF_SYNC, W.NU_REF_DUST, W.ELL_REF, W.T_DUST
MAPS = [('s', nu) for nu in SYNC_FREQS] + [('d', nu) for nu in DUST_FREQS]
I_S23, I_D353 = MAPS.index(('s', NU_S)), MAPS.index(('d', NU_D))
PAIRS = ([(i, j) for i in range(3) for j in range(i, 3)]                  # synchrotron
         + [(i, j) for i in range(3, 6) for j in range(i, 6)]             # dust
         + [(I_S23, I_D353)])


def component_maps():
    """Q, U (uK_RJ) of each map of MAPS."""
    import pysm3
    import pysm3.units as u
    out = []
    for comp, preset in (('s', 's1'), ('d', 'd1')):
        sky = pysm3.Sky(nside=NSIDE, preset_strings=[preset])
        for c, nu in MAPS:
            if c == comp:
                iqu = sky.get_emission(nu * u.GHz).to(u.uK_RJ).value
                out.append(np.asarray(iqu[1:], dtype=np.float64))
    return out


def log_sed_dust(nu, beta):
    return np.log(F.mbb_scaling_KRJ(nu, nu0_GHz=NU_D, beta=beta, T_d=T_D))


def slope(x, y):
    """Least-squares slope of y = s x (through the origin) and the rms residual."""
    x, y = np.ravel(x), np.ravel(y)
    s = np.sum(x * y) / np.sum(x * x)
    return s, np.sqrt(np.mean((y - s * x) ** 2))


def line(x, y):
    """Least-squares y = a + s x: (a, s, rms residual)."""
    s, a = np.polyfit(x, y, 1)
    return a, s, np.sqrt(np.mean((y - a - s * x) ** 2))


def fit_reference(C, ell):
    """
    C: {(i, j): bandpowers} of one mode, divided by the pixel window squared.
    Returns the power-law parameters (7), the bin-to-bin ones (n_bins, 7) and
    the rms residuals of the power-law steps.
    """
    for p, c in C.items():
        if p != (I_S23, I_D353) and not np.all(c > 0):
            raise ValueError(f'non-positive noiseless spectrum {MAPS[p[0]]} x {MAPS[p[1]]}: {c}')
    x_ell = np.log(ell / ELL_REF)
    sync = [p for p in PAIRS if p[1] < 3]
    dust = [p for p in PAIRS if p[0] >= 3]
    nu = {i: MAPS[i][1] for i in range(len(MAPS))}
    xs = {p: np.log(nu[p[0]] * nu[p[1]] / NU_S ** 2) for p in sync}
    xd = {p: np.log(nu[p[0]] * nu[p[1]] / NU_D ** 2) for p in dust}
    cd = {p: log_sed_dust(nu[p[0]], 0.0) + log_sed_dust(nu[p[1]], 0.0) for p in dust}

    # power law
    logA_s, alpha_s, rms_s = line(x_ell, np.log(C[I_S23, I_S23]))
    logA_d, alpha_d, rms_d = line(x_ell, np.log(C[I_D353, I_D353]))
    beta_s, rms_bs = slope([np.full_like(ell, xs[p]) for p in sync],
                           [np.log(C[p]) - logA_s - alpha_s * x_ell for p in sync])
    beta_d, rms_bd = slope([np.full_like(ell, xd[p]) for p in dust],
                           [np.log(C[p]) - logA_d - alpha_d * x_ell - cd[p] for p in dust])
    A_s, A_d = np.exp(logA_s), np.exp(logA_d)
    m_x = np.sqrt(A_s * A_d) * np.exp(0.5 * (alpha_s + alpha_d) * x_ell)
    rho = float(np.mean(C[I_S23, I_D353] / m_x))
    pl = np.array([A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho])

    # bin to bin
    btb = []
    for b in range(len(ell)):
        a_s, a_d = C[I_S23, I_S23][b], C[I_D353, I_D353][b]
        bs, _ = slope([xs[p] for p in sync], [np.log(C[p][b] / a_s) for p in sync])
        bd, _ = slope([xd[p] for p in dust], [np.log(C[p][b] / a_d) - cd[p] for p in dust])
        btb.append([a_s, 0.0, bs, a_d, 0.0, bd, C[I_S23, I_D353][b] / np.sqrt(a_s * a_d)])
    return pl, np.array(btb), np.array([rms_s, rms_d, rms_bs, rms_bd])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--masks', default='all', help='comma-separated mask keys (default: all)')
    keys = mask_names(ap.parse_args().masks)

    t0 = time.time()
    maps = component_maps()
    print(f'PySM maps in {time.time() - t0:.0f} s', flush=True)
    b = S.dl20()
    ell = np.asarray(b.get_effective_ells(), float)
    pix2 = F.pixel_window(ell) ** 2
    pl, btb, rms, spectra = [], [], [], []
    for key in keys:
        mask = hp.read_map(mask_path(key))
        w22 = F.coupling_workspace(mask, b)
        fields = [F.spin2_field(mask, qu) for qu in maps]
        cl = {p: w22.decouple_cell(nmt.compute_coupled_cell(fields[p[0]], fields[p[1]])) for p in PAIRS}
        per_mode = []
        for c, m in ((0, 'EE'), (3, 'BB')):
            per_mode.append(fit_reference({p: v[c] / pix2 for p, v in cl.items()}, ell))
            s = per_mode[-1][0]
            print(f'[{key} {m}] A_s = {s[0]:.4g}  alpha_s = {s[1]:+.3f}  beta_s = {s[2]:+.3f}  '
                  f'A_d = {s[3]:.4g}  alpha_d = {s[4]:+.3f}  beta_d = {s[5]:.3f}  rho = {s[6]:+.3f}', flush=True)
        pl.append([x[0] for x in per_mode])
        btb.append([x[1] for x in per_mode])
        rms.append([x[2] for x in per_mode])
        spectra.append([[cl[p][c] / pix2 for p in PAIRS] for c in (0, 3)])
    out = os.path.join(S.SIM_DIR, 'reference.npz')
    os.makedirs(S.SIM_DIR, exist_ok=True)
    np.savez(out, masks=np.array(keys), modes=np.array(S.MODES), params=np.array(W.PARAM_NAMES),
             pl=np.array(pl), btb=np.array(btb), rms_log_residual=np.array(rms),
             rms_names=np.array(['A_s, alpha_s', 'A_d, alpha_d', 'beta_s', 'beta_d']),
             spectra=np.array(spectra), pairs=np.array(PAIRS), maps=np.array([f'{c}{nu:g}' for c, nu in MAPS]),
             ell_eff=ell, amplitude_unit='uK^2_RJ')
    print(f'  -> {out}')


if __name__ == '__main__':
    main()
