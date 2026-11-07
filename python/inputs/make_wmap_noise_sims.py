#%%
"""
Noise simulations of the WMAP bands, for Appendix A and the simulation tests.
=============================================================================

    python make_wmap_noise_sims.py

White noise in every pixel, from the number of observations of the band map
(its second extension: N_obs, N_QQ, N_QU, N_UU) and the noise per
observation sigma0 of the band: sigma_I = sigma0_I / sqrt(N_obs) and, for Q and
U, sigma0_QU / sqrt(N_eff) with N_eff = N_QQ - N_QU^2/N_UU (Q) and
N_UU - N_QU^2/N_QQ (U).  Writes N_SIM realizations per band,
data.map_file(band, 'noise', k) for k = 1..N_SIM.

The noise simulations of the other bands are those of each experiment: PR4
for Planck (inputs/download_planck_noise_sims.py) and the QUIJOTE-MFI
collaboration simulations.
"""

import os
import sys

import healpy as hp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import N_SIM, NSIDE, WMAP_BANDS  # noqa: E402
from data import map_file  # noqa: E402

# noise per observation (sigma0) of the band maps, mK, I and Q/U (WMAP 9-year)
SIGMA0 = {'23': (1.429, 1.435), '33': (1.466, 1.472), '41': (2.188, 2.197),
          '61': (3.131, 3.141), '94': (6.544, 6.560)}
SEED = 42

npix = hp.nside2npix(NSIDE)
for band in WMAP_BANDS:
    nobs_i, nobs_q, nobs_qu, nobs_u = hp.read_map(map_file(band, 'map'), hdu=2, field=[0, 1, 2, 3])
    sigma_i = SIGMA0[band][0] / np.sqrt(nobs_i)
    sigma_q = SIGMA0[band][1] / np.sqrt(nobs_q - nobs_qu**2 / nobs_u)
    sigma_u = SIGMA0[band][1] / np.sqrt(nobs_u - nobs_qu**2 / nobs_q)
    rng = np.random.RandomState(SEED + int(band))
    for k in range(1, N_SIM + 1):
        noise = np.zeros([3, npix])
        noise[0] = rng.normal(0, sigma_i, npix)
        noise[1] = rng.normal(0, sigma_q, npix)
        noise[2] = rng.normal(0, sigma_u, npix)
        out = map_file(band, 'noise', k)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        hp.write_map(out, noise, dtype=np.float64, overwrite=True)
    print(f'[{band}] {N_SIM} realizations -> {os.path.dirname(map_file(band, "noise", 1))}', flush=True)
