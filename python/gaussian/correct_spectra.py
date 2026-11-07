#%%
"""
Corrected spectra and their uncertainties, for the Gaussian likelihood.
=======================================================================

    python correct_spectra.py [--masks all]

From the data spectra (compute_spectra.py) and the mean and standard
deviation of the simulations (run_simulations.py), dl20, for every band pair:

  1. the noise N_l is subtracted from the auto-spectra (the noise of different
     bands is uncorrelated, so the cross-spectra carry no bias): the HMDM
     spectrum or, with SPECIFIC_REGIONS_NOISE=sims (config.NOISE), the mean of
     the noise simulations;
  2. the Planck 2018 CMB spectrum, times the beams and pixel window, is subtracted;
  3. the beams and pixel window are deconvolved and K_CMB^2 -> K_RJ^2;
  4. sigma = sqrt(sigma_S+N^2 + sigma_N^2) for the auto-spectra (the scatter of
     the subtracted noise estimate is included) and sigma_S+N for the
     cross-spectra, with the same deconvolution and units, from the simulations
     whatever the noise N_l.

Writes results/spectra/corrected_power_spectra_<mask>_dl20_hmdm.fits (with sims,
..._dl20.fits; columns EE_SPECTRUM, EE_ERROR, BB_SPECTRUM, BB_ERROR, in
mK^2_RJ).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import functions as F  # noqa: E402
from config import BANDS, NOISE, corrected_path, mask_names, sims_path, spectra_path  # noqa: E402

MODES = ('EE', 'BB')


def correct_spectra(key, noise=NOISE):
    """noise: N_l subtracted from the auto-spectra, 'sims' or 'hmdm' (default config.NOISE)."""
    cols = ['ell1', 'ell2', 'ell_eff'] + list(MODES)
    raw = F.read_pairs(spectra_path(key), BANDS, cols)
    stats = {kind: F.read_pairs(sims_path(key, kind), BANDS, [f'{m}_{s}' for m in MODES for s in ('MEAN', 'STD')])
             for kind in ('skyplusnoise', 'noise')}
    if noise == 'hmdm':
        hmdm = F.read_pairs(spectra_path(key, hmdm=True), BANDS, list(MODES))
        bias = {p: {m: v[m] for m in MODES} for p, v in hmdm.items()}
    else:
        bias = {p: {m: v[f'{m}_MEAN'] for m in MODES} for p, v in stats['noise'].items()}

    ell_eff = raw[f'{BANDS[0]}_{BANDS[0]}']['ell_eff']
    wpix = F.pixel_window(ell_eff)
    beam = {(b, m): F.beam(b, ell_eff, m) for b in BANDS for m in MODES}
    unit = {b: F.unit_conversion(b) for b in BANDS}
    cmb = {m: F.cmb_spectrum(ell_eff, m) for m in MODES}               # K^2

    out = {}
    for i, b1 in enumerate(BANDS):
        for b2 in BANDS[i:]:
            key_ = f'{b1}_{b2}'
            out[key_] = {}
            for m in MODES:
                phys = beam[(b1, m)] * beam[(b2, m)] * wpix * wpix       # beams x pixel window
                unit_factor = unit[b1] * unit[b2]
                cl = raw[key_][m]
                if b1 == b2:
                    cl = cl - bias[key_][m]
                cl = cl - cmb[m] * 1e6 * phys                            # K^2 -> mK^2, convolved
                out[key_][f'{m}_SPECTRUM'] = cl / phys * unit_factor
                if b1 == b2:
                    err = np.sqrt(stats['skyplusnoise'][key_][f'{m}_STD']**2
                                  + stats['noise'][key_][f'{m}_STD']**2)
                else:
                    err = stats['skyplusnoise'][key_][f'{m}_STD']
                out[key_][f'{m}_ERROR'] = np.abs(err / phys * unit_factor)
    ell = {c: raw[f'{BANDS[0]}_{BANDS[0]}'][c] for c in ('ell1', 'ell2', 'ell_eff')}
    F.write_pairs(corrected_path(key), BANDS, ell,
                  lambda i, j: out[f'{BANDS[i]}_{BANDS[j]}'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--masks', default='all', help='comma-separated mask keys (default: all)')
    for key in mask_names(ap.parse_args().masks):
        correct_spectra(key)
