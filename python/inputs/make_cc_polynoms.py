#%%
"""
Colour-correction polynomials of the fourteen bands.
====================================================

    python make_cc_polynoms.py

Each band has two polynomials, one for each foreground; functions.py
evaluates them at alpha = 2 + beta (the spectral index in flux density for a
spectral index beta in K_RJ):

  synchrotron, every band     the fastcc power-law polynomial
                              a0 + a1 alpha + a2 alpha^2 (option 3,
                              'frequencies_v3': QUIJOTE-MFI DR1, WMAP 2015,
                              Planck LFI 2015 and HFI 2018 with bandpass shift),
                              copied below
  dust, QUIJOTE, WMAP, LFI    the same power-law polynomial, at alpha = 2 + beta_d
  dust, HFI                   a quadratic in beta_d fitted to the corrections of
                              a modified black body at T_d = 19.6 K in the
                              bandpass-shifted table
                              c_td_10-40_beta_hfi_bps_pr3.fits (variant [1],
                              the one interpcc reads; variant [0] is empty);
                              functions.py converts it to a polynomial in alpha

The corrections refer to the fastcc reference frequencies, the `freq` of
data.py.

Writes data/color_corrections/cc_polynoms.fits (table CC_POLYNOMS).
"""

import os
import sys

import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import BANDS, CC_DIR  # noqa: E402
from data import CHANNELS  # noqa: E402

T_DUST = 19.6
HFI_BANDS = ['100', '143', '217', '353']

# fastcc 'frequencies_v3' [a0, a1, a2] of each band
FASTCC = {
    '11': [0.9821236098857145, 0.011920227628586971, -0.0015347819996889738],
    '17': [1.0067957381497863, -0.0019621413830759947, -0.000715949539348768],
    '23': [0.972902, 0.0190469, -0.00276464],
    '33': [0.983787, 0.0117567, -0.00183716],
    '41': [0.996854, 0.00496893, -0.00181359],
    '61': [0.980322, 0.0143631, -0.00223596],
    '94': [0.984848, 0.0112743, -0.00164595],
    '30': [1.00513, 0.00301399, -0.00300699],
    '44': [0.994769, 0.00596703, -0.00173626],
    '70': [0.989711, 0.0106943, -0.00328671],
    '100': [0.99868757, -0.00512203, -0.00428818],
    '143': [1.0125835, 0.00767883, -0.00468418],
    '217': [0.98670965, -0.01582359, -0.00362294],
    '353': [0.98479307, -0.0174746, -0.00318638],
}

# (experiment, band, component, variable, a0, a1, a2, T_d)
rows = [(CHANNELS[b]['experiment'], b, 'synch', 'alpha', *FASTCC[b], np.nan) for b in BANDS]
rows += [(CHANNELS[b]['experiment'], b, 'dust_nonHFI', 'alpha', *FASTCC[b], np.nan)
         for b in BANDS if b not in HFI_BANDS]

with fits.open(os.path.join(CC_DIR, 'c_td_10-40_beta_hfi_bps_pr3.fits')) as hdul:
    # columns BANDS, TD, BETA and CA (corrections [variant, band, beta, T_d])
    table_bands, td, beta, cc = hdul[1].data[0][0], hdul[1].data[0][1], hdul[1].data[0][2], hdul[1].data[0][3]
    beta_grid = np.asarray(beta, dtype=float).ravel()
    for b in HFI_BANDS:
        cc_band = cc[1][list(table_bands).index(b)]          # [1]: the Planck HFI corrections
        interp = RegularGridInterpolator((td, beta), cc_band.T, method='linear')
        values = np.array([interp([T_DUST, x])[()][0] for x in beta_grid], dtype=float)
        a2, a1, a0 = map(float, np.polyfit(beta_grid, values, deg=2))
        rows.append(('Planck', b, 'dust_HFI', 'beta', a0, a1, a2, T_DUST))

columns = [fits.Column(name='EXPERIM', format='20A', array=np.array([r[0] for r in rows])),
           fits.Column(name='BAND', format='8A', array=np.array([r[1] for r in rows])),
           fits.Column(name='COMP', format='12A', array=np.array([r[2] for r in rows])),
           fits.Column(name='VAR', format='8A', array=np.array([r[3] for r in rows]))]
columns += [fits.Column(name=name, format='D', array=np.array([r[k] for r in rows], dtype=float))
            for k, name in ((4, 'A0'), (5, 'A1'), (6, 'A2'), (7, 'TDUST'))]
out = os.path.join(CC_DIR, 'cc_polynoms.fits')
fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name='CC_POLYNOMS')]
             ).writeto(out, overwrite=True)
print(f'{len(rows)} polynomials -> {out}')
