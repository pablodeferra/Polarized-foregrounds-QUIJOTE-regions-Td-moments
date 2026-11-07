#%%
"""
PySM sky maps of the fourteen bands, for the simulations of Sect. 3.6 and Appendix A.
=====================================================================================

    python make_pysm_maps.py

The PySM sky with synchrotron (s1), thermal dust (d1), AME (a1), free-free
(f1) and lensed CMB (c1), at N_side 512, evaluated at the frequency of each
band in data.py (monochromatic) and convolved with its beam (data/beams/; the
beam is taken as 1 beyond the multipoles of the beam file).  Units as the
data maps: mK_CMB for QUIJOTE and WMAP, K_CMB for Planck.  Writes
data.map_file(band, 'pysm').
"""

import os
import sys

import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import functions as F  # noqa: E402
from config import BANDS, NSIDE  # noqa: E402
from data import CHANNELS, map_file  # noqa: E402

LMAX = 3 * NSIDE - 1


def beam_lmax(band, mode):
    """Beam of the band for l = 0..LMAX, 1 outside the range of the beam file."""
    bl = np.asarray(F.beam_table(band, mode), dtype=float)
    return np.interp(np.arange(LMAX + 1, dtype=float), np.arange(len(bl), dtype=float), bl,
                     left=1.0, right=1.0)


sky = pysm3.Sky(nside=NSIDE, preset_strings=['s1', 'd1', 'a1', 'f1', 'c1'])
for band in BANDS:
    ch = CHANNELS[band]
    nu = ch['freq']
    iqu = sky.get_emission(nu).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu)).value
    alm = hp.map2alm(np.asarray(iqu, dtype=np.float64, order='C'), lmax=LMAX, pol=True, iter=3)
    alm = [hp.almxfl(a, beam_lmax(band, mode)) for a, mode in zip(alm, ('TT', 'EE', 'BB'))]
    iqu = hp.alm2map(alm, nside=NSIDE, pol=True, lmax=LMAX)
    if ch['experiment'] == 'Planck':
        iqu, unit = iqu / 1e6, 'K_CMB'
    else:
        iqu, unit = iqu / 1e3, 'mK_CMB'
    out = map_file(band, 'pysm')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    hp.write_map(out, iqu, dtype=np.float32, overwrite=True, column_names=['I', 'Q', 'U'],
                 extra_header=[('FREQ_GHZ', float(nu.to_value(u.GHz)), 'Band center frequency [GHz]'),
                               ('NSIDE', NSIDE, 'HEALPix NSIDE'), ('UNIT', unit, 'Units'),
                               ('EXPERIM', ch['experiment'], 'Experiment'), ('BAND', band, 'Band name')])
    print(f'[{band}] -> {out}', flush=True)
