#%%
"""
Half-mission difference maps (HMDM), the noise estimate of every band.
======================================================================

    python make_hmdm.py [--bands all]

The difference of two halves of the data, in which the sky cancels, scaled so
that its noise is that of the full map:

  QUIJOTE-MFI   the two half-survey maps of the DR1, (h1 - h2)/w with
                w = sqrt((w1 + w2)(1/w1 + 1/w2)) from their weight maps WEI_*
  WMAP          the maps of years 1-4 and 5-9, (h1 - h2) sqrt(20/81).  They are
                first built from the yearly maps of each differencing assembly
                (DA), coadded with their numbers of observations
  Planck        the two half-ring maps, data.map_file(band, 'half_1' / 'half_2'),
                (h1 - h2)/w with w from their variance maps; at HFI, pixels with bad data
                in either half are set to zero

Writes data.map_file(band, 'hmdm') for the fourteen bands (WMAP: also the
two halves, data.map_file(band, 'half_1' / 'half_2')).
"""

import os
import sys

import healpy as hp
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import BANDS, MAPS_DIR  # noqa: E402
from data import CHANNELS, map_file  # noqa: E402

# Differencing assemblies of each WMAP band, and the yearly maps of each DA
WMAP_DAS = {'K': ['K1'], 'Ka': ['Ka1'], 'Q': ['Q1', 'Q2'], 'V': ['V1', 'V2'],
            'W': ['W1', 'W2', 'W3', 'W4']}
WMAP_YEARLY = os.path.join(MAPS_DIR, 'WMAP', 'yearly', 'wmap_iqumap_r9_yr{year}_{da}_v5.fits')


def wmap_years(letter, first, last):
    """I, Q, U of a WMAP band over years first..last: each year the DAs, then the years, weighted by N_obs."""
    maps, nobs = [], []
    for year in range(first, last + 1):
        sum_map = sum_nobs = None
        for da in WMAP_DAS[letter]:
            I, Q, U, n = hp.read_map(WMAP_YEARLY.format(year=year, da=da), field=[0, 1, 2, 3], hdu=1)
            m = np.array([I, Q, U])
            if sum_map is None:
                sum_map, sum_nobs = m * n, n.copy()
            else:
                sum_map += m * n
                sum_nobs += n
        maps.append(sum_map / sum_nobs)
        nobs.append(sum_nobs)
    sum_map = sum_nobs = None
    for m, n in zip(maps, nobs):
        if sum_map is None:
            sum_map, sum_nobs = m * n, n.copy()
        else:
            sum_map += m * n
            sum_nobs += n
    return sum_map / sum_nobs


def hmdm_quijote(band):
    comp = 'IQU'
    h1 = hp.read_map(map_file(band, 'half_1'), [c + '_STOKES' for c in comp], nest=False)
    h2 = hp.read_map(map_file(band, 'half_2'), [c + '_STOKES' for c in comp], nest=False)
    w1 = hp.read_map(map_file(band, 'half_1'), ['WEI_' + c for c in comp], nest=False)
    w2 = hp.read_map(map_file(band, 'half_2'), ['WEI_' + c for c in comp], nest=False)
    w1[np.isnan(w1)] = 0
    w2[np.isnan(w2)] = 0
    w1[w1 < 0] = 0
    w2[w2 < 0] = 0
    w = np.sqrt((w1 + w2) * (1. / w1 + 1. / w2))
    hmdm = (h1 - h2) / w
    hmdm[w1 * w2 == 0] = 0
    return hmdm


def hmdm_wmap(band):
    letter = CHANNELS[band]['letter']
    for key, (first, last) in (('half_1', (1, 4)), ('half_2', (5, 9))):
        hp.write_map(map_file(band, key), wmap_years(letter, first, last), overwrite=True)
    h1 = hp.read_map(map_file(band, 'half_1'), field=[0, 1, 2])
    h2 = hp.read_map(map_file(band, 'half_2'), field=[0, 1, 2])
    return (h1 - h2) * np.sqrt(20. / 81.)          # 4 and 5 years


def hmdm_lfi(band):
    h1 = hp.read_map(map_file(band, 'half_1'), field=[0, 1, 2])
    h2 = hp.read_map(map_file(band, 'half_2'), field=[0, 1, 2])
    sigma1 = np.sqrt(hp.read_map(map_file(band, 'half_1'), field=[4, 7, 9]))      # II, QQ, UU
    sigma2 = np.sqrt(hp.read_map(map_file(band, 'half_2'), field=[4, 7, 9]))
    w = np.sqrt((1 / sigma1**2 + 1 / sigma2**2) * (sigma1**2 + sigma2**2))
    return (h1 - h2) / w


def hmdm_hfi(band):
    path1, path2 = map_file(band, 'half_1'), map_file(band, 'half_2')
    h1 = np.array(hp.read_map(path1, field=[0, 1, 2]))
    h2 = np.array(hp.read_map(path2, field=[0, 1, 2]))
    cov1 = np.array(hp.read_map(path1, field=[4, 7, 9]))
    cov2 = np.array(hp.read_map(path2, field=[4, 7, 9]))
    with fits.open(path1) as hdul:
        bad_data = hdul[1].header.get('BAD_DATA', -1.6375e+30)

    # pixels with bad data (or absurd values) in either half
    bad_threshold = np.abs(bad_data) * 0.01
    bad = ((np.abs(h1 - bad_data) < bad_threshold) | (np.abs(h2 - bad_data) < bad_threshold)
           | (np.abs(h1) > 1e10) | (np.abs(h2) > 1e10))
    h1[bad] = 0.0
    h2[bad] = 0.0
    bad_cov1 = (np.abs(cov1 - bad_data) < bad_threshold) | bad
    bad_cov2 = (np.abs(cov2 - bad_data) < bad_threshold) | bad
    cov1[bad_cov1] = 1e20                          # huge variance: no weight
    cov2[bad_cov2] = 1e20
    var1 = np.sqrt(np.abs(cov1) + 1e-20)**2
    var2 = np.sqrt(np.abs(cov2) + 1e-20)**2

    inv1 = np.where(var1 > 1e-30, 1.0 / var1, 0.0)
    inv2 = np.where(var2 > 1e-30, 1.0 / var2, 0.0)
    w = np.sqrt((inv1 + inv2) * (var1 + var2))
    w = np.where(w > 1e-30, w, 1e30)
    hmdm = (h1 - h2) / w
    hmdm[bad | bad_cov1 | bad_cov2] = 0.0
    return hmdm


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--bands', default='all', help='comma-separated bands (default: all)')
    arg = ap.parse_args().bands
    for band in (BANDS if arg == 'all' else arg.split(',')):
        exp = CHANNELS[band]['experiment']
        if exp == 'QUIJOTE':
            hmdm = hmdm_quijote(band)
        elif exp == 'WMAP':
            hmdm = hmdm_wmap(band)
        elif int(band) < 100:
            hmdm = hmdm_lfi(band)
        else:
            hmdm = hmdm_hfi(band)
        hp.write_map(map_file(band, 'hmdm'), hmdm, overwrite=True)
        print(f'[{band}] -> {map_file(band, "hmdm")}', flush=True)
