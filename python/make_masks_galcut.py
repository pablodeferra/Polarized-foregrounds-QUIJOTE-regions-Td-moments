#%%
"""
Masks of the three Galactic cuts.
=================================

    python make_masks_galcut.py

The pixels of the QUIJOTE-MFI footprint (config.FOOTPRINT) at Galactic
latitude |b| > 10, 15 and 20 deg, apodized with NaMaster's C2 scheme and a
scale of 5 deg.  Writes data/masks/mask_galcut<cut>_apodC2_5.fits.
"""

import healpy as hp
import numpy as np
import pymaster as nmt

from config import FOOTPRINT, NSIDE, mask_path

CUTS = (10, 15, 20)            # deg
APOD_SCALE = 5.0               # deg

footprint = hp.read_map(FOOTPRINT)
latitude = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)[1]
for cut in CUTS:
    galcut = np.where(np.abs(latitude) < cut, 0., 1.)
    mask = np.ones_like(galcut)
    mask *= galcut
    mask *= footprint
    mask = nmt.mask_apodization(mask, APOD_SCALE, apotype='C2')
    hp.write_map(mask_path(f'galcut{cut}'), mask, dtype='float64', overwrite=True)
    print(f'|b| > {cut} deg: mean of the apodized mask {np.mean(mask):.4f} -> {mask_path(f"galcut{cut}")}')
