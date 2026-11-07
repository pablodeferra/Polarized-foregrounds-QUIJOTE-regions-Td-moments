#%%
"""
Masks of the four regions: Fan, NPS, North and South.
=====================================================

    python make_masks_regions.py

The regions are lists of pixels of an N_side = 8 HEALPix grid (RING
ordering).  The pixels of no region (the excluded zone: the Galactic plane
around W51 and Cygnus, Cas A, the Crab and IC 443) are a list as well, and
North is every pixel in no other list.  Each list is upgraded to N_side 256
and smoothed with a 15 deg FWHM Gaussian, every pixel is assigned to the
region (or the excluded zone) with the largest smoothed value, so that the
regions do not overlap and all their borders are equally smooth, and the
result is downgraded to N_side 512, multiplied by the QUIJOTE-MFI footprint
and apodized with NaMaster's C2 scheme and a scale of 5 deg.  The excluded
zone gets no mask.  Writes data/masks/mask_<region>_apodC2_5.fits.
"""

import healpy as hp
import numpy as np
import pymaster as nmt

from config import FOOTPRINT, NSIDE, mask_path

NSIDE_LOW = 8                      # grid of the pixel lists
NSIDE_SMOOTH = 256                 # resolution of the smoothing
SMOOTH_FWHM = np.deg2rad(15.0)
APOD_SCALE = 5.0                   # deg

nps_pixels = [
    275, 242, 210, 178, 146, 113, 84, 60, 59,
    39, 24, 25, 42, 62, 86, 115, 147, 179, 212,
    243, 211, 114, 61, 85, 41, 40, 148, 87,
    12, 23, 83, 276, 244, 180, 116, 11, 22, 38,
    63, 4, 13, 3, 10, 21, 36, 37
]

fan_pixels = [
    411, 442, 443, 444, 445, 446, 447, 415,
    383, 351, 319, 318, 317, 316, 315, 347, 346, 313,
    379, 380, 381, 382, 412, 413, 414, 348, 349,
    350, 284, 314, 285, 286, 287,
    252, 253, 254, 476, 477, 508, 378
]

south_pixels = [
    468, 469, 470, 471, 472, 473, 474, 475,
    478, 479, 480, 512, 544, 576, 608, 545, 513, 482,
    481, 511, 510, 509, 507, 506, 505,
    504, 503, 502, 501, 500, 533, 534, 535, 536, 537,
    538, 539, 540, 541, 542, 543, 565, 566, 567, 568,
    569, 570, 571, 572, 573, 574, 575, 607, 606, 605,
    604, 603, 602, 601, 600, 599, 598, 630, 631, 632,
    633, 634, 635, 636, 637, 638, 639, 662, 663, 664,
    665, 666, 667, 668, 693, 692, 691, 690, 714, 715,
    716, 717, 669, 694
]

excluded_pixels = [
    # anticentre plane beyond the Fan: the Crab (Tau A, in 448, with its wings
    # in 416) and IC 443 (353)
    416, 448, 417, 384, 418, 386, 449, 385, 353, 354,
    320,
    # first-quadrant plane at l < 62 deg (W51 is in 372) and 352 (l = 180, b = +5)
    307, 308, 309, 339, 340, 341, 371, 372, 373, 404, 405, 436, 437,
    352,
    # Cygnus and Cas A
    438, 407,
    375, 374, 344, 343, 342,
    310, 311, 376, 406, 408, 345, 312, 439, 440,
    409,
    410, 377, 441
]

npix_low = hp.nside2npix(NSIDE_LOW)
north_pixels = sorted(set(range(npix_low)) - set(nps_pixels) - set(fan_pixels)
                      - set(south_pixels) - set(excluded_pixels))

# the excluded zone competes with the regions (last) but gets no mask
REGIONS = {'nps': nps_pixels, 'fan': fan_pixels, 'south': south_pixels, 'north': north_pixels,
           'excluded': excluded_pixels}

smoothed = []
for pixels in REGIONS.values():
    m = np.zeros(npix_low)
    m[pixels] = 1
    m = hp.ud_grade(m, NSIDE_SMOOTH, order_in='RING', order_out='RING')
    smoothed.append(hp.smoothing(m, fwhm=SMOOTH_FWHM))
winner = np.argmax(np.array(smoothed), axis=0)

footprint = hp.read_map(FOOTPRINT)
for k, name in enumerate(REGIONS):
    if name == 'excluded':
        continue
    m = hp.ud_grade((winner == k).astype(float), NSIDE, order_in='RING', order_out='RING')
    mask = nmt.mask_apodization(footprint * m, APOD_SCALE, apotype='C2')
    hp.write_map(mask_path(name), mask, dtype='float64', overwrite=True)
    print(f'{name}: mean of the apodized mask {np.mean(mask):.4f} -> {mask_path(name)}')
