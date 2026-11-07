#%%
"""
Beam transfer functions of the WMAP bands, from those of their DAs.
===================================================================

    python make_wmap_beams.py

Each WMAP band map combines several differencing assemblies (DAs).  Its beam
is the weighted average of the DA beams (data/beams/wmap_da/, from LAMBDA),
with weights proportional to N_obs / sigma0^2: the number of observations of
the DA summed over the sky (from its 9-year map) over the square of its
polarization noise per observation.  The fractional errors are averaged with
the same weights.  Writes data/beams/wmap_ampl_bl_<band>_9yr_v5p1.txt in the
format of the DA files.
"""

import os
import sys

import healpy as hp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import BEAM_DIR, MAPS_DIR  # noqa: E402

DAS = {'K': ['K1'], 'Ka': ['Ka1'], 'Q': ['Q1', 'Q2'], 'V': ['V1', 'V2'],
       'W': ['W1', 'W2', 'W3', 'W4']}
# polarization noise per observation of each DA, mK (WMAP 9-year)
SIGMA0_QU = {'K1': 1.456, 'Ka1': 1.490, 'Q1': 2.280, 'Q2': 2.164, 'V1': 3.348, 'V2': 2.979,
             'W1': 5.940, 'W2': 6.612, 'W3': 6.983, 'W4': 6.840}
DA_MAP = os.path.join(MAPS_DIR, 'WMAP', 'wmap_iqumap_r9_9yr_{da}_v5.fits')
DA_BEAM = os.path.join(BEAM_DIR, 'wmap_da', 'wmap_ampl_bl_{da}_9yr_v5p1.txt')


def read_beam(path):
    header, rows = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                header.append(line.rstrip())
            else:
                rows.append([float(x) for x in line.split()])
    return header, np.array(rows)


for band, das in DAS.items():
    w = np.array([float(np.sum(hp.read_map(DA_MAP.format(da=da), field=3, hdu=1))) / SIGMA0_QU[da] ** 2
                  for da in das])
    w = w / np.sum(w)
    w = w / np.sum(w)                   # (normalized twice, as the beams of the paper)
    beams = [read_beam(DA_BEAM.format(da=da)) for da in das]
    header = beams[0][0]
    ell = beams[0][1][:, 0]
    assert all(np.array_equal(b[:, 0], ell) for _, b in beams)
    avg = np.zeros_like(beams[0][1])
    avg[:, 0] = ell
    for wi, (_, b) in zip(w, beams):
        avg[:, 1] += wi * b[:, 1]
        avg[:, 2] += wi * b[:, 2]
    header = [f'# Beam Transfer Function (amplitude) for frequency band {band},'
              if 'differencing assembly' in line else line for line in header]
    out = os.path.join(BEAM_DIR, f'wmap_ampl_bl_{band}_9yr_v5p1.txt')
    with open(out, 'w') as f:
        for line in header:
            f.write(line + '\n')
        for row in avg:
            f.write(f'{int(row[0]):>5} {row[1]:>14.8f} {row[2]:>14.8f}\n')
    print(f'{band}: DA weights {dict(zip(das, np.round(w, 4)))} -> {out}')
