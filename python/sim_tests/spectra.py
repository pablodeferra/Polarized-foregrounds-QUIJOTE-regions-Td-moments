#%%
"""
Spectra of the PySM sky plus noise simulation k (simulation tests).
===================================================================

    python spectra.py --masks galcut10,nps --first 1 --last 50

For each mask and k = first..last, the EE and BB auto- and cross-spectra of
the 14 bands of the paper's PySM maps plus noise simulation k, dl20: the
realizations of run_simulations.py skyplusnoise (functions.coupled_spectra),
which keeps only their mean and standard deviation.  Writes, after every
realization (a failed job keeps what it finished),

    results/sim_tests/spectra/pysm_noise_<mask>_dl20_k<first>-<last>.npz

with spectra (n_k, n_pairs, 2, n_bins) in mK^2_CMB (pairs i <= j of
config.BANDS, modes EE and BB), k, bands, pairs, modes, ell1, ell2, ell_eff.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simlib as S  # noqa: E402  (sets the paths)
import healpy as hp  # noqa: E402
import numpy as np  # noqa: E402
import functions as F  # noqa: E402
from config import BANDS, N_SIM, mask_names, mask_path  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--masks', default='galcut10,nps', help='comma-separated mask keys')
ap.add_argument('--first', type=int, default=1)
ap.add_argument('--last', type=int, default=N_SIM)
args = ap.parse_args()
if not 1 <= args.first <= args.last <= N_SIM:
    raise SystemExit(f'need 1 <= first <= last <= {N_SIM}')

b = S.dl20()
for key in mask_names(args.masks):
    mask = hp.read_map(mask_path(key))
    w22 = F.coupling_workspace(mask, b)
    out = S.spectra_file(key, args.first, args.last)
    ks, spectra = [], []
    for k in range(args.first, args.last + 1):
        t0 = time.time()
        sp = F.decouple_spectra(F.coupled_spectra(BANDS, mask, kind='skyplusnoise', k=k), w22)
        ks.append(k)
        spectra.append([[sp[p][m] for m in S.MODES] for p in S.PAIRS])
        S.save_spectra(out, ks, np.array(spectra), b)
        print(f'[{key}] realization {k:3d} in {time.time() - t0:.0f} s', flush=True)
    print(f'  -> {out}', flush=True)
