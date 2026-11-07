#%%
"""
Simulations: the Gaussian uncertainties and the noise term of the simulation tests.
==================================================================================

    python run_simulations.py noise [--masks all] [--first 1 --last 100]
    python run_simulations.py skyplusnoise [--masks all] [--first 1 --last 100]

For k = 1..100, the EE and BB spectra of noise simulation k of every band
alone (noise), or of the PySM sky plus noise simulation k (skyplusnoise),
processed like the data (functions.compute_spectra) and binned as in
config.BINNINGS (each realization is read and its pseudo-C_l computed once,
then decoupled for every binning).  Only their mean and standard deviation
(population, over the 100) are written:

    results/spectra/spectra_avg_std_<mask>_100_<kind>_<binning>.fits

(columns EE_MEAN, EE_STD, BB_MEAN, BB_STD, in mK^2_CMB).  The standard deviations
are the Gaussian uncertainties (correct_spectra.py, dl20); the noise means are
the N_l of the fits to the simulations (sim_tests/fit.py).
A few minutes per realization; NaMaster uses OMP_NUM_THREADS threads.

With --first/--last (e.g. 1-50 and 51-100) only those realizations are run and
their sums are written to results/spectra/partial_<mask>_<kind>_<binning>_k<first>-<last>.fits;
combine_simulations.py then joins the pieces into the files above, exactly as a
single run of all 100.  A job that fails then loses only its piece.
"""

import argparse
import os
import sys
import time

import healpy as hp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import functions as F  # noqa: E402
from config import BANDS, BINNINGS, N_SIM, binning, mask_names, mask_path, sims_chunk_path, sims_path  # noqa: E402

MODES = ('EE', 'BB')

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('kind', choices=['noise', 'skyplusnoise'])
ap.add_argument('--masks', default='all', help='comma-separated mask keys (default: all)')
ap.add_argument('--first', type=int, default=1)
ap.add_argument('--last', type=int, default=N_SIM)
args = ap.parse_args()
if not 1 <= args.first <= args.last <= N_SIM:
    raise SystemExit(f'need 1 <= first <= last <= {N_SIM}')
n_run = args.last - args.first + 1
full = (args.first, args.last) == (1, N_SIM)

bins = {name: binning(name) for name in BINNINGS}
pairs = [(i, j) for i in range(len(BANDS)) for j in range(i, len(BANDS))]
for key in mask_names(args.masks):
    mask = hp.read_map(mask_path(key))
    workspaces = {name: F.coupling_workspace(mask, b) for name, b in bins.items()}
    s1 = {name: 0.0 for name in bins}            # running sums of x and x^2 [pair, mode, bandpower]
    s2 = {name: 0.0 for name in bins}
    for k in range(args.first, args.last + 1):
        t0 = time.time()
        coupled = F.coupled_spectra(BANDS, mask, kind=args.kind, k=k)
        for name, w22 in workspaces.items():
            spectra = F.decouple_spectra(coupled, w22)
            x = np.array([[np.asarray(spectra[p][m], float) for m in MODES] for p in pairs])
            s1[name], s2[name] = s1[name] + x, s2[name] + x ** 2
        print(f'[{key} {args.kind}] realization {k:3d} in {time.time() - t0:.0f} s', flush=True)
    index = {p: n for n, p in enumerate(pairs)}
    for name, b in bins.items():
        if full:
            mean = s1[name] / N_SIM
            std = np.sqrt(np.clip(s2[name] / N_SIM - mean ** 2, 0, None))
            F.write_pairs(sims_path(key, args.kind, name), BANDS, F.ell_columns(b),
                          lambda i, j: {f'{m}_{s}': v[index[i, j], c] for c, m in enumerate(MODES)
                                        for s, v in (('MEAN', mean), ('STD', std))},
                          header={'NREAL': N_SIM, 'SIMKIND': args.kind, 'MASK': key, 'BINNING': name})
        else:
            F.write_pairs(sims_chunk_path(key, args.kind, name, args.first, args.last), BANDS, F.ell_columns(b),
                          lambda i, j: {f'{m}_{s}': v[index[i, j], c] for c, m in enumerate(MODES)
                                        for s, v in (('SUM', s1[name]), ('SUM2', s2[name]))},
                          header={'FIRST': args.first, 'LAST': args.last, 'SIMKIND': args.kind, 'MASK': key,
                                  'BINNING': name})
