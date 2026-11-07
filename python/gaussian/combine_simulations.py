#%%
"""
Joins the pieces of run_simulations.py into the files of all N_SIM realizations.
================================================================================

    python combine_simulations.py [--masks all] [--delete]

For every mask, kind (noise, skyplusnoise) and binning, adds the partial sums
results/spectra/partial_<mask>_<kind>_<binning>_k<first>-<last>.fits (which
must cover realizations 1..N_SIM exactly once, with no overlap or gap) and
writes the mean and standard deviation (population) of
results/spectra/spectra_avg_std_<mask>_100_<kind>_<binning>.fits, as a single
run of run_simulations.py.  --delete removes the pieces afterwards.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import functions as F  # noqa: E402
from config import BANDS, BINNINGS, N_SIM, SPECTRA_DIR, mask_names, sims_path  # noqa: E402

MODES = ('EE', 'BB')

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--masks', default='all', help='comma-separated mask keys (default: all)')
ap.add_argument('--delete', action='store_true', help='remove the pieces after combining them')
args = ap.parse_args()

for key in mask_names(args.masks):
    for kind in ('noise', 'skyplusnoise'):
        for name in BINNINGS:
            files = sorted(glob.glob(os.path.join(SPECTRA_DIR, f'partial_{key}_{kind}_{name}_k*.fits')))
            if not files:
                print(f'  {key} {kind} {name}: no pieces, skipped', flush=True)
                continue
            ranges = sorted((int(m.group(1)), int(m.group(2))) for f in files
                            for m in [re.search(r'_k(\d+)-(\d+)\.fits$', f)])
            expected, nxt = [], 1
            for first, last in ranges:
                if first != nxt:
                    break
                expected.append((first, last))
                nxt = last + 1
            if nxt != N_SIM + 1 or len(expected) != len(ranges):
                raise SystemExit(f'{key} {kind} {name}: the pieces {ranges} do not cover 1..{N_SIM} exactly once')
            bands = BANDS
            tot = {}
            for f in files:
                with fits.open(f) as hdul:
                    ell = {c: np.asarray(hdul[f'{bands[0]}_{bands[0]}'].data[c], float) for c in ('ell1', 'ell2', 'ell_eff')}
                    for a in bands:
                        for b in bands:
                            d = hdul[f'{a}_{b}'].data
                            for col in [f'{m}_{s}' for m in MODES for s in ('SUM', 'SUM2')]:
                                tot[a, b, col] = tot.get((a, b, col), 0.0) + np.asarray(d[col], float)
            F.write_pairs(sims_path(key, kind, name), bands, ell,
                          lambda i, j: {f'{m}_{s}': v for m in MODES for s, v in (
                              ('MEAN', tot[BANDS[i], BANDS[j], f'{m}_SUM'] / N_SIM),
                              ('STD', np.sqrt(np.clip(tot[BANDS[i], BANDS[j], f'{m}_SUM2'] / N_SIM
                                                      - (tot[BANDS[i], BANDS[j], f'{m}_SUM'] / N_SIM) ** 2, 0, None))))},
                          header={'NREAL': N_SIM, 'SIMKIND': kind, 'MASK': key, 'BINNING': name})
            if args.delete:
                for f in files:
                    os.remove(f)
