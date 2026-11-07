#%%
"""
Joins the pieces of fit.py into one file per kind, likelihood and mask.
=======================================================================

    python combine_fits.py [--delete]

For every kind (pl, btb), likelihood and mask with pieces
results/sim_tests/fits/<kind>_<likelihood>_<mask>_k<first>-<last>.npz,
concatenates their realizations in the order of k (no k may appear in two
pieces) into results/sim_tests/fits/<kind>_<likelihood>_<mask>.npz.  A
later run adds new pieces to that file; a realization of a piece replaces
the one already there (a rerun).  Prints the realizations missing in
1..max(k).  --delete removes the pieces afterwards.
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simlib as S  # noqa: E402
import numpy as np  # noqa: E402

PER_K = ('k', 'percentiles', 'mean', 'std', 'theta_map', 'chi2_reduced')

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--delete', action='store_true', help='remove the pieces after combining them')
args = ap.parse_args()

groups = {}
for path in sorted(glob.glob(os.path.join(S.FITS_DIR, '*_k[0-9][0-9][0-9]-[0-9][0-9][0-9].npz'))):
    groups.setdefault(re.sub(r'_k\d{3}-\d{3}\.npz$', '', os.path.basename(path)), []).append(path)

for name, pieces in groups.items():
    out = os.path.join(S.FITS_DIR, f'{name}.npz')
    parts = [dict(np.load(p)) for p in pieces]
    new_k = np.concatenate([z['k'] for z in parts])
    if len(set(new_k)) != len(new_k):
        raise SystemExit(f'{name}: realizations {sorted(k for k in set(new_k) if list(new_k).count(k) > 1)} '
                         f'appear in two of {pieces}')
    if os.path.exists(out):
        old = dict(np.load(out))
        keep = ~np.isin(old['k'], new_k)
        parts.append({**old, **{key: old[key][keep] for key in PER_K}})
    for z in parts[1:]:
        for key in ('params', 'modes', 'lengths', 'nwalkers', 'nsteps'):
            if not np.array_equal(z[key], parts[0][key]):
                raise SystemExit(f'{name}: {key} differs between the pieces (or the existing {out})')
    joined = {key: np.concatenate([z[key] for z in parts]) for key in PER_K}
    ks = joined['k']
    order = np.argsort(ks)
    joined = {key: v[order] for key, v in joined.items()}
    np.savez(out, **{**{k: v for k, v in parts[0].items() if k not in PER_K}, **joined})
    missing = sorted(set(range(1, int(ks.max()) + 1)) - set(ks))
    print(f'{name}: {len(ks)} realizations' + (f', missing {missing}' if missing else '') + f'\n  -> {out}')
    if args.delete:
        for p in pieces:
            os.remove(p)
