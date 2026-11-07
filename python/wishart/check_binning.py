#%%
"""
Sect. 3.2: the baseline binning against Delta l = 30 and the equal-mode binning.
===============================================================================

    python check_binning.py

Prints the numbers behind Sect. 3.2, from the Wishart fits of the three
binnings (run_mcmc_*.py --binning dl20 | dl30 | eqmode):

  1. power-law fit: (median_alt - median_dl20) / sigma_dl20 of every parameter,
     mask, mode and dataset, and the largest ones, for all the masks, the
     Galactic cuts and the regions;
  2. bin-to-bin fit: inverse-variance weighted means of beta_s, beta_d and rho
     over the bandpowers, compared in units of the error of the baseline mean.

The refits with the band-averaged model are in check_bandaveraged_model.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import MASKS, wishart_path  # noqa: E402

PARAMS = ['A_s', 'alpha_s', 'beta_s', 'A_d', 'alpha_d', 'beta_d', 'rho']
DATASETS = [('QJ+WP+Pl', ''), ('WP+Pl', '_WPPl')]
ALTS = ('dl30', 'eqmode')


def summarize(samples):
    lo, med, hi = np.percentile(samples, [15.865, 50.0, 84.135], axis=0)
    return med, 0.5 * (hi - lo)


def keep(worst, key, z, who):
    """Keep the largest |z| for `key`."""
    if abs(z) > abs(worst.get(key, (0.0,))[0]):
        worst[key] = (z,) + who


# 1. power-law shifts ---------------------------------------------------------
print('1. power law: (alt - dl20)/sigma_dl20')
worst = {}
for m in MASKS:
    for alt in ALTS:
        for d, tag in DATASETS:
            for mode in ('EE', 'BB'):
                m0, s0 = summarize(np.load(wishart_path('pl', m['key'], mode, 'dl20', tag))['samples'])
                m1, _ = summarize(np.load(wishart_path('pl', m['key'], mode, alt, tag))['samples'])
                z = (m1 - m0) / s0
                print(f"  {alt:6s} {m['key']:9s} {d:9s} {mode}  "
                      + ' '.join(f'{p}={v:+5.2f}' for p, v in zip(PARAMS, z)))
                for i, p in enumerate(PARAMS):
                    for sel in ('all', m['kind'], f"{m['kind']} {d}"):
                        keep(worst, (alt, p, sel), z[i], (m['key'], d, mode))
for alt in ALTS:
    for sel in ('all', 'galcut', 'region', 'galcut QJ+WP+Pl', 'galcut WP+Pl',
                'region QJ+WP+Pl', 'region WP+Pl'):
        print(f'  max |shift| {alt} ({sel}): '
              + '  '.join(f'{p} {worst[(alt, p, sel)][0]:+.2f} '
                          f'({worst[(alt, p, sel)][1]} {worst[(alt, p, sel)][3]} '
                          f'{worst[(alt, p, sel)][2]})' for p in PARAMS))

# 2. bin-to-bin weighted means -------------------------------------------------
print('\n2. bin-to-bin: inverse-variance weighted means over the bandpowers')
idx = {'beta_s': 2, 'beta_d': 5, 'rho': 6}


def wmean(percentiles, i):
    """percentiles: (bandpower, 3, parameter), the 15.865, 50 and 84.135 percentiles."""
    lo, med, hi = percentiles[:, 0, i], percentiles[:, 1, i], percentiles[:, 2, i]
    w = 1.0 / (0.5 * (hi - lo)) ** 2
    return (w * med).sum() / w.sum(), 1.0 / np.sqrt(w.sum())


worst_b = {}
for m in MASKS:
    for mode in ('EE', 'BB'):
        ref = {p: wmean(np.load(wishart_path('btb', m['key'], mode))['percentiles'], i)
               for p, i in idx.items()}
        for alt in ALTS:
            ch = np.load(wishart_path('btb', m['key'], mode, alt))['percentiles']
            for p, i in idx.items():
                mu, _ = wmean(ch, i)
                z = (mu - ref[p][0]) / ref[p][1]
                for sel in ('all', m['kind']):
                    keep(worst_b, (alt, p, sel), z, (m['key'], mode))
for alt in ALTS:
    for sel in ('all', 'galcut', 'region'):
        print(f'  max |shift| {alt} ({sel}): ' + '  '.join(
            f'{p} {worst_b[(alt, p, sel)][0]:+.2f} ({worst_b[(alt, p, sel)][1]} '
            f'{worst_b[(alt, p, sel)][2]})' for p in idx))
