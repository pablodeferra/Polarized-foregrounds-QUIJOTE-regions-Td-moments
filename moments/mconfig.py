"""
Settings of the fits with moment expansion (moments/).
======================================================

Every script of moments/ imports this module before anything else.  It
imports ../t_dust/tdconfig (noise term, HMDM by default; the paper's python/
and python/wishart/ on the path, only read) and puts t_dust/ on the path: the
moment models are built on the model with T_d free (td_model.py), whose
results are the baseline of the comparison.

Outputs, all inside moments/: results/wishart_hmdm/ (results/wishart/ for the
simulation noise).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
T_DUST_DIR = os.path.join(ROOT, 't_dust')
if T_DUST_DIR not in sys.path:
    sys.path.insert(1, T_DUST_DIR)        # after the script's own folder

import tdconfig  # noqa: E402
import config  # noqa: E402,F401

_TAG = tdconfig._TAG
RESULTS_DIR = os.path.join(HERE, 'results', 'wishart' + _TAG)

# the three sets of moments: synchrotron only, dust only, both
MOMENT_SETS = ('sync', 'dust', 'both')


def result_path(kind, key, mode, moments, binning='dl20', tag=''):
    """Fit with the moments `moments` ('sync', 'dust' or 'both'); kind 'pl' or 'btb'."""
    return os.path.join(RESULTS_DIR, f'{kind}_wishart_mom_{moments}_{key}_{mode}{tag}_{binning}.npz')


def comparison_path(binning='dl20'):
    """Maxima of the likelihood of every model, AIC and BIC (compare_models.py)."""
    return os.path.join(RESULTS_DIR, f'model_comparison_moments_{binning}.npz')
