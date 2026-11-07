"""
Settings of the fits with the dust temperature free (t_dust/).
==============================================================

Every script of t_dust/ imports this module before anything else.  It

  - sets the noise term N_b of the Wishart model to the HMDM spectra
    (SPECIFIC_REGIONS_NOISE=hmdm, as in the paper), unless
    the variable is already set: SPECIFIC_REGIONS_NOISE=sims gives the mean of
    the noise simulations;
  - puts the paper's python/ and python/wishart/ on the path.  Their config,
    data, functions and wishart_likelihood are only imported: the paper's code
    and results are never written;
  - defines the outputs, all inside t_dust/: results/wishart_hmdm/
    (results/wishart/ for the simulation noise).

The spectra, noise spectra, masks and f_sky are those of the paper
(config.spectra_path, config.noise_path, config.MASK).
"""

import os
import sys

os.environ.setdefault('SPECIFIC_REGIONS_NOISE', 'hmdm')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# after the script's own folder, so that its modules come first (python/wishart/
# has scripts with the same names)
for _p in (os.path.join(ROOT, 'python'), os.path.join(ROOT, 'python', 'wishart')):
    if _p not in sys.path:
        sys.path.insert(1, _p)

import config  # noqa: E402

if config.NOISE != os.environ['SPECIFIC_REGIONS_NOISE']:
    raise RuntimeError('config was imported before tdconfig: import tdconfig first')

_TAG = '' if config.NOISE == 'sims' else '_hmdm'

DATA_DIR = os.path.join(HERE, 'data')
CC_SURFACE = os.path.join(DATA_DIR, 'cc_dust_hfi_surface.fits')
RESULTS_DIR = os.path.join(HERE, 'results', 'wishart' + _TAG)


def result_path(kind, key, mode, binning='dl20', tag=''):
    """Fit with T_d free; kind 'pl' or 'btb', tag '' (QJ+WP+Pl) or '_WPPl'."""
    return os.path.join(RESULTS_DIR, f'{kind}_wishart_td_{key}_{mode}{tag}_{binning}.npz')


def comparison_path(binning='dl20'):
    """Maximum likelihoods of T_d fixed and free, and their AIC and BIC (compare_models.py)."""
    return os.path.join(RESULTS_DIR, f'model_comparison_td_{binning}.npz')
