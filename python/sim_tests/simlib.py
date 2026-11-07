"""
Shared pieces of the simulation tests (Sect. 3.6 and Appendix A.2).
===================================================================

First test: the paper's PySM sky (s1, d1, a1, f1, c1, beam-convolved; a1 and f1
are unpolarized) plus noise simulation 1, fitted with both likelihoods
(power law and bin to bin), against the reference of noiseless
single-component maps (reference.py).  Second test: the same sky plus noise
simulations k = 1..100 (the same 100 that give N_l and the Gaussian errors),
power law, to compare the scatter of the medians with the posterior widths.
All at dl20 with the 14 bands.

The PySM maps are evaluated at the central frequency nu0 of each band
(monochromatic), so their colour corrections are exactly 1 and their
K_CMB -> K_RJ factor is the exact one at nu0.  monochromatic() makes the
paper's code use these; the likelihoods, their data loading and
gaussian/correct_spectra.py are then used unchanged, reading the spectra of a
realization from a temporary file in the format of compute_spectra.py.
"""

import contextlib
import glob
import io
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ('..', '../wishart', '../gaussian'):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, _d)))

import functions as F  # noqa: E402
from config import BANDS, RESULTS_DIR, binning  # noqa: E402

SIM_DIR = os.path.join(RESULTS_DIR, 'sim_tests')
SPECTRA_DIR = os.path.join(SIM_DIR, 'spectra')
FITS_DIR = os.path.join(SIM_DIR, 'fits')
BINNING = 'dl20'
MODES = ('EE', 'BB')
PAIRS = [(i, j) for i in range(len(BANDS)) for j in range(i, len(BANDS))]


def monochromatic():
    """In this process: colour corrections 1 and the exact K_CMB -> K_RJ at nu0 for every band."""
    F.unit_conversion = lambda band: float(F.cmb_unit_conversion(F.freq_ghz(band)))
    ones = {b: (1.0, 0.0, 0.0) for b in BANDS}
    F.load_color_correction_polynomials = lambda: {'synch': dict(ones), 'dust': dict(ones)}


# ---------------------------------------------------------------------------
# Spectra of the realizations: one file per mask and range of k
# ---------------------------------------------------------------------------
def spectra_file(key, first, last):
    return os.path.join(SPECTRA_DIR, f'pysm_noise_{key}_{BINNING}_k{first:03d}-{last:03d}.npz')


def save_spectra(path, ks, spectra, b):
    """spectra (n_k, n_pairs, 2, n_bins) in mK^2_CMB, pairs PAIRS of BANDS, modes MODES."""
    ell = F.ell_columns(b)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, k=np.array(ks, int), spectra=np.asarray(spectra, float), bands=np.array(BANDS),
             pairs=np.array(PAIRS), modes=np.array(MODES), **ell)


def load_realization(key, k):
    """(spectra (n_pairs, 2, n_bins), ell columns) of realization k, from the file that holds it."""
    for path in sorted(glob.glob(os.path.join(SPECTRA_DIR, f'pysm_noise_{key}_{BINNING}_k*.npz'))):
        with np.load(path) as z:
            ks = list(z['k'])
            if k in ks:
                if list(z['bands']) != BANDS:
                    raise ValueError(f'{path}: bands {list(z["bands"])} are not {BANDS}')
                return z['spectra'][ks.index(k)], {c: z[c] for c in ('ell1', 'ell2', 'ell_eff')}
    raise FileNotFoundError(f'realization {k} of {key} is in no file of {SPECTRA_DIR}')


@contextlib.contextmanager
def realization_fits(key, k):
    """
    Path of a temporary FITS with the spectra of realization k, in the format of
    compute_spectra.py, and a free path next to it (for its corrected spectra).
    """
    spectra, ell = load_realization(key, k)
    index = {p: n for n, p in enumerate(PAIRS)}
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, 'spectra.fits')
        with contextlib.redirect_stdout(io.StringIO()):
            F.write_pairs(raw, BANDS, ell, lambda i, j: {m: spectra[index[i, j], c] for c, m in enumerate(MODES)})
        yield raw, os.path.join(tmp, 'corrected.fits')


def correct_spectra(key, raw_path, out_path):
    """gaussian/correct_spectra.py on the spectra raw_path (noise mean and errors of the paper's simulations)."""
    import correct_spectra as CS
    CS.spectra_path = lambda *_a, **_k: raw_path
    CS.corrected_path = lambda *_a, **_k: out_path
    with contextlib.redirect_stdout(io.StringIO()):
        CS.correct_spectra(key, noise='sims')


def dl20():
    return binning(BINNING)
