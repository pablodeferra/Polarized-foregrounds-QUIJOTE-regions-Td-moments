"""
Settings of the analysis: paths, frequency bands, binnings and sky masks.
=========================================================================

Every script reads its settings from here, so the spectra and the fits cannot
drift apart.

The large inputs (sky maps, HMDM maps, PySM maps and noise simulations) are
not in the repository: MAPS_DIR points to them, data/maps/ by default or the
directory in the environment variable SPECIFIC_REGIONS_MAPS.  Their layout is
described in README.md.

The noise term N_l of both likelihoods is NOISE: 'hmdm' (the spectra of the
half-mission difference maps, as in the paper, the default) or 'sims' (the mean
of the spectra of the noise simulations), chosen with the environment variable
SPECIFIC_REGIONS_NOISE.  The fits and corrected spectra of 'hmdm' go to
results/wishart_hmdm/, results/gaussian_hmdm/ and
corrected_power_spectra_<mask>_dl20_hmdm.fits, those of 'sims' to
results/wishart/, results/gaussian/ and corrected_power_spectra_<mask>_dl20.fits.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOISE = os.environ.get('SPECIFIC_REGIONS_NOISE', 'hmdm')
if NOISE not in ('sims', 'hmdm'):
    raise ValueError(f"SPECIFIC_REGIONS_NOISE must be 'sims' or 'hmdm', not {NOISE!r}")
_N = '' if NOISE == 'sims' else '_hmdm'

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, 'data')
MAPS_DIR = os.environ.get('SPECIFIC_REGIONS_MAPS', os.path.join(DATA_DIR, 'maps'))
BEAM_DIR = os.path.join(DATA_DIR, 'beams')
CC_DIR = os.path.join(DATA_DIR, 'color_corrections')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
# Planck 2018 best-fitting LCDM spectrum (EE and lensing BB), from the PLA
CMB_FILE = os.path.join(DATA_DIR, 'cmb', 'COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-'
                        'lensing-minimum-theory_R3.01.txt')
# QUIJOTE-MFI DR1 analysis mask (low declinations, satellite band, NCP)
FOOTPRINT = os.path.join(MASK_DIR, 'mask_quijote_ncp_lowdec_satband_nside512.fits')

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(ROOT, 'results')
SPECTRA_DIR = os.path.join(RESULTS_DIR, 'spectra')
WISHART_DIR = os.path.join(RESULTS_DIR, 'wishart' + _N)
GAUSS_DIR = os.path.join(RESULTS_DIR, 'gaussian' + _N)

NSIDE = 512
LMAX = 2 * NSIDE - 1

# ---------------------------------------------------------------------------
# Frequency bands (data.py holds their properties).  WP+Pl is the WMAP and
# Planck set, QJ+WP+Pl adds the two QUIJOTE-MFI bands; the tag names the files.
# ---------------------------------------------------------------------------
QUIJOTE_BANDS = ['11', '17']
WMAP_BANDS = ['23', '33', '41', '61', '94']
PLANCK_BANDS = ['30', '44', '70', '100', '143', '217', '353']
BANDS = QUIJOTE_BANDS + WMAP_BANDS + PLANCK_BANDS
DATASETS = {'WP+Pl': (WMAP_BANDS + PLANCK_BANDS, '_WPPl'),
            'QJ+WP+Pl': (BANDS, '')}

# ---------------------------------------------------------------------------
# Binnings over 30 <= l <= 209: first and last multipole of each bandpower,
# both included.  dl20 is the baseline; dl30 and eqmode (bandpowers with the
# same number of modes to within 10 per cent) are the alternatives of Sect. 3.2.
# ---------------------------------------------------------------------------
BINNINGS = {
    'dl20': ([30, 50, 70, 90, 110, 130, 150, 170, 190],
             [49, 69, 89, 109, 129, 149, 169, 189, 209]),
    'dl30': ([30, 60, 90, 120, 150, 180],
             [59, 89, 119, 149, 179, 209]),
    'eqmode': ([30, 72, 98, 118, 135, 150, 164, 176, 188, 199],
               [71, 97, 117, 134, 149, 163, 175, 187, 198, 209]),
}

# ---------------------------------------------------------------------------
# Masks, in the order of the paper.  fsky is the effective sky
# fraction w2^2/w4 of the apodized mask (w_n the mean of its n-th power over
# the sphere), which sets the number of modes of the Wishart likelihood.
# ---------------------------------------------------------------------------
MASKS = [
    dict(key='galcut10', label=r'$|b|>10^\circ$', kind='galcut', fsky=0.2950),
    dict(key='galcut15', label=r'$|b|>15^\circ$', kind='galcut', fsky=0.2626),
    dict(key='galcut20', label=r'$|b|>20^\circ$', kind='galcut', fsky=0.2317),
    dict(key='fan', label='Fan', kind='region', fsky=0.0354),
    dict(key='nps', label='NPS', kind='region', fsky=0.0374),
    dict(key='north', label='North', kind='region', fsky=0.1440),
    dict(key='south', label='South', kind='region', fsky=0.0815),
]
MASK = {m['key']: m for m in MASKS}
GALCUTS = [m for m in MASKS if m['kind'] == 'galcut']
REGIONS = [m for m in MASKS if m['kind'] == 'region']

# Gaussian-likelihood comparison (Appendix A): number of simulations
N_SIM = 100


def binning(name):
    """NmtBin of a binning, checked against BINNINGS."""
    import pymaster as nmt
    ell1, ell2 = BINNINGS[name]
    # NmtBin.from_edges excludes the upper edge
    b = nmt.NmtBin.from_edges(list(ell1), [e + 1 for e in ell2])
    got = [(b.get_ell_min(i), b.get_ell_max(i)) for i in range(b.get_n_bands())]
    if got != list(zip(ell1, ell2)):
        raise RuntimeError(f'NaMaster built the bandpowers {got}, not {list(zip(ell1, ell2))}')
    return b


def mask_names(arg):
    """The masks named in a comma-separated command-line argument ('all': every mask)."""
    keys = list(MASK) if arg == 'all' else arg.split(',')
    unknown = [k for k in keys if k not in MASK]
    if unknown:
        raise SystemExit(f'unknown mask(s) {unknown}; choose from {list(MASK)}')
    return keys


# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------
def mask_path(key):
    return os.path.join(MASK_DIR, f'mask_{key}_apodC2_5.fits')


def spectra_path(key, binning_name='dl20', hmdm=False):
    """Data (hmdm=False) or HMDM spectra of a mask."""
    return os.path.join(SPECTRA_DIR, f"power_spectra_{key}{'_hmdm' if hmdm else ''}_{binning_name}.fits")


def sims_path(key, kind, binning_name='dl20'):
    """Mean and standard deviation of the N_SIM simulations; kind 'skyplusnoise' or 'noise'."""
    return os.path.join(SPECTRA_DIR, f'spectra_avg_std_{key}_{N_SIM}_{kind}_{binning_name}.fits')


def sims_chunk_path(key, kind, binning_name, first, last):
    """Partial sums of simulations first..last (run_simulations.py --first --last)."""
    return os.path.join(SPECTRA_DIR, f'partial_{key}_{kind}_{binning_name}_k{first:03d}-{last:03d}.fits')


def noise_path(key, binning_name='dl20'):
    """N_l of the likelihoods (NOISE): the HMDM spectra or the mean of the noise simulations."""
    if NOISE == 'hmdm':
        return spectra_path(key, binning_name, hmdm=True)
    return sims_path(key, 'noise', binning_name)


def corrected_path(key):
    return os.path.join(SPECTRA_DIR, f'corrected_power_spectra_{key}_dl20{_N}.fits')


def wishart_path(kind, key, mode, binning_name='dl20', tag=''):
    """Wishart chain; kind 'pl' (power law) or 'btb' (bin to bin), tag '' or '_WPPl'."""
    return os.path.join(WISHART_DIR, f'{kind}_wishart_{key}_{mode}{tag}_{binning_name}.npz')


def gauss_path(kind, key, mode, tag=''):
    """Gaussian-likelihood fit summary; kind and tag as in wishart_path."""
    return os.path.join(GAUSS_DIR, f'{kind}_gauss_{key}_{mode}{tag}_dl20.npz')
