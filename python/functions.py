"""
Functions shared by the Wishart and the Gaussian analyses.
==========================================================

  maps and spectra   Q/U maps of a band and their EE and BB auto- and
                     cross-spectra with NaMaster (E and B purification)
  spectra files      FITS files with one table per band pair
  instrument         beams, pixel window, K_CMB -> K_RJ conversion
  sky model          CMB spectrum, colour corrections, dust SED
"""

import os

import healpy as hp
import numpy as np
import pymaster as nmt
from astropy.io import fits
from scipy.constants import c, h, k

from config import CC_DIR, CMB_FILE, LMAX, NSIDE
from data import CHANNELS, beam_file, freq_ghz, map_file

HFI_BANDS = ('100', '143', '217', '353')


# ===========================================================================
# Maps and spectra
# ===========================================================================
def read_qu(path, experiment):
    """
    Q and U of a map at N_side 512, unseen pixels set to zero.  Planck maps
    are converted from K to mK, the unit of the QUIJOTE and WMAP maps, and
    downgraded from N_side 1024 or 2048.
    """
    m = hp.read_map(path, field=[1, 2])
    m = np.where(m == hp.UNSEEN, 0, m)
    if experiment == 'Planck':
        m *= 1e3
        if hp.get_nside(m[0]) != NSIDE:
            m = np.array([hp.ud_grade(x, nside_out=NSIDE) for x in m])
    return m


def band_qu(band, kind='map', k=None):
    """
    Q/U of `band`: the data (kind 'map'), the HMDM ('hmdm'), the PySM sky plus
    noise simulation k ('skyplusnoise') or noise simulation k alone ('noise').
    """
    exp = CHANNELS[band]['experiment']
    if kind in ('map', 'hmdm'):
        return read_qu(map_file(band, kind), exp)
    noise = read_qu(map_file(band, 'noise', k), exp)
    if kind == 'noise':
        return noise
    return read_qu(map_file(band, 'pysm'), exp) + noise


def spin2_field(mask, qu):
    """Spin-2 NaMaster field of Q/U on `mask`, with E and B purification."""
    return nmt.NmtField(mask, [qu[0], qu[1]], lmax_sht=LMAX, purify_e=True, purify_b=True,
                        beam=None)


def coupling_workspace(mask, b):
    """Mode-coupling matrix of two purified spin-2 fields on `mask`, binning `b`."""
    npix = hp.nside2npix(NSIDE)
    f2 = spin2_field(mask, [np.zeros(npix), np.zeros(npix)])
    w22 = nmt.NmtWorkspace()
    w22.compute_coupling_matrix(f2, f2, b)
    return w22


def coupled_spectra(bands, mask, kind='map', k=None):
    """
    Coupled pseudo-C_l, multipole by multipole, of every pair i <= j of `bands`
    (maps as in band_qu): {(i, j): array (4, lmax + 1)}.  Each map is read and
    its field built once.  They do not depend on the binning.
    """
    fields = [spin2_field(mask, band_qu(band, kind, k)) for band in bands]
    return {(i, j): nmt.compute_coupled_cell(fields[i], fields[j])
            for i in range(len(bands)) for j in range(i, len(bands))}


def decouple_spectra(coupled, w22):
    """EE and BB bandpowers of coupled_spectra with the workspace w22 (its binning)."""
    out = {}
    for pair, cl in coupled.items():
        cl = w22.decouple_cell(cl)
        out[pair] = {'EE': cl[0], 'BB': cl[3]}
    return out


def compute_spectra(bands, mask, w22, kind='map', k=None):
    """
    EE and BB auto- and cross-spectra of `bands` (maps as in band_qu), in
    mK^2_CMB, neither noise-debiased nor beam-deconvolved: {(i, j): {'EE', 'BB'}}
    for i <= j.
    """
    return decouple_spectra(coupled_spectra(bands, mask, kind, k), w22)


def ell_columns(b):
    """First, last and effective multipole of each bandpower of the NmtBin `b`."""
    n = b.get_n_bands()
    return {'ell1': np.array([b.get_ell_min(i) for i in range(n)], float),
            'ell2': np.array([b.get_ell_max(i) for i in range(n)], float),
            'ell_eff': np.asarray(b.get_effective_ells(), float)}


# ===========================================================================
# Spectra files: one table per ordered band pair 'a_b', named after the pair,
# with the columns ell1, ell2, ell_eff and those of the spectra ('EE', 'BB';
# 'EE_MEAN', 'EE_STD', ... for simulations; 'EE_SPECTRUM', 'EE_ERROR', ...
# for corrected spectra).  a_b and b_a hold the same values.
# ===========================================================================
def write_pairs(path, bands, ell, columns, header=None):
    """
    columns(i, j) -> {name: array} for i <= j; ell: dict of ell1, ell2, ell_eff.
    header: optional cards for the primary header.
    """
    hdus = fits.HDUList([fits.PrimaryHDU()])
    for key, value in (header or {}).items():
        hdus[0].header[key] = value
    for i, a in enumerate(bands):
        for j, b in enumerate(bands):
            cols = dict(ell, **columns(min(i, j), max(i, j)))
            hdu = fits.BinTableHDU.from_columns(
                [fits.Column(name=n, format='D', array=v) for n, v in cols.items()])
            hdu.header['BAND_I'], hdu.header['BAND_J'] = a, b
            hdu.name = f'{a}_{b}'
            hdus.append(hdu)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdus.writeto(path, overwrite=True)
    print(f'  -> {path}', flush=True)


def write_spectra(path, spectra, bands, b):
    """Spectra of compute_spectra."""
    write_pairs(path, bands, ell_columns(b), lambda i, j: spectra[i, j])


def read_pairs(path, bands, columns):
    """{'a_b': {column: array}} for every ordered pair of `bands`."""
    with fits.open(path) as hdul:
        return {f'{a}_{b}': {col: np.asarray(hdul[f'{a}_{b}'].data[col], float) for col in columns}
                for a in bands for b in bands}


# ===========================================================================
# Instrument
# ===========================================================================
def beam_table(band, mode):
    """
    Beam transfer function of `band` for l = 0, 1, ... from the official
    files in data/beams/; mode 'TT', 'EE' or 'BB' (only the HFI beams differ).
    """
    ch = CHANNELS[band]
    if ch['experiment'] == 'QUIJOTE':
        with fits.open(beam_file(band)) as hdul:
            return np.array(hdul[1].data[ch['beam_column']][0])
    if ch['experiment'] == 'WMAP':
        return np.loadtxt(beam_file(band)).T[1]
    with fits.open(beam_file(band)) as hdul:
        if band not in HFI_BANDS:                        # Planck LFI
            return np.array(hdul[f'BEAMWF_0{band}X0{band}'].data['BL'])
        return np.array(hdul['WINDOW FUNCTIONS'].data[mode[0]])    # Planck HFI


def beam(band, ell, mode):
    """Beam transfer function of `band` for mode 'EE' or 'BB', interpolated at the multipoles `ell`."""
    bl = beam_table(band, mode)
    out = np.interp(ell, np.arange(len(bl)), bl)
    if not np.all(out > 0):
        raise ValueError(f'non-positive beam values for band {band}')
    return out


def pixel_window(ell):
    """HEALPix pixel window function at N_side 512, interpolated at `ell`."""
    wpix = hp.pixwin(NSIDE)
    return np.interp(ell, np.arange(len(wpix)), wpix)


def cmb_unit_conversion(nu_GHz):
    """K_CMB -> K_RJ factor at the frequency nu_GHz (T_CMB = 2.72548 K)."""
    nu = np.asarray(nu_GHz) * 1e9
    x = h * nu / (k * 2.72548)
    return x**2 * np.exp(x) / (np.exp(x) - 1.0)**2


def planck_uc_hfi():
    """
    K_CMB -> K_RJ factors of the Planck HFI bands 100, 143, 217 and 353 GHz:
    K_CMB -> MJy/sr integrated over the bandpasses including the bandpass
    shifts, times MJy/sr -> K_RJ at nu0 (values computed by RGS, Feb 2025; the
    UC column of data/color_corrections/c_td_10-40_beta_hfi_bps_pr3.fits; see
    https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/UC_CC_Tables).
    """
    kcmb_to_mjysr = np.array([242.09786, 370.53512, 481.93046, 287.22432])
    mjysr_to_krj = 1. / np.array([307.09143, 627.97125, 1446.0629, 3826.6356])
    return kcmb_to_mjysr * mjysr_to_krj


def unit_conversion(band):
    """K_CMB -> K_RJ factor of `band`: bandpass-integrated for HFI, at the reference frequency nu0 otherwise."""
    if band in HFI_BANDS:
        return float(planck_uc_hfi()[HFI_BANDS.index(band)])
    return float(cmb_unit_conversion(freq_ghz(band)))


# ===========================================================================
# Sky model
# ===========================================================================
def cmb_spectrum(ell, mode):
    """C_l (K_CMB^2) of mode 'EE' or 'BB' of the Planck 2018 best fit, interpolated at `ell`."""
    data = np.loadtxt(CMB_FILE)                  # columns L, TT, TE, EE, BB, PP: D_l in uK^2
    ell_file = data[:, 0].astype(int)
    dl = data[:, {'EE': 3, 'BB': 4}[mode]]
    cl = np.zeros_like(dl)
    sel = ell_file >= 2
    cl[sel] = dl[sel] * (2 * np.pi) / (ell_file[sel] * (ell_file[sel] + 1))
    return np.interp(np.asarray(ell), ell_file, cl * 1e-12, left=0, right=0)


def load_color_correction_polynomials():
    """
    Colour-correction polynomials of every band, {'synch': {band: (a0, a1, a2)},
    'dust': {band: (a0, a1, a2)}}, to be evaluated at the spectral index of
    the component in K_RJ, alpha = 2 + beta (inputs/make_cc_polynoms.py).
    """
    with fits.open(os.path.join(CC_DIR, 'cc_polynoms.fits')) as hdul:
        synch, dust = {}, {}
        for row in hdul['CC_POLYNOMS'].data:
            comp, band = str(row['COMP']), str(row['BAND'])
            a0, a1, a2 = float(row['A0']), float(row['A1']), float(row['A2'])
            if comp == 'synch':
                synch[band] = (a0, a1, a2)
            elif comp == 'dust_HFI':
                # fitted in beta_d: P(beta) = a0 + a1 beta + a2 beta^2 becomes
                # Q(alpha) = P(alpha - 2) = (a0 - 2 a1 + 4 a2) + (a1 - 4 a2) alpha + a2 alpha^2
                dust[band] = (a0 - 2.0*a1 + 4.0*a2, a1 - 4.0*a2, a2)
            else:                                # 'dust_nonHFI', in alpha
                dust[band] = (a0, a1, a2)
    return {'synch': synch, 'dust': dust}


def planck(nu_GHz, T):
    """Planck function B_nu(T) [W m^-2 Hz^-1 sr^-1]."""
    nu = np.asarray(nu_GHz, dtype=float) * 1e9
    x = h * nu / (k * T)
    return (2.0 * h * nu**3 / c**2) / np.expm1(x)


def g_RJ(nu_GHz):
    """Rayleigh-Jeans brightness of 1 K."""
    nu = np.asarray(nu_GHz, dtype=float) * 1e9
    return 2.0 * k * nu**2 / c**2


def mbb_scaling_KRJ(nu_GHz, nu0_GHz=353.0, beta=1.59, T_d=19.6):
    """Modified black body in K_RJ, normalized at nu0_GHz."""
    nu_GHz = np.asarray(nu_GHz, dtype=float)
    power = (nu_GHz / nu0_GHz) ** beta
    planck_ratio = planck(nu_GHz, T_d) / planck(nu0_GHz, T_d)
    rj_ratio = g_RJ(nu0_GHz) / g_RJ(nu_GHz)
    return power * planck_ratio * rj_ratio
