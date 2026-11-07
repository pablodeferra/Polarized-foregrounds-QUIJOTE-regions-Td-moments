"""
The fourteen frequency channels: reference frequency, beam and input files.
===========================================================================

`freq` is the reference frequency nu0 of the colour corrections (fastcc
option 3; inputs/make_cc_polynoms.py), at which the foreground SEDs are
evaluated and the K_CMB -> K_RJ factor is computed (for the HFI bands the
factor is bandpass-integrated, see functions.unit_conversion); `fwhm` is only
quoted in Table 1.

Map files are relative to config.MAPS_DIR, beams to config.BEAM_DIR:

  map        full-mission I, Q, U map (QUIJOTE and WMAP in mK, Planck in K)
  half_1/2   the two halves the HMDM is built from (inputs/make_hmdm.py)
  hmdm       half-mission difference map, the noise estimate
  pysm       PySM sky (inputs/make_pysm_maps.py), for the simulations
  noise      noise simulation k (format with k = 1, 2, ...), idem

The Planck entries are PR4 (NPIPE maps, with the bandpass-leakage correction,
and npipe6v20 noise simulations, numbered from 00200).  PR4 has no PSB-only
353 GHz map: its full-channel map is used.
"""

import os

import astropy.units as u

from config import BEAM_DIR, MAPS_DIR


def _quijote(band, freq, fwhm, horn, beam_column):
    name = f'quijote_mfi_skymap_{band}ghz_512_dr1'
    return dict(experiment='QUIJOTE', freq=freq * u.GHz, fwhm=fwhm * u.arcmin,
                map=f'QUIJOTE/{name}.fits',
                half_1=f'QUIJOTE/hmdm/{name}_half1.fits',
                half_2=f'QUIJOTE/hmdm/{name}_half2.fits',
                hmdm=f'QUIJOTE/hmdm/{name}_hmdm.fits',
                pysm=f'PYSM/map_QUIJOTE_{band}_nside512_beamconv.fits',
                noise=f'QUIJOTE/noise_simulations/{band}_quijote/quijote_{band}GHz_{horn}_{{:04d}}_sm1deg.fits',
                beam='rimo_quijote_mfi_beamtf_dr1.fits', beam_column=beam_column)


def _wmap(band, letter, freq, fwhm):
    return dict(experiment='WMAP', freq=freq * u.GHz, fwhm=fwhm * u.arcmin,
                map=f'WMAP/wmap_band_iqumap_r9_9yr_{letter}_v5.fits',
                half_1=f'WMAP/hmdm/wmap_iqumap_r9_yr1to4_{letter}_v5.fits',
                half_2=f'WMAP/hmdm/wmap_iqumap_r9_yr5to9_{letter}_v5.fits',
                hmdm=f'WMAP/hmdm/wmap_iqumap_r9_hmdm_{letter}_v5.fits',
                pysm=f'PYSM/map_WMAP_{band}_nside512_beamconv.fits',
                noise=f'WMAP/noise_simulations/{band}/white_noise_{band}ghz_{{:04d}}.fits',
                beam=f'wmap_ampl_bl_{letter}_9yr_v5p1.txt', letter=letter)


def _lfi(band, freq, fwhm):
    name = f'LFI_SkyMap_{int(band):03d}-BPassCorrected_1024_R4.00'
    return dict(experiment='Planck', freq=freq * u.GHz, fwhm=fwhm * u.arcmin,
                map=f'Planck/{name}_full.fits',
                half_1=f'Planck/hmdm/{name}_full-ringhalf-1.fits',
                half_2=f'Planck/hmdm/{name}_full-ringhalf-2.fits',
                hmdm=f'Planck/hmdm/{name}_hmdm.fits',
                pysm=f'PYSM/map_Planck_{band}_nside512_beamconv.fits',
                noise=f'Planck/noise_simulations/{band}_pr4/npipe6v20_noise_{int(band):03d}_mc_{{:05d}}.fits',
                noise_offset=199, beam='LFI_RIMO_R3.31.fits')


def _hfi(band, freq, fwhm):
    name = f'HFI_SkyMap_{band}-BPassCorrected_2048_R4.00'
    return dict(experiment='Planck', freq=freq * u.GHz, fwhm=fwhm * u.arcmin,
                map=f'Planck/{name}_full.fits',
                half_1=f'Planck/hmdm/{name}_full-ringhalf-1.fits',
                half_2=f'Planck/hmdm/{name}_full-ringhalf-2.fits',
                hmdm=f'Planck/hmdm/{name}_hmdm.fits',
                pysm=f'PYSM/map_Planck_{band}_nside512_beamconv.fits',
                noise=f'Planck/noise_simulations/{band}_pr4/npipe6v20_noise_{band}_mc_{{:05d}}.fits',
                noise_offset=199, beam=f'Bl_TEB_R3.01_fullsky_{band}x{band}.fits')


CHANNELS = {
    '11': _quijote('11', 11.1, 55.38, 'horn3', 'Bl_311'),
    '17': _quijote('17', 16.8, 38.95, 'hornx', 'Bl_417'),
    '23': _wmap('23', 'K', 22.8, 52.8),
    '33': _wmap('33', 'Ka', 33.0, 39.6),
    '41': _wmap('41', 'Q', 40.6, 30.6),
    '61': _wmap('61', 'V', 60.8, 21.0),
    '94': _wmap('94', 'W', 93.5, 13.2),
    '30': _lfi('30', 28.4, 32.29),
    '44': _lfi('44', 44.1, 26.99),
    '70': _lfi('70', 70.4, 13.22),
    '100': _hfi('100', 100.0, 9.680),
    '143': _hfi('143', 143.0, 7.300),
    '217': _hfi('217', 217.0, 5.020),
    '353': _hfi('353', 353.0, 4.940),
}


def map_file(band, key, k=None):
    """Full path of a map of `band` (key 'map', 'hmdm', 'pysm', 'noise', ...); k: noise realization 1, 2, ..."""
    name = CHANNELS[band][key]
    if k is not None:
        name = name.format(k + CHANNELS[band].get('noise_offset', 0))
    return os.path.join(MAPS_DIR, name)


def beam_file(band):
    return os.path.join(BEAM_DIR, CHANNELS[band]['beam'])


def freq_ghz(band):
    """Reference frequency nu0 in GHz."""
    return float(CHANNELS[band]['freq'].to('GHz').value)
