#%%
"""
Data power spectra of every mask.
=================================

    python compute_spectra.py [--binning all] [--masks all] [--hmdm | --hmdm-only]

EE and BB auto- and cross-spectra of the fourteen bands, with NaMaster's
pseudo-C_l estimator and E and B purification, binned as in config.BINNINGS
(all of them by default: the maps are read and their pseudo-C_l computed once,
then decoupled for each binning).  They are neither noise-debiased nor
beam-deconvolved.  Both likelihoods read them; dl30 and eqmode are the
alternative binnings of Sect. 3.2.  Writes

    results/spectra/power_spectra_<mask>_<binning>.fits

With --hmdm, also the spectra of the half-mission difference maps, the noise
N_l of the likelihoods (config.NOISE), and with --hmdm-only those alone:

    results/spectra/power_spectra_<mask>_hmdm_<binning>.fits

NaMaster uses OMP_NUM_THREADS threads.
"""

import argparse

import healpy as hp

import functions as F
from config import BANDS, BINNINGS, binning, mask_names, mask_path, spectra_path

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--binning', default='all', choices=['all'] + list(BINNINGS))
ap.add_argument('--masks', default='all', help='comma-separated mask keys (default: all)')
ap.add_argument('--hmdm', action='store_true', help='also the spectra of the HMDM maps')
ap.add_argument('--hmdm-only', action='store_true', help='only the spectra of the HMDM maps')
args = ap.parse_args()
kinds = (True,) if args.hmdm_only else ((False, True) if args.hmdm else (False,))

names = list(BINNINGS) if args.binning == 'all' else [args.binning]
bins = {name: binning(name) for name in names}
for key in mask_names(args.masks):
    print(f'[{key} {" ".join(names)}]', flush=True)
    mask = hp.read_map(mask_path(key))
    workspaces = {name: F.coupling_workspace(mask, b) for name, b in bins.items()}
    for hmdm in kinds:
        coupled = F.coupled_spectra(BANDS, mask, kind='hmdm' if hmdm else 'map')
        for name, w22 in workspaces.items():
            F.write_spectra(spectra_path(key, name, hmdm), F.decouple_spectra(coupled, w22), BANDS, bins[name])
