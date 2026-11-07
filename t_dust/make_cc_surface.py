#%%
"""
Dust colour corrections of the HFI bands as a function of beta_d and T_d.
=========================================================================

    python make_cc_surface.py

The paper's HFI dust colour corrections (python/inputs/make_cc_polynoms.py)
are a quadratic in beta_d fitted at T_d = 19.6 K to RGS's bandpass-shifted
table data/color_corrections/c_td_10-40_beta_hfi_bps_pr3.fits (variant [1]:
10 <= T_d <= 40 K in steps of 0.1 K, 1.0 <= beta_d <= 2.4 in steps of 0.01).
With T_d free, each band gets a surface fitted to the whole table,

    cc(beta_d, T_d) = sum_{i<=2, j<=4} c_ij (beta_d - 1.5)^i (20 K / T_d)^j,

quadratic in beta_d like the paper's polynomial, and so extrapolated like it
beyond the table, and quartic in 1/T_d.  It reproduces the table to 2.2e-5
(the paper's polynomial: 1.9e-5 at 19.6 K) and the paper's polynomial at
T_d = 19.6 K to 8e-8 for 0 <= beta_d <= 6, so that at 19.6 K the model of
t_dust/ is the paper's.  A cubic in beta_d would fit the table to 6e-7 but
would extrapolate differently from the paper beyond beta_d = 2.4.

The table has no T_d dependence outside 10-40 K: the prior of T_d is that range.

Writes t_dust/data/cc_dust_hfi_surface.fits, table CC_SURFACE with BAND,
COEF (c_ij, 3 x 5, row i the power of beta_d - 1.5) and MAXRES (largest
residual over the table).
"""

import os

import tdconfig as C

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

import functions as F  # noqa: E402
from config import CC_DIR  # noqa: E402

HFI_BANDS = ['100', '143', '217', '353']
BETA0, T_SCALE = 1.5, 20.0          # the surface is in beta_d - BETA0 and T_SCALE / T_d
DEG_BETA, DEG_INVT = 2, 4


def design(beta, T_d):
    """Columns (beta - BETA0)^i (T_SCALE/T_d)^j, i <= DEG_BETA, j <= DEG_INVT (i major)."""
    u, v = np.ravel(beta) - BETA0, T_SCALE / np.ravel(T_d)
    return np.stack([u**i * v**j for i in range(DEG_BETA + 1) for j in range(DEG_INVT + 1)], axis=1)


with fits.open(os.path.join(CC_DIR, 'c_td_10-40_beta_hfi_bps_pr3.fits')) as hdul:
    row = hdul[1].data[0]
    table_bands = list(row['BANDS'])
    td = np.asarray(row['TD'], float)
    beta = np.asarray(row['BETA'], float)
    ca = np.asarray(row['CA'], float)[1]          # [1]: the Planck HFI corrections; [band, beta, T_d]

TT, BB = np.meshgrid(td, beta)                    # (beta, T_d), as ca[band]
A = design(BB, TT)
paper = F.load_color_correction_polynomials()['dust']
beta_wide = np.linspace(0.0, 6.0, 601)            # the prior range of beta_d
i196 = int(np.argmin(np.abs(td - 19.6)))

coefs, maxres = [], []
print(f'grid: T_d {td[0]:.1f}-{td[-1]:.1f} K ({td.size}), beta_d {beta[0]:.2f}-{beta[-1]:.2f} ({beta.size})')
for b in HFI_BANDS:
    y = ca[table_bands.index(b)].ravel()
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = np.abs(A @ c - y)
    coefs.append(c.reshape(DEG_BETA + 1, DEG_INVT + 1))
    maxres.append(res.max())

    # checks: the paper's polynomial against the table at 19.6 K, and the
    # surface against the paper's polynomial over the whole prior of beta_d
    a = paper[b]
    alpha = 2.0 + beta
    paper_res = np.abs(a[0] + a[1] * alpha + a[2] * alpha**2 - ca[table_bands.index(b)][:, i196]).max()
    alpha = 2.0 + beta_wide
    old = a[0] + a[1] * alpha + a[2] * alpha**2
    new = design(beta_wide, np.full_like(beta_wide, 19.6)) @ c
    print(f'{b:>4} GHz: surface - table max {res.max():.1e} (T_d 13-35 K: '
          f'{res[((TT >= 13) & (TT <= 35)).ravel()].max():.1e}); paper polynomial - table at '
          f'19.6 K {paper_res:.1e}; surface - paper polynomial at 19.6 K, beta_d 0-6: '
          f'{np.abs(new - old).max():.1e}; cc over the table {y.min():.4f}-{y.max():.4f}')

os.makedirs(C.DATA_DIR, exist_ok=True)
cols = [fits.Column(name='BAND', format='8A', array=np.array(HFI_BANDS)),
        fits.Column(name='COEF', format=f'{(DEG_BETA + 1) * (DEG_INVT + 1)}D',
                    dim=f'({DEG_INVT + 1},{DEG_BETA + 1})', array=np.array(coefs)),
        fits.Column(name='MAXRES', format='D', array=np.array(maxres))]
hdu = fits.BinTableHDU.from_columns(cols, name='CC_SURFACE')
hdu.header['BETA0'] = (BETA0, 'surface in (beta_d - BETA0)^i')
hdu.header['TSCALE'] = (T_SCALE, 'and (TSCALE / T_d)^j, T_d in K')
hdu.header['TDMIN'], hdu.header['TDMAX'] = float(td[0]), float(td[-1])
hdu.header['SOURCE'] = 'c_td_10-40_beta_hfi_bps_pr3.fits, variant [1]'
fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(C.CC_SURFACE, overwrite=True)
print(f'  -> {C.CC_SURFACE}')
