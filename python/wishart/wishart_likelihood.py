"""
Wishart likelihood of the band-band bandpower matrices (Section 3).
===================================================================

For every bandpower b the p x p matrix of raw EE (or BB) auto- and
cross-spectra, D_b, is compared with the model

    M_b^ij = W_b^i W_b^j [ C^fg(nu_i, nu_j) / (u_i u_j) + C^CMB ] + N_b^ij

(W: beam x pixel window, u: K_CMB -> K_RJ factor, N: diagonal noise, the spectra
of the half-mission difference maps)
through the Wishart deviance

    -2 ln L = sum_b nu_b [ Tr(D_b M_b^-1) - ln det(D_b M_b^-1) - p ],
    nu_b = f_sky [(l_2 + 1)^2 - l_1^2].

The foreground model has seven parameters, theta = (A_s, alpha_s, beta_s,
A_d, alpha_d, beta_d, rho), amplitudes in uK^2_RJ at l = 80, 23 and 353 GHz.
"""

import copy
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from astropy.io import fits
from scipy.linalg import LinAlgError, cho_factor, cho_solve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import functions as F  # noqa: E402
from data import freq_ghz  # noqa: E402

NU_REF_SYNC = 23.0    # GHz
NU_REF_DUST = 353.0   # GHz
ELL_REF = 80.0        # pivot multipole
T_DUST = 19.6         # K

PARAM_NAMES = ('A_s', 'alpha_s', 'beta_s', 'A_d', 'alpha_d', 'beta_d', 'rho')
MK2_TO_UK2 = 1.0e6    # the spectra files are in mK^2

# Starting point of every fit (A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho)
DEFAULT_START = (5.0e-3, -3.0, -3.0, 2.0e-3, -2.5, 1.5, 0.0)
# Bin-to-bin fit: each bandpower alone, slopes fixed, Gaussian prior on beta_s
BINBIN_FIXED = {'alpha_s': 0.0, 'alpha_d': 0.0}
BINBIN_PRIORS = {'beta_s': (-3.1, 0.30)}


# ===========================================================================
# Foreground SEDs, in K_RJ, divided by the colour correction of each band
# evaluated at alpha = 2 + beta
# ===========================================================================
def _colour_correction(poly, alpha):
    return poly[0] + poly[1] * alpha + poly[2] * alpha * alpha


def sed_synchrotron(freqs, beta_s, bands, cc_dict):
    s = (np.asarray(freqs, float) / NU_REF_SYNC) ** beta_s
    alpha_cc = 2.0 + float(beta_s)
    cc = np.array([_colour_correction(cc_dict['synch'][str(b)], alpha_cc) for b in bands])
    return s / cc


def sed_dust(freqs, beta_d, T_d, bands, cc_dict):
    d = F.mbb_scaling_KRJ(np.asarray(freqs, float), nu0_GHz=NU_REF_DUST, beta=beta_d, T_d=T_d)
    alpha_cc = 2.0 + float(beta_d)
    cc = np.array([_colour_correction(cc_dict['dust'][str(b)], alpha_cc) for b in bands])
    return np.asarray(d / cc, float)


# ===========================================================================
# Data
# ===========================================================================
@dataclass
class WishartData:
    """
    Everything the likelihood needs for one mask, mode and set of bands.

    bands (p), freqs (p, GHz), ell_eff/ell1/ell2 (n_bins), nu_dof (n_bins),
    D_hat (n_bins, p, p) raw bandpowers in uK^2_CMB, N (n_bins, p, p)
    noise (diagonal), W (n_bins, p) beam x pixel window, unit (p) K_CMB ->
    K_RJ factors, C_cmb (n_bins) CMB in uK^2_CMB, cc_dict colour corrections.
    """
    bands: List[str]
    freqs: np.ndarray
    ell_eff: np.ndarray
    ell1: np.ndarray
    ell2: np.ndarray
    nu_dof: np.ndarray
    D_hat: np.ndarray
    N: np.ndarray
    W: np.ndarray
    unit: np.ndarray
    C_cmb: np.ndarray
    mode: str
    f_sky: float
    cc_dict: dict
    _logdet_D: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_bins(self):
        return len(self.ell_eff)

    @property
    def n_bands(self):
        return len(self.bands)

    @property
    def logdet_D(self):
        """log det of each data matrix (constant, cached)."""
        if self._logdet_D is None:
            out = np.empty(self.n_bins)
            for b in range(self.n_bins):
                sign, val = np.linalg.slogdet(self.D_hat[b])
                out[b] = val if sign > 0 else -np.inf
            self._logdet_D = out
        return self._logdet_D

    def check_positive_definite(self):
        """The Wishart is defined only for positive-definite data matrices."""
        bad = [b for b in range(self.n_bins) if not np.linalg.eigvalsh(self.D_hat[b])[0] > 0]
        if bad:
            raise ValueError(f'the {self.mode} data matrix is not positive definite at '
                             f'l_eff = {self.ell_eff[bad]}')

    def select_bins(self, idx):
        """A copy holding only the bandpowers idx (bin-to-bin fit)."""
        k = np.atleast_1d(np.arange(self.n_bins)[idx])
        new = copy.copy(self)
        for name in ('ell_eff', 'ell1', 'ell2', 'nu_dof', 'D_hat', 'N', 'W', 'C_cmb'):
            setattr(new, name, np.asarray(getattr(self, name))[k])
        new._logdet_D = None
        return new


def transfer_functions(bands, mode, ell):
    """W_l^i = B_l^i w_l, beam times pixel window, (len(ell), p)."""
    B = np.zeros((len(ell), len(bands)))
    for i, b in enumerate(bands):
        B[:, i] = np.asarray(F.beam(b, ell, mode), float)
    return B * F.pixel_window(ell)[:, None]


def load_wishart_data(path_raw, path_noise, bands, mode, f_sky):
    """
    WishartData of one mask from its data spectra and noise files (mK^2_CMB),
    for the bands `bands` (in that order) and mode 'EE' or 'BB'.  The noise file
    holds the HMDM spectra (column <mode>) or, in the simulation tests, the mean of
    the noise simulations (config.sims_path(key, 'noise', binning), column <mode>_MEAN).
    """
    bands = [str(b) for b in bands]
    p = len(bands)
    with fits.open(path_raw) as hdul:
        ref = hdul[f'{bands[0]}_{bands[0]}'].data
        ell_eff = np.asarray(ref['ell_eff'], float)
        ell1 = np.asarray(ref['ell1'], float)
        ell2 = np.asarray(ref['ell2'], float)
        D = np.zeros((len(ell_eff), p, p))
        for i, a in enumerate(bands):
            for j, b in enumerate(bands):
                D[:, i, j] = np.asarray(hdul[f'{a}_{b}'].data[mode], float)
    # a_b and b_a hold the same values; symmetrize against rounding
    D_hat = 0.5 * (D + np.transpose(D, (0, 2, 1))) * MK2_TO_UK2

    # the noise of the selected bands is uncorrelated: N is the diagonal of noise auto-spectra
    N = np.zeros((len(ell_eff), p, p))
    with fits.open(path_noise) as hdul:
        column = f'{mode}_MEAN' if f'{mode}_MEAN' in hdul[f'{bands[0]}_{bands[0]}'].columns.names else mode
        for i, a in enumerate(bands):
            N[:, i, i] = np.asarray(hdul[f'{a}_{a}'].data[column], float) * MK2_TO_UK2
    if not np.all(np.einsum('bii->bi', N) > 0):
        raise ValueError(f'non-positive noise auto-spectrum in {path_noise}')

    return WishartData(
        bands=bands, freqs=np.array([freq_ghz(b) for b in bands]),
        ell_eff=ell_eff, ell1=ell1, ell2=ell2,
        nu_dof=f_sky * ((ell2 + 1.0) ** 2 - ell1 ** 2),
        D_hat=D_hat, N=N, W=transfer_functions(bands, mode, ell_eff),
        unit=np.array([F.unit_conversion(b) for b in bands]),
        C_cmb=np.asarray(F.cmb_spectrum(ell_eff, mode), float) * 1e12,     # K^2 -> uK^2
        mode=mode, f_sky=f_sky, cc_dict=F.load_color_correction_polynomials())


# ===========================================================================
# Model and likelihood
# ===========================================================================
def build_model_matrix(theta, wd):
    """Model matrix M_b (n_bins, p, p), in the units and domain of wd.D_hat."""
    A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho = [float(t) for t in theta[:7]]
    ell = wd.ell_eff / ELL_REF

    s = sed_synchrotron(wd.freqs, beta_s, wd.bands, wd.cc_dict)
    d = sed_dust(wd.freqs, beta_d, T_DUST, wd.bands, wd.cc_dict)
    F_s = A_s * ell ** alpha_s
    F_d = A_d * ell ** alpha_d
    F_x = rho * np.sqrt(max(A_s * A_d, 0.0)) * ell ** (0.5 * (alpha_s + alpha_d))

    # foreground matrix in K_RJ
    ss = np.outer(s, s)
    dd = np.outer(d, d)
    sd = np.outer(s, d)
    C_fg = (F_s[:, None, None] * ss
            + F_d[:, None, None] * dd
            + F_x[:, None, None] * (sd + sd.T))

    # beams, pixel window and K_RJ -> K_CMB, then the CMB and the noise
    t = wd.W / wd.unit[None, :]
    M = C_fg * (t[:, :, None] * t[:, None, :])
    M = M + wd.C_cmb[:, None, None] * (wd.W[:, :, None] * wd.W[:, None, :])
    return M + wd.N


def wishart_deviance(D_hat, M, nu, logdet_D):
    """sum_b nu_b [Tr(D_b M_b^-1) - ln det(D_b M_b^-1) - p]; +inf if an M_b is not positive definite."""
    nb, p = M.shape[0], M.shape[1]
    total = 0.0
    for b in range(nb):
        try:
            c, low = cho_factor(M[b], lower=True, check_finite=False)
        except (LinAlgError, np.linalg.LinAlgError, ValueError):
            return np.inf
        diag = np.diag(c)
        if not np.all(diag > 0) or not np.all(np.isfinite(diag)):
            return np.inf
        logdet_M = 2.0 * np.sum(np.log(diag))
        tr = np.trace(cho_solve((c, low), D_hat[b], check_finite=False))
        total += nu[b] * (tr + logdet_M - logdet_D[b] - p)
    return float(total) if np.isfinite(total) else np.inf


def log_likelihood(theta, wd):
    dev = wishart_deviance(wd.D_hat, build_model_matrix(theta, wd), wd.nu_dof, wd.logdet_D)
    if not np.isfinite(dev):
        return -np.inf
    return -0.5 * dev


def log_prior(theta, gaussian_priors=None):
    """
    Uniform priors A > 0, -6 <= alpha <= 0, -6 <= beta_s <= 0, 0 <= beta_d <= 6,
    -1 <= rho <= 1, plus optional Gaussian priors {name: (mean, sigma)}.
    """
    A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho = [float(t) for t in theta[:7]]
    if not (A_s > 0):
        return -np.inf
    if not (-6.0 <= alpha_s <= 0.0):
        return -np.inf
    if not (-6.0 <= beta_s <= 0.0):
        return -np.inf
    if not (A_d > 0):
        return -np.inf
    if not (-6.0 <= alpha_d <= 0.0):
        return -np.inf
    if not (0.0 <= beta_d <= 6.0):
        return -np.inf
    if not (-1.0 <= rho <= 1.0):
        return -np.inf

    lp = 0.0
    if gaussian_priors:
        values = dict(zip(PARAM_NAMES, (A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho)))
        for name, (mu, sig) in gaussian_priors.items():
            lp += -0.5 * ((values[name] - mu) / sig) ** 2
    return lp


def log_posterior(theta, wd, gaussian_priors=None):
    lp = log_prior(theta, gaussian_priors)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, wd)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def expand_theta(theta_free, fixed=None):
    """The full 7-vector, with the fixed parameters inserted."""
    fixed = fixed or {}
    if not fixed:
        return np.asarray(theta_free, float)
    full = np.empty(len(PARAM_NAMES))
    i_free = 0
    for i, name in enumerate(PARAM_NAMES):
        if name in fixed:
            full[i] = float(fixed[name])
        else:
            full[i] = theta_free[i_free]
            i_free += 1
    return full


def _log_posterior_free(theta_free, wd, fixed, gaussian_priors):
    return log_posterior(expand_theta(theta_free, fixed), wd, gaussian_priors)


# emcee runs the posterior in worker processes, which receive the data once
_WORKER_STATE = {}


def _init_worker(wd, gaussian_priors, fixed):
    _WORKER_STATE.update(wd=wd, gaussian_priors=gaussian_priors, fixed=fixed)


def _log_posterior_worker(theta_free):
    return _log_posterior_free(theta_free, _WORKER_STATE['wd'], _WORKER_STATE['fixed'],
                               _WORKER_STATE['gaussian_priors'])


def n_data_points(wd):
    """Independent entries of the data matrices: n_bins p(p+1)/2."""
    p = wd.n_bands
    return wd.n_bins * p * (p + 1) // 2


def reduced_chi2(theta, wd, fixed=None):
    """Deviance at theta over n_data - n_par."""
    dev = wishart_deviance(wd.D_hat, build_model_matrix(theta, wd), wd.nu_dof, wd.logdet_D)
    return dev / (n_data_points(wd) - (len(PARAM_NAMES) - len(fixed or {})))


# ===========================================================================
# Maximum of the posterior and MCMC
# ===========================================================================
def _clip_to_prior(theta):
    """theta moved just inside the prior box."""
    t = list(map(float, theta))
    eps = 1e-6
    t[0] = max(t[0], 1e-12)
    t[1] = min(max(t[1], -6 + eps), -eps)
    t[2] = min(max(t[2], -6 + eps), -eps)
    t[3] = max(t[3], 1e-12)
    t[4] = min(max(t[4], -6 + eps), -eps)
    t[5] = min(max(t[5], eps), 6 - eps)
    t[6] = min(max(t[6], -1 + eps), 1 - eps)
    return np.array(t)


def maximize_posterior(wd, theta_start=DEFAULT_START, gaussian_priors=None, n_restarts=3,
                       fixed=None, seed=None):
    """
    Maximum of the posterior with Nelder-Mead, amplitudes in log space; each
    restart starts from the previous optimum with a random jitter.  Returns
    the full 7-vector and the deviance there.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    fixed = fixed or {}
    free = [i for i, n in enumerate(PARAM_NAMES) if n not in fixed]
    log_idx = [j for j, i in enumerate(free) if i in (0, 3)]          # the amplitudes
    step = np.array([0.3, 0.15, 0.10, 0.3, 0.15, 0.05, 0.05])[free]

    def unpack(x):
        xf = np.array(x, float)
        for j in log_idx:
            xf[j] = np.exp(xf[j])
        return expand_theta(xf, fixed)

    def neg(x):
        val = log_posterior(unpack(x), wd, gaussian_priors)
        return 1e30 if not np.isfinite(val) else -val

    best_x, best_f = None, np.inf
    t0 = _clip_to_prior(theta_start)
    for r in range(max(1, n_restarts)):
        x0 = np.array(t0, float)[free]
        for j in log_idx:
            x0[j] = np.log(max(x0[j], 1e-300))
        if r > 0:
            x0 = x0 + rng.normal(0, step)
        res = minimize(neg, x0, method='Nelder-Mead',
                       options=dict(maxiter=20000, maxfev=20000, xatol=1e-8, fatol=1e-8))
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x
        t0 = unpack(res.x)

    theta_best = unpack(best_x)
    return theta_best, wishart_deviance(wd.D_hat, build_model_matrix(theta_best, wd),
                                        wd.nu_dof, wd.logdet_D)


def run_wishart_mcmc(wd, nwalkers, nsteps, discard_fraction=0.5, theta_start=DEFAULT_START,
                     gaussian_priors=None, fixed=None, n_processes=1, seed=42):
    """
    Sample the posterior with emcee, the walkers starting in a small ball
    around its maximum.  Parameters in `fixed` are held at those values.

    Returns the samples after discarding the first discard_fraction of every
    chain, always with 7 columns (fixed ones constant), and the
    maximum-posterior sample with chi2_red there.
    """
    import emcee

    rng = np.random.default_rng(seed)
    fixed = fixed or {}
    free = [i for i, n in enumerate(PARAM_NAMES) if n not in fixed]
    ndim = len(free)

    theta_map, _ = maximize_posterior(wd, theta_start=theta_start, gaussian_priors=gaussian_priors,
                                      fixed=fixed, seed=seed)

    # amplitudes get a relative scatter, indices an absolute one
    scale = np.array([0.05 * max(theta_map[0], 1e-12), 0.02, 0.02,
                      0.05 * max(theta_map[3], 1e-12), 0.02, 0.01, 0.02])[free]
    pos = np.empty((nwalkers, ndim))
    theta_map_free = np.asarray(theta_map)[free]
    for w in range(nwalkers):
        for _ in range(200):
            cand = theta_map_free + rng.normal(0, scale)
            if np.isfinite(log_prior(expand_theta(cand, fixed), gaussian_priors)):
                pos[w] = cand
                break
        else:
            pos[w] = np.asarray(_clip_to_prior(theta_map))[free]

    def run(sampler):
        sampler.random_state = np.random.RandomState(seed).get_state()
        sampler.run_mcmc(pos, nsteps, progress=False)
        return sampler

    if n_processes > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(processes=n_processes, initializer=_init_worker,
                                         initargs=(wd, gaussian_priors, fixed)) as pool:
            sampler = run(emcee.EnsembleSampler(nwalkers, ndim, _log_posterior_worker, pool=pool))
    else:
        sampler = run(emcee.EnsembleSampler(nwalkers, ndim, _log_posterior_free,
                                            args=(wd, fixed, gaussian_priors)))

    discard = int(discard_fraction * nsteps)
    samples = np.array([expand_theta(t, fixed)
                        for t in sampler.get_chain(discard=discard, flat=True)])
    theta_best = samples[int(np.argmax(sampler.get_log_prob(discard=discard, flat=True)))]
    return samples, dict(theta_map=theta_best, chi2_reduced=reduced_chi2(theta_best, wd, fixed),
                         n_data=n_data_points(wd))


PERCENTILES = [15.865, 50.0, 84.135]      # lower edge, median and upper edge of the 68 per cent interval


def summarize_samples(samples):
    """{parameter: (median, upper error, lower error)}, 68 per cent interval."""
    return summarize_percentiles(np.percentile(samples, PERCENTILES, axis=0))


def summarize_percentiles(percentiles):
    """summarize_samples from the PERCENTILES of the samples, (3, n_par)."""
    out = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, med, hi = percentiles[:, i]
        out[name] = (med, hi - med, med - lo)
    return out


def wishart_sigma(wd, M):
    """Standard deviation of each bandpower implied by the Wishart, sqrt((M_ij^2 + M_ii M_jj)/nu)."""
    diag = np.einsum('bii->bi', M)
    var = (M ** 2 + diag[:, :, None] * diag[:, None, :]) / wd.nu_dof[:, None, None]
    return np.sqrt(np.maximum(var, 0.0))
