"""
Wishart likelihood with the dust temperature free.
==================================================

The model of the paper (python/wishart/wishart_likelihood.py) with the dust
temperature as an eighth parameter,

    theta = (A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, T_d, rho),

and a uniform prior 10 <= T_d <= 40 K, the range of the HFI colour-correction
table.  The data, noise term, beams, pixel window, K_CMB -> K_RJ factors, CMB,
the other priors and the deviance are the paper's.  At T_d = 19.6 K the model
matrix is the paper's to 1e-7 (relative): the paper's model is this one with
T_d fixed, so the two are nested.

Colour corrections (only those of the dust depend on T_d):

  HFI dust                the surface in (beta_d, 1/T_d) of make_cc_surface.py,
                          fitted to RGS's bandpass-shifted table
  QUIJOTE/WMAP/LFI dust   as in the paper, the power-law correction at
                          alpha = 2 + beta_d
  synchrotron             as in the paper

The K_CMB -> K_RJ factors depend only on the bandpasses, not on the SED of the
source, and do not change.

The machinery below (Model: priors, maxima of the likelihood and of the
posterior, emcee) works for any list of parameters and any foreground matrix,
so that moments/ adds its moment terms on top of this model.  The deviance is
the paper's (wishart_likelihood.wishart_deviance) computed for all the
bandpowers at once, about twice as fast.
"""

import os

import numpy as np
from astropy.io import fits
from scipy.constants import h, k

import tdconfig as C
import wishart_likelihood as W  # noqa: E402
from config import MASK, noise_path, spectra_path  # noqa: E402

NU_REF_SYNC, NU_REF_DUST, ELL_REF = W.NU_REF_SYNC, W.NU_REF_DUST, W.ELL_REF
PARAM_NAMES = ('A_s', 'alpha_s', 'beta_s', 'A_d', 'alpha_d', 'beta_d', 'T_d', 'rho')

# name: (lower bound, upper bound, start, walkers' ball, optimizer restart step).
# Parameters with an infinite upper bound are amplitudes: they must be > 0, get
# a relative ball and are optimized in log.  Everything but T_d is the paper's
# (wishart_likelihood: log_prior, DEFAULT_START, run_wishart_mcmc, maximize_posterior).
SPEC = {
    'A_s': (0.0, np.inf, 5.0e-3, 0.05, 0.3),
    'alpha_s': (-6.0, 0.0, -3.0, 0.02, 0.15),
    'beta_s': (-6.0, 0.0, -3.0, 0.02, 0.10),
    'A_d': (0.0, np.inf, 2.0e-3, 0.05, 0.3),
    'alpha_d': (-6.0, 0.0, -2.5, 0.02, 0.15),
    'beta_d': (0.0, 6.0, 1.5, 0.01, 0.05),
    'T_d': (10.0, 40.0, W.T_DUST, 0.5, 2.0),
    'rho': (-1.0, 1.0, 0.0, 0.02, 0.05),
}

T_D_FIXED = {'T_d': W.T_DUST}                 # the paper's model
BINBIN_FIXED = dict(W.BINBIN_FIXED)           # alpha_s = alpha_d = 0
BINBIN_PRIORS = dict(W.BINBIN_PRIORS)         # beta_s ~ N(-3.1, 0.3^2)
PERCENTILES = [2.275, 15.865, 50.0, 84.135, 97.725]     # 95 and 68 per cent intervals, median


# ===========================================================================
# Colour corrections
# ===========================================================================
def load_cc_surface():
    """HFI dust surfaces of make_cc_surface.py: {band: c_ij (3, 5)}, BETA0 and TSCALE."""
    with fits.open(C.CC_SURFACE) as hdul:
        t = hdul['CC_SURFACE']
        surface = {str(b).strip(): np.asarray(cf, float) for b, cf in zip(t.data['BAND'], t.data['COEF'])}
        return surface, float(t.header['BETA0']), float(t.header['TSCALE'])


class ColourCorrections:
    """Colour corrections of the bands of one fit, evaluated for all of them at once."""

    def __init__(self, bands, cc_dict):
        surface, self.beta0, self.tscale = load_cc_surface()
        self._synch = np.array([cc_dict['synch'][b] for b in bands])        # (p, 3), in alpha
        self._dust = np.array([cc_dict['dust'][b] for b in bands])          # (p, 3), in alpha
        self._hfi = np.array([b in surface for b in bands])
        self._surface = np.array([surface[b] for b in bands if b in surface])   # (n_hfi, 3, 5)

    def synchrotron(self, beta_s):
        a = 2.0 + beta_s
        return self._synch[:, 0] + self._synch[:, 1] * a + self._synch[:, 2] * a * a

    def dust(self, beta_d, T_d):
        a = 2.0 + beta_d
        cc = self._dust[:, 0] + self._dust[:, 1] * a + self._dust[:, 2] * a * a
        if self._surface.size:
            u = (beta_d - self.beta0) ** np.arange(self._surface.shape[1])
            v = (self.tscale / T_d) ** np.arange(self._surface.shape[2])
            cc[self._hfi] = (self._surface @ v) @ u
        return cc

    def synchrotron_dlog(self, beta_s):
        """d ln cc / d beta_s (moment expansion)."""
        a = 2.0 + beta_s
        return (self._synch[:, 1] + 2.0 * self._synch[:, 2] * a) / self.synchrotron(beta_s)

    def dust_dlog(self, beta_d, T_d, cc=None):
        """d ln cc / d beta_d and d ln cc / d T_d (moment expansion); cc: self.dust(beta_d, T_d) if known."""
        a = 2.0 + beta_d
        d_beta = self._dust[:, 1] + 2.0 * self._dust[:, 2] * a
        d_T = np.zeros_like(d_beta)
        if self._surface.size:
            nb, nt = self._surface.shape[1:]
            u, v = beta_d - self.beta0, self.tscale / T_d
            U, V = u ** np.arange(nb), v ** np.arange(nt)
            dU = np.concatenate([[0.0], np.arange(1, nb) * u ** np.arange(nb - 1)])
            # d v^j / d T_d = j v^(j-1) dv/dT_d, dv/dT_d = -v / T_d
            dV = np.concatenate([[0.0], np.arange(1, nt) * v ** np.arange(nt - 1)]) * (-v / T_d)
            d_beta[self._hfi] = (self._surface @ V) @ dU
            d_T[self._hfi] = (self._surface @ dV) @ U
        if cc is None:
            cc = self.dust(beta_d, T_d)
        return d_beta / cc, d_T / cc


# ===========================================================================
# Data and model matrix
# ===========================================================================
def load_wishart(key, mode, bands, binning='dl20'):
    """The paper's WishartData of a mask (spectra, noise term N_b of config.NOISE, f_sky)."""
    wd = W.load_wishart_data(spectra_path(key, binning), noise_path(key, binning), bands, mode,
                             MASK[key]['fsky'])
    wd.check_positive_definite()
    return wd


class Data:
    """
    A WishartData with what the model needs precomputed: colour corrections,
    beam x pixel window / unit factors and the constant part of the model
    matrix (CMB and noise).  For a bin-to-bin fit, build it from
    wd.select_bins(b).
    """

    def __init__(self, wd):
        self.wd = wd
        self.freqs = np.asarray(wd.freqs, float)
        self.ell = np.asarray(wd.ell_eff, float) / ELL_REF
        self.cc = ColourCorrections(wd.bands, wd.cc_dict)
        t = wd.W / wd.unit[None, :]
        self.tt = t[:, :, None] * t[:, None, :]
        self.base = wd.C_cmb[:, None, None] * (wd.W[:, :, None] * wd.W[:, None, :]) + wd.N
        self.logdet_D = wd.logdet_D
        self.n_data = W.n_data_points(wd)
        # for the SEDs: ln(nu / nu_ref) and h nu / k (K) of every band and of 353 GHz
        self.ln_sync = np.log(self.freqs / NU_REF_SYNC)
        self.ln_dust = np.log(self.freqs / NU_REF_DUST)
        self.hnu_k = h * self.freqs * 1e9 / k
        self.hnu0_k = h * NU_REF_DUST * 1e9 / k


def sed_synchrotron(beta_s, data):
    """Synchrotron SED in K_RJ, normalized at 23 GHz, over the colour corrections."""
    return np.exp(beta_s * data.ln_sync) / data.cc.synchrotron(beta_s)


def mbb(beta_d, T_d, data):
    """
    Dust MBB in K_RJ normalized at 353 GHz, functions.mbb_scaling_KRJ:
    (nu/nu0)^beta B_nu(T)/B_nu0(T) (nu0/nu)^2 = (nu/nu0)^(beta+1) (e^x0 - 1)/(e^x - 1).
    """
    return np.exp((beta_d + 1.0) * data.ln_dust) * (np.expm1(data.hnu0_k / T_d) / np.expm1(data.hnu_k / T_d))


def sed_dust(beta_d, T_d, data):
    """Dust MBB over the colour corrections."""
    return mbb(beta_d, T_d, data) / data.cc.dust(beta_d, T_d)


def foreground(p, data):
    """
    Foreground matrices C_fg (n_bins, p, p) in uK^2_RJ of the paper's model,
    with the dust at T_d = p['T_d']; p: {parameter: value}.
    """
    return foreground_seds(p, data, sed_synchrotron(p['beta_s'], data),
                           sed_dust(p['beta_d'], p['T_d'], data))


def foreground_seds(p, data, s, d):
    """foreground with the SEDs s and d already computed."""
    F_s = p['A_s'] * data.ell ** p['alpha_s']
    F_d = p['A_d'] * data.ell ** p['alpha_d']
    F_x = (p['rho'] * np.sqrt(max(p['A_s'] * p['A_d'], 0.0))
           * data.ell ** (0.5 * (p['alpha_s'] + p['alpha_d'])))
    sd = np.outer(s, d)
    return (F_s[:, None, None] * np.outer(s, s) + F_d[:, None, None] * np.outer(d, d)
            + F_x[:, None, None] * (sd + sd.T))


def deviance(M, data, per_bin=False):
    """
    Wishart deviance sum_b nu_b [Tr(D_b M_b^-1) - ln det(D_b M_b^-1) - p]
    (wishart_likelihood.wishart_deviance); +inf if an M_b is not positive definite.
    per_bin: the term of each bandpower instead of the sum.
    """
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return np.full(len(M), np.inf) if per_bin else np.inf
    logdet_M = 2.0 * np.log(np.einsum('bii->bi', L)).sum(axis=1)
    tr = np.einsum('bii->b', np.linalg.solve(M, data.wd.D_hat))
    dev = data.wd.nu_dof * (tr + logdet_M - data.logdet_D - M.shape[1])
    if per_bin:
        return dev
    dev = float(np.sum(dev))
    return dev if np.isfinite(dev) else np.inf


# ===========================================================================
# Free parameters and priors
# ===========================================================================
class Model:
    """
    The free parameters of one fit, their priors and the model matrix.

    foreground  foreground(p, data) -> C_fg (n_bins, p, p) in uK^2_RJ, p a dict
                with every parameter in `names`
    names       the parameters, in the order of the columns of the samples
    spec        {name: (lower, upper, start, ball, step)}, added to SPEC
    fixed       {name: value} held fixed (T_D_FIXED: the paper's model)
    priors      {name: (mean, sigma)}, Gaussian priors on top of the uniform ones
    """

    def __init__(self, foreground, names=PARAM_NAMES, spec=None, fixed=None, priors=None):
        self.foreground = foreground
        self.names = tuple(names)
        self.spec = dict(SPEC, **(spec or {}))
        self.fixed = {n: float(v) for n, v in (fixed or {}).items()}
        self.priors = dict(priors or {})
        self.free = [n for n in self.names if n not in self.fixed]
        self.n_par = len(self.free)
        lo, hi, start, ball, step = (np.array([self.spec[n][i] for n in self.free], float)
                                     for i in range(5))
        self.lower, self.upper, self.start, self.ball, self.step = lo, hi, start, ball, step
        self.amplitude = ~np.isfinite(hi)
        self._gauss = [(self.free.index(n), mu, sig) for n, (mu, sig) in self.priors.items()
                       if n in self.free]

    def params(self, x):
        """{name: value} of every parameter, from the free ones x."""
        p = dict(self.fixed)
        p.update(zip(self.free, map(float, x)))
        return p

    def full(self, x):
        """Every parameter, in the order of names (samples: one row each)."""
        x = np.atleast_2d(x)
        out = np.empty((x.shape[0], len(self.names)))
        for i, n in enumerate(self.names):
            out[:, i] = self.fixed[n] if n in self.fixed else x[:, self.free.index(n)]
        return out

    def in_box(self, x):
        """Inside the uniform priors (amplitudes > 0)."""
        x = np.asarray(x, float)
        return bool(np.all(x >= self.lower) and np.all(x <= self.upper)
                    and np.all(x[self.amplitude] > 0) and np.all(np.isfinite(x)))

    def gaussian_log_prior(self, x):
        """Log of the Gaussian priors, for one vector x or for samples (n, n_par)."""
        x = np.asarray(x, float)
        return sum(-0.5 * ((x[..., i] - mu) / sig) ** 2 for i, mu, sig in self._gauss)

    def deviance(self, x, data, per_bin=False):
        return deviance(self.foreground(self.params(x), data) * data.tt + data.base, data, per_bin)

    def log_posterior(self, x, data, gaussian=True):
        """Log posterior; gaussian=False: log likelihood within the uniform priors."""
        if not self.in_box(x):
            return -np.inf
        dev = self.deviance(x, data)
        if not np.isfinite(dev):
            return -np.inf
        return -0.5 * dev + (self.gaussian_log_prior(x) if gaussian else 0.0)

    def clip(self, x):
        """x moved just inside the uniform priors."""
        eps = 1e-6
        x = np.array(x, float)
        x[self.amplitude] = np.maximum(x[self.amplitude], 1e-12)
        fin = ~self.amplitude
        x[fin] = np.minimum(np.maximum(x[fin], self.lower[fin] + eps), self.upper[fin] - eps)
        return x


def power_law_model(fixed=None):
    """Global power-law fit; fixed=T_D_FIXED for the paper's model."""
    return Model(foreground, fixed=fixed)


def bin_to_bin_model(fixed=None):
    """One bandpower, alpha_s = alpha_d = 0 and the Gaussian prior on beta_s, as in the paper."""
    return Model(foreground, fixed=dict(BINBIN_FIXED, **(fixed or {})), priors=BINBIN_PRIORS)


def from_paper(model, theta, T_d=W.T_DUST):
    """
    Free parameters of `model` from a 7-vector of the paper
    (wishart_likelihood.PARAM_NAMES) and T_d: the paper's solution as a start.
    """
    p = dict(zip(W.PARAM_NAMES, map(float, theta)), T_d=T_d)
    return np.array([p[n] for n in model.free])


def information_criteria(deviance_ml, n_par, n_data):
    """
    AIC = D_ml + 2k and BIC = D_ml + k ln N, with D_ml the deviance at the
    maximum of the likelihood (-2 ln L up to a constant that cancels between
    models fitted to the same data), k free parameters and N = n_data
    independent entries of the data matrices.
    """
    return deviance_ml + 2.0 * n_par, deviance_ml + n_par * np.log(n_data)


# ===========================================================================
# Maximum of the likelihood or posterior, and MCMC
# ===========================================================================
def maximize(model, data, x0, gaussian=True, n_restarts=3, seed=None):
    """
    Maximum of the posterior (gaussian=True) or of the likelihood within the
    uniform priors (gaussian=False), with Nelder-Mead, amplitudes in log space;
    each restart starts from the best point so far with a random jitter.
    Returns the free parameters there and the deviance.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    amp = model.amplitude

    def unpack(y):
        x = np.array(y, float)
        x[amp] = np.exp(x[amp])
        return x

    def neg(y):
        v = model.log_posterior(unpack(y), data, gaussian)
        return 1e30 if not np.isfinite(v) else -v

    best_y = np.array(model.clip(x0), float)
    best_y[amp] = np.log(best_y[amp])
    best_f = neg(best_y)
    for r in range(max(1, n_restarts)):
        y0 = best_y
        if r > 0:
            for _ in range(100):
                cand = best_y + rng.normal(0, model.step)
                if model.in_box(unpack(cand)):
                    y0 = cand
                    break
        res = minimize(neg, y0, method='Nelder-Mead',
                       options=dict(maxiter=40000, maxfev=40000, xatol=1e-8, fatol=1e-8,
                                    adaptive=True))
        if res.fun < best_f:
            best_f, best_y = res.fun, res.x
    x = unpack(best_y)
    return x, model.deviance(x, data)


# emcee runs the posterior in worker processes, which receive the model and the data once
_WORKER = {}


def _init_worker(model, data):
    _WORKER.update(model=model, data=data)


def _log_posterior_worker(x):
    return _WORKER['model'].log_posterior(x, _WORKER['data'])


def run_mcmc(model, data, nwalkers, nsteps, discard=0.5, seed=42, n_processes=1, x_start=None):
    """
    Sample the posterior with emcee, the walkers starting in a small ball
    around its maximum (as the paper's run_wishart_mcmc), then find the maximum
    of the likelihood from the best samples.  Returns a dict:

      samples       (n, len(model.names)) after discarding the first
                    discard*nsteps steps of every walker, fixed parameters as
                    constant columns, in emcee's order (step-major)
      log_prob      log posterior of each sample
      theta_map     the sample of highest posterior; chi2_reduced the deviance
                    there over n_data - n_par (the paper's chi2_red)
      theta_ml      maximum of the likelihood within the uniform priors, never
                    worse than the best sample; deviance_ml the deviance there
      aic, bic      information_criteria at theta_ml
      tau           integrated autocorrelation time of each free parameter, in steps
      acceptance    mean acceptance fraction of the walkers
    """
    import emcee

    rng = np.random.default_rng(seed)
    x_map, _ = maximize(model, data, model.start if x_start is None else x_start, seed=seed)

    # amplitudes get a relative scatter, the rest an absolute one
    scale = np.where(model.amplitude, model.ball * np.maximum(x_map, 1e-12), model.ball)
    pos = np.empty((nwalkers, model.n_par))
    for w in range(nwalkers):
        for _ in range(200):
            cand = x_map + rng.normal(0, scale)
            if model.in_box(cand):
                pos[w] = cand
                break
        else:
            pos[w] = model.clip(x_map)

    def run(sampler):
        sampler.random_state = np.random.RandomState(seed).get_state()
        sampler.run_mcmc(pos, nsteps, progress=False)
        return sampler

    if n_processes > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(processes=n_processes, initializer=_init_worker,
                                         initargs=(model, data)) as pool:
            sampler = run(emcee.EnsembleSampler(nwalkers, model.n_par, _log_posterior_worker,
                                                pool=pool))
    else:
        sampler = run(emcee.EnsembleSampler(nwalkers, model.n_par, model.log_posterior,
                                            args=(data,)))

    burn = int(discard * nsteps)
    chain = sampler.get_chain(discard=burn, flat=True)
    log_prob = sampler.get_log_prob(discard=burn, flat=True)
    tau = sampler.get_autocorr_time(discard=burn, quiet=True)
    x_best = chain[int(np.argmax(log_prob))]

    # maximum of the likelihood: from the best sample of the posterior and from
    # the best sample of the likelihood (they differ with Gaussian priors)
    dev_samples = -2.0 * (log_prob - model.gaussian_log_prior(chain))
    x_ml, dev_ml = maximize(model, data, x_best, gaussian=False, seed=seed)
    i_best = int(np.argmin(dev_samples))
    if model._gauss or dev_ml > dev_samples[i_best]:
        x2, d2 = maximize(model, data, chain[i_best], gaussian=False, seed=seed + 1)
        if d2 < dev_ml:
            x_ml, dev_ml = x2, d2
    if dev_ml > dev_samples[i_best] + 1e-6:
        x_ml, dev_ml = chain[i_best], float(dev_samples[i_best])

    aic, bic = information_criteria(dev_ml, model.n_par, data.n_data)
    return dict(samples=model.full(chain), log_prob=log_prob,
                theta_map=model.full(x_best)[0],
                chi2_reduced=model.deviance(x_best, data) / (data.n_data - model.n_par),
                theta_ml=model.full(x_ml)[0], deviance_ml=dev_ml,
                deviance_min_sample=float(dev_samples[i_best]),
                aic=aic, bic=bic, n_par=model.n_par, n_data=data.n_data,
                tau=np.asarray(tau, float), acceptance=float(np.mean(sampler.acceptance_fraction)))


# the per-bandpower entries of a bin-to-bin result file
BANDPOWER_KEYS = ('percentiles', 'theta_map', 'chi2_reduced', 'theta_ml', 'deviance_ml',
                  'deviance_min_sample', 'aic', 'bic', 'tau', 'acceptance')


def kept_over_tau(z):
    """Kept steps of each walker over the longest autocorrelation time, from a result file."""
    nsteps = np.broadcast_to(z['nsteps'], np.shape(z['tau'])[:-1])
    return nsteps * (1.0 - float(z['discard'])) / np.max(z['tau'], axis=-1)


def converged_bandpowers(path, rerun_below):
    """
    {bandpower: its entries} of the bin-to-bin file `path`, for the bandpowers
    whose kept chains are at least rerun_below autocorrelation times long;
    {} when rerun_below is None or the file does not exist.
    """
    if rerun_below is None or not os.path.exists(path):
        return {}
    z = np.load(path)
    r = kept_over_tau(z)
    nsteps = np.broadcast_to(z['nsteps'], r.shape)
    return {b: dict({k: z[k][b] for k in BANDPOWER_KEYS}, nsteps=int(nsteps[b]), n_data=int(z['n_data']))
            for b in map(int, np.flatnonzero(r >= rerun_below))}


def summarize(samples, names=PARAM_NAMES):
    """{parameter: (median, upper error, lower error)}, 68 per cent interval."""
    lo, med, hi = np.percentile(samples, [15.865, 50.0, 84.135], axis=0)
    return {n: (med[i], hi[i] - med[i], med[i] - lo[i]) for i, n in enumerate(names)}
