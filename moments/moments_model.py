"""
Wishart likelihood with first-order moments of the synchrotron and dust SEDs.
=============================================================================

The model with T_d free (t_dust/td_model.py) plus the first-order moment
terms of the cross-spectra: Vacher et al. (2022, A&A 660, A111), Eq. 8, which
Vinzl et al. (2026, arXiv:2607.20080, Sect. 4.4) use for "all auto- and
cross-spectra of dust and synchrotron moments up to first order".

For a component c with SED e_c(nu; p) (colour-corrected, K_RJ, normalized
at its reference frequency) and spectral parameters p_k, the expansion of the
emission to first order around pivots p̄,

    S_c(nu, n) = A_c(n) e_c(nu; p̄) + sum_k omega^k(n) de_c/dp_k,   omega^k = A_c (p_k - p̄_k),

gives in the cross-spectra A x A terms (the model of the paper), A x omega^k
terms and omega^k x omega^k' terms.  An A x omega^k term is a shift of the
pivot p̄_k: Vinzl et al. update their pivots until these terms vanish (their
Eq. 4.1).  Here the pivots beta_s, beta_d and T_d are free parameters of the
fit, which absorbs them.  What remains are the omega x omega terms,

    C^c(nu_i, nu_j) += W_c(l) sum_{k,k'} w_c^{kk'} de_c/dp_k(nu_i) de_c/dp_k'(nu_j),

  synchrotron   p = (beta_s):        w_s_bb                        1 parameter
  dust          p = (beta_d, T_d):   w_d_bb, w_d_TT, w_d_bT        3 parameters

with W_c(l) = A_c (l/80)^alpha_c the amplitude spectrum of the component.
w_c^{kk'} is the moment spectrum relative to it: w_bb is about the variance of
beta across the mask (weighted by the squared amplitude), w_d_TT that of T_d
(K^2) and w_d_bT their covariance (K).  In the power-law fit the w are constant,
so the moment spectra have the multipole slope of their component (one parameter
per term).

The derivatives are those of the colour-corrected SEDs, i.e. of the
band-integrated emission:

    d ln e_s / d beta_s = ln(nu / 23 GHz) - d ln cc_s / d beta_s
    d ln e_d / d beta_d = ln(nu / 353 GHz) - d ln cc_d / d beta_d
    d ln e_d / d T_d    = Theta(nu, T_d) - Theta(353 GHz, T_d) - d ln cc_d / d T_d,
    Theta(nu, T) = d ln B_nu / dT = x e^x / [T (e^x - 1)],   x = h nu / k T

(Vacher et al. 2022, Eq. 5, expand in T, not in 1/T; the two differ by the
factor -T^2 at first order.)

The w can have either sign; only the model matrix must stay positive
definite.  So the model without moments, w = 0, is inside the prior, and a
significantly negative auto term would flag a mismodeling.  There are no
dust x synchrotron moment terms: the rho term of the paper is kept as it is.
"""

import mconfig as MC  # noqa: F401  (paths and noise term, before the paper's modules)

import numpy as np  # noqa: E402

import td_model as TD  # noqa: E402

MOMENT_PARAMS = {'sync': ('w_s_bb',), 'dust': ('w_d_bb', 'w_d_TT', 'w_d_bT')}
COMPONENTS = {'sync': ('sync',), 'dust': ('dust',), 'both': ('sync', 'dust')}

# (lower, upper, start, walkers' ball, optimizer restart step), as td_model.SPEC.
# Wide enough for sigma_beta up to 1 and sigma_T up to 10 K
SPEC = {
    'w_s_bb': (-1.0, 1.0, 0.0, 0.005, 0.02),
    'w_d_bb': (-1.0, 1.0, 0.0, 0.005, 0.02),
    'w_d_TT': (-100.0, 100.0, 0.0, 0.5, 2.0),
    'w_d_bT': (-10.0, 10.0, 0.0, 0.05, 0.2),
}


def theta_planck(hnu_k, T):
    """d ln B_nu / dT = x e^x / [T (e^x - 1)], x = h nu / k T, in 1/K; hnu_k = h nu / k in K."""
    x = hnu_k / T
    return x / (T * -np.expm1(-x))


def sed_derivatives_synchrotron(beta_s, data, s=None):
    """de_s / d beta_s for every band; s: the SED, if already computed."""
    if s is None:
        s = TD.sed_synchrotron(beta_s, data)
    return s * (data.ln_sync - data.cc.synchrotron_dlog(beta_s))


def sed_derivatives_dust(beta_d, T_d, data, d=None, cc=None):
    """de_d / d beta_d and de_d / d T_d for every band; d, cc: the SED and its colour corrections, if known."""
    if cc is None:
        cc = data.cc.dust(beta_d, T_d)
    if d is None:
        d = TD.mbb(beta_d, T_d, data) / cc
    dlog_beta, dlog_T = data.cc.dust_dlog(beta_d, T_d, cc)
    d_beta = d * (data.ln_dust - dlog_beta)
    d_T = d * (theta_planck(data.hnu_k, T_d) - theta_planck(data.hnu0_k, T_d) - dlog_T)
    return d_beta, d_T


class Foreground:
    """td_model.foreground plus the omega x omega terms of `components` ('sync', 'dust')."""

    def __init__(self, components):
        self.sync = 'sync' in components
        self.dust = 'dust' in components

    def __call__(self, p, data):
        s = TD.sed_synchrotron(p['beta_s'], data)
        cc_d = data.cc.dust(p['beta_d'], p['T_d'])
        d = TD.mbb(p['beta_d'], p['T_d'], data) / cc_d
        C = TD.foreground_seds(p, data, s, d)
        if self.sync:
            ds = sed_derivatives_synchrotron(p['beta_s'], data, s)
            W_s = p['A_s'] * data.ell ** p['alpha_s']
            C = C + (p['w_s_bb'] * W_s)[:, None, None] * np.outer(ds, ds)
        if self.dust:
            db, dT = sed_derivatives_dust(p['beta_d'], p['T_d'], data, d, cc_d)
            W_d = p['A_d'] * data.ell ** p['alpha_d']
            bT = np.outer(db, dT)
            m = (p['w_d_bb'] * np.outer(db, db) + p['w_d_TT'] * np.outer(dT, dT)
                 + p['w_d_bT'] * (bT + bT.T))
            C = C + W_d[:, None, None] * m
        return C


def param_names(moments):
    """The parameters of the model with the moments `moments` ('sync', 'dust' or 'both')."""
    return TD.PARAM_NAMES + tuple(n for c in COMPONENTS[moments] for n in MOMENT_PARAMS[c])


def power_law_model(moments, fixed=None):
    return TD.Model(Foreground(COMPONENTS[moments]), param_names(moments), spec=SPEC, fixed=fixed)


def at_prior_edge(p, frac=0.01):
    """The moment parameters of p ({name: value}) within frac of their prior range from a bound."""
    return [n for n, (lo, hi, *_) in SPEC.items()
            if np.isfinite(p.get(n, np.nan)) and min(p[n] - lo, hi - p[n]) < frac * (hi - lo)]


def dust_moments_psd(p):
    """
    Whether the dust moments of p form a covariance matrix,
    [[w_d_bb, w_d_bT], [w_d_bT, w_d_TT]] positive semidefinite, as moments of
    a real distribution of beta_d and T_d must.
    """
    return p['w_d_bb'] >= 0 and p['w_d_TT'] >= 0 and p['w_d_bT'] ** 2 <= p['w_d_bb'] * p['w_d_TT']


def start_from(model, theta, names=TD.PARAM_NAMES):
    """Free parameters of `model` from a full vector `theta` over `names`, the missing moments at 0."""
    p = dict(zip(names, map(float, theta)))
    return np.array([p.get(n, 0.0) for n in model.free])
