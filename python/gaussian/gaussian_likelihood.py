"""
Gaussian likelihood of the corrected spectra (Appendix A).
==========================================================

Every bandpower of every auto- and cross-spectrum is an independent Gaussian
variable, with the uncertainty of the corrected spectra (correct_spectra.py):

    -2 ln L = sum_b sum_{i<=j} [C_b^ij - C^fg_b(nu_i, nu_j)]^2 / (sigma_b^ij)^2.

Same foreground model, colour corrections, central frequencies and priors as
the Wishart fits; amplitudes in mK^2_RJ.  The global power-law fit has the
seven parameters (A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho); the
bin-to-bin fit has (A_s, beta_s, A_d, beta_d, rho) per bandpower, with a
Gaussian prior on beta_s.
"""

import multiprocessing as mp
import os
import sys

import emcee
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import functions as F  # noqa: E402
from data import freq_ghz  # noqa: E402

PARAMS_PL = ['A_s', 'alpha_s', 'beta_s', 'A_d', 'alpha_d', 'beta_d', 'rho']
PARAMS_BTB = ['A_s', 'beta_s', 'A_d', 'beta_d', 'rho']
BETA_S_PRIOR_BTB = (-3.1, 0.30)


def load_fit_data(path_corrected, bands, mode):
    """
    The corrected spectra of all the pairs i <= j of `bands` for mode 'EE' or
    'BB', as one data vector: {'ell_eff', 'y_all', 'yerr_all', 'datasets'},
    each dataset being one pair with its slice of the vector and the central
    frequencies of its bands.
    """
    pairs = [(a, b) for i, a in enumerate(bands) for b in bands[i:]]
    spectra = F.read_pairs(path_corrected, bands, ['ell_eff', f'{mode}_SPECTRUM', f'{mode}_ERROR'])
    datasets, y, yerr = [], [], []
    for a, b in pairs:
        s = spectra[f'{a}_{b}']
        start = sum(len(v) for v in y)
        y.append(s[f'{mode}_SPECTRUM'])
        yerr.append(s[f'{mode}_ERROR'])
        datasets.append({'pair': f'{a}_{b}', 'freqs': (freq_ghz(a), freq_ghz(b)),
                         'spectrum': y[-1], 'error': yerr[-1], 'slice': (start, start + len(y[-1]))})
    return {'ell_eff': spectra[f'{bands[0]}_{bands[0]}']['ell_eff'],
            'y_all': np.concatenate(y), 'yerr_all': np.concatenate(yerr), 'datasets': datasets}


def _cc(cc_dict, comp, band, alpha):
    """Colour correction of `band` for component comp ('synch' or 'dust') at alpha."""
    poly = cc_dict[comp][band]
    return poly[0] + poly[1]*alpha + poly[2]*(alpha**2)


# ===========================================================================
# Global power-law fit
# ===========================================================================
def model_synchrotron(theta, datasets, ell, cc_dict, freq_ref=23., ell_ref=80.0):
    A_s, alpha_s, beta_s = theta[:3]
    model = []
    for d in datasets:
        band1, band2 = d['pair'].split('_')
        f1, f2 = d['freqs']
        scale_f1 = (f1 / freq_ref) ** beta_s
        scale_f2 = (f2 / freq_ref) ** beta_s
        ell_scale = (ell / ell_ref) ** alpha_s
        alpha_cc = 2.0 + float(beta_s)
        cc_s1 = _cc(cc_dict, 'synch', band1, alpha_cc)
        cc_s2 = _cc(cc_dict, 'synch', band2, alpha_cc)
        Cl = A_s * ell_scale * scale_f1 * scale_f2
        # the data are not colour corrected, so the model is divided by the corrections
        Cl = Cl / (cc_s1 * cc_s2)
        model.append(Cl)
    return np.concatenate(model)


def model_dust(theta, datasets, ell, cc_dict, freq_ref=353.0, T_d=19.6, ell_ref=80.0):
    A_d, alpha_d, beta_d = theta[:3]
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}
    S = F.mbb_scaling_KRJ(np.array(unique_freqs), nu0_GHz=freq_ref, beta=beta_d, T_d=T_d)
    model = []
    for d in datasets:
        band1, band2 = d['pair'].split('_')
        f1, f2 = d['freqs']
        s1 = S[freq_to_idx[f1]]
        s2 = S[freq_to_idx[f2]]
        ell_scale = (ell / ell_ref) ** alpha_d
        alpha_cc = 2.0 + float(beta_d)
        cc_d1 = _cc(cc_dict, 'dust', band1, alpha_cc)
        cc_d2 = _cc(cc_dict, 'dust', band2, alpha_cc)
        model.append(A_d * ell_scale * s1 * s2 / (cc_d1 * cc_d2))
    return np.concatenate(model)


def model_cross(theta, datasets, ell, cc_dict, ref_sync=23., ref_dust=353.0, T_d=19.6,
                ell_ref=80.0):
    """Synchrotron-dust correlation term; theta = (rho, A_s, A_d, alpha_s, alpha_d, beta_s, beta_d)."""
    rho, A_s, A_d, alpha_s, alpha_d, beta_s, beta_d = theta
    unique_freqs = sorted({f for d in datasets for f in d['freqs']})
    freq_to_idx = {f: i for i, f in enumerate(unique_freqs)}
    Sd = F.mbb_scaling_KRJ(np.array(unique_freqs), nu0_GHz=ref_dust, beta=beta_d, T_d=T_d)
    model = []
    for d in datasets:
        band1, band2 = d['pair'].split('_')
        f1, f2 = d['freqs']
        s1 = (f1 / ref_sync) ** beta_s
        s2 = (f2 / ref_sync) ** beta_s
        d1 = Sd[freq_to_idx[f1]]
        d2 = Sd[freq_to_idx[f2]]
        ell_scale_cross = (ell / ell_ref) ** ((alpha_s + alpha_d) / 2)
        alpha_s_cc = 2.0 + float(beta_s)
        alpha_d_cc = 2.0 + float(beta_d)
        cc_s1 = _cc(cc_dict, 'synch', band1, alpha_s_cc)
        cc_s2 = _cc(cc_dict, 'synch', band2, alpha_s_cc)
        cc_d1 = _cc(cc_dict, 'dust', band1, alpha_d_cc)
        cc_d2 = _cc(cc_dict, 'dust', band2, alpha_d_cc)
        mix = ((s1 / cc_s1) * (d2 / cc_d2) + (s2 / cc_s2) * (d1 / cc_d1))
        As_Ad = A_s * A_d
        model.append(rho * np.sign(As_Ad) * np.sqrt(As_Ad) * mix * ell_scale_cross)
    return np.concatenate(model)


def model_pl(theta, datasets, ell, cc_dict):
    A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho = theta[:7]
    y_model = np.zeros(sum(len(d['spectrum']) for d in datasets))
    y_model += model_synchrotron([A_s, alpha_s, beta_s], datasets, ell, cc_dict)
    y_model += model_dust([A_d, alpha_d, beta_d], datasets, ell, cc_dict)
    y_model += model_cross([rho, A_s, A_d, alpha_s, alpha_d, beta_s, beta_d], datasets, ell, cc_dict)
    return y_model


def lnprior_pl(theta):
    """Uniform priors, as the Wishart fits."""
    A_s, alpha_s, beta_s, A_d, alpha_d, beta_d, rho = theta[:7]
    if A_s <= 0: return -np.inf                                         # noqa: E701
    if not (-6 <= alpha_s <= 0): return -np.inf                         # noqa: E701
    if not (-6 <= beta_s <= 0): return -np.inf                          # noqa: E701
    if A_d <= 0: return -np.inf                                         # noqa: E701
    if not (-6 <= alpha_d <= 0): return -np.inf                         # noqa: E701
    if not (0 <= beta_d <= 6): return -np.inf                           # noqa: E701
    if not (-1 <= rho <= 1): return -np.inf                             # noqa: E701
    return 0.0


def lnprob_pl(theta, datasets, ell, y_all, yerr_all, cc_dict):
    theta = np.array(theta, dtype=float)
    lp = lnprior_pl(theta)
    if not np.isfinite(lp):
        return -np.inf
    residual = y_all - model_pl(theta, datasets, ell, cc_dict)
    return lp + -0.5 * np.sum((residual / yerr_all) ** 2)


def chi2_reduced_pl(theta, fit, cc_dict):
    y_model = model_pl(theta, fit['datasets'], fit['ell_eff'], cc_dict)
    chi2 = np.sum(((fit['y_all'] - y_model) / fit['yerr_all']) ** 2)
    return chi2 / (len(fit['y_all']) - len(PARAMS_PL))


def run_power_law(fit, nwalkers, nsteps, discard_fraction=0.5, n_processes=1, seed=42):
    """
    emcee sampling of the power-law posterior: the walkers start around fixed
    values, run max(100, nsteps/5) burn-in steps, are reset and run nsteps
    more, of which the first discard_fraction is discarded.  Returns the
    sampler, the samples and chi2_red at the maximum-posterior sample.
    """
    cc_dict = F.load_color_correction_polynomials()
    rng = np.random.default_rng(seed)
    center = np.array([1., -3., -3., 1., -2.3, 1.59, 0.05])
    spread = np.array([0.1 * abs(center[0]), 0.05, 0.05, 0.1 * abs(center[3]), 0.05, 0.05, 0.02])
    p0 = center + spread * rng.standard_normal((nwalkers, len(PARAMS_PL)))
    # every walker inside the prior
    p0[:, 0] = np.clip(p0[:, 0], 1e-6, None)
    p0[:, 2] = np.clip(p0[:, 2], -5.0, -0.5)
    p0[:, 3] = np.clip(p0[:, 3], 1e-6, None)
    p0[:, 5] = np.clip(p0[:, 5], 0.5, 3.0)
    p0[:, 6] = np.clip(p0[:, 6], -0.99, 0.99)

    with mp.Pool(processes=n_processes) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, len(PARAMS_PL), lnprob_pl,
            args=(fit['datasets'], fit['ell_eff'], fit['y_all'], fit['yerr_all'], cc_dict), pool=pool)
        sampler.random_state = np.random.RandomState(seed).get_state()
        p0, _, _ = sampler.run_mcmc(p0, max(100, int(0.2 * nsteps)), progress=False)
        sampler.reset()
        sampler.run_mcmc(p0, nsteps, progress=False)

    discard = int(nsteps * discard_fraction)
    samples = sampler.get_chain(discard=discard, flat=True)
    best = samples[np.argmax(sampler.get_log_prob(discard=discard, flat=True))]
    return sampler, samples, chi2_reduced_pl(best, fit, cc_dict)


# ===========================================================================
# Bin-to-bin fit: no multipole dependence, one bandpower at a time
# ===========================================================================
def _model_btb(theta, datasets, cc_dict):
    A_s, beta_s, A_d, beta_d, rho = theta
    freq_ref_sync, freq_ref_dust, T_d = 23., 353.0, 19.6
    y_model = np.zeros(len(datasets))
    for idx, d in enumerate(datasets):
        f1, f2 = d['freqs']
        b1, b2 = d['pair'].split('_')
        alpha_s_cc = 2.0 + float(beta_s)
        alpha_d_cc = 2.0 + float(beta_d)
        cc_s1 = _cc(cc_dict, 'synch', b1, alpha_s_cc)
        cc_s2 = _cc(cc_dict, 'synch', b2, alpha_s_cc)
        cc_d1 = _cc(cc_dict, 'dust', b1, alpha_d_cc)
        cc_d2 = _cc(cc_dict, 'dust', b2, alpha_d_cc)
        s1 = (f1 / freq_ref_sync) ** beta_s
        s2 = (f2 / freq_ref_sync) ** beta_s
        d1 = F.mbb_scaling_KRJ(np.array([f1]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
        d2 = F.mbb_scaling_KRJ(np.array([f2]), nu0_GHz=freq_ref_dust, beta=beta_d, T_d=T_d)[0]
        model_val = 0.0
        model_val += A_s * (s1 / cc_s1) * (s2 / cc_s2)
        model_val += A_d * (d1 / cc_d1) * (d2 / cc_d2)
        model_val += rho * np.sqrt(A_s * A_d) * ((s1 / cc_s1) * (d2 / cc_d2) + (s2 / cc_s2) * (d1 / cc_d1))
        y_model[idx] = model_val
    return y_model


def _lnprob_btb(theta, datasets, y, yerr, cc_dict, beta_s_prior):
    A_s, beta_s, A_d, beta_d, rho = theta
    if A_s <= 0 or A_d <= 0:
        return -np.inf
    if not (-6 <= beta_s <= 0) or not (0 <= beta_d <= 6) or not (-1 <= rho <= 1):
        return -np.inf
    mu, sig = beta_s_prior
    lp = 0.0
    lp += -0.5 * ((beta_s - mu)/sig)**2
    ll = -0.5 * np.sum(((y - _model_btb(theta, datasets, cc_dict)) / yerr) ** 2)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def run_bin_to_bin(fit, nwalkers, nsteps, discard_fraction=0.5, n_processes=1, seed=42,
                   beta_s_prior=BETA_S_PRIOR_BTB):
    """
    emcee sampling of each bandpower on its own (no burn-in run; the first
    discard_fraction of every chain is discarded).  Returns, per bandpower,
    the samplers, the samples and chi2_red at the maximum-posterior sample.
    """
    cc_dict = F.load_color_correction_polynomials()
    samplers, samples, chi2_reduced = [], [], []
    for i in range(len(fit['ell_eff'])):
        y = np.array([d['spectrum'][i] for d in fit['datasets']])
        yerr = np.array([d['error'][i] for d in fit['datasets']])
        datasets = [{'pair': d['pair'], 'freqs': d['freqs']} for d in fit['datasets']]

        # amplitudes start at the median of the data of this bandpower
        rng = np.random.default_rng(seed + i)
        rough_scale = float(np.median(np.abs(y)))
        center = np.array([max(rough_scale, 1e-12), -3.0, max(rough_scale, 1e-12), 1.59, 0.05])
        p0 = center + np.maximum(np.abs(center) * 0.2, 1e-12) * rng.standard_normal((nwalkers, len(center)))

        with mp.Pool(processes=n_processes) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, len(center), _lnprob_btb,
                                            args=(datasets, y, yerr, cc_dict, beta_s_prior), pool=pool)
            sampler.random_state = np.random.RandomState(seed + i).get_state()
            sampler.run_mcmc(p0, nsteps, progress=False)

        discard = int(nsteps * discard_fraction)
        s = sampler.get_chain(discard=discard, flat=True)
        best = s[np.argmax(sampler.get_log_prob(discard=discard, flat=True))]
        chi2 = np.sum(((y - _model_btb(best, datasets, cc_dict)) / yerr) ** 2)
        samplers.append(sampler)
        samples.append(s)
        chi2_reduced.append(chi2 / max(1, len(y) - len(PARAMS_BTB)))
    return samplers, samples, chi2_reduced
