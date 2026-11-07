# Fits with the dust temperature free

The Wishart power-law and bin-to-bin fits of the paper (binning dl20), with the
dust temperature T_d as a free parameter instead of fixed at 19.6 K. The data
spectra, noise term, masks, beams, unit conversions, CMB and the other priors are
the paper's. The code and results in `../python/` and `../results/` are only read:
everything this folder produces stays inside it.

## What changes with respect to the paper

| | Paper | Here |
|---|---|---|
| Parameters | A_s, α_s, β_s, A_d, α_d, β_d, ρ | the same plus T_d, uniform prior 10–40 K |
| Dust SED | MBB at 19.6 K | MBB at T_d |
| HFI dust colour corrections | quadratic in β_d at 19.6 K | surface in (β_d, 1/T_d), `make_cc_surface.py` |
| QUIJOTE/WMAP/LFI dust colour corrections | power law at α = 2 + β_d | unchanged |
| Synchrotron colour corrections | fastcc | unchanged |
| K_CMB → K_RJ | | unchanged: it depends only on the bandpass, not on the SED |

**Colour corrections.** The bandpass-shifted HFI table
(`data/color_corrections/c_td_10-40_beta_hfi_bps_pr3.fits`, variant [1], the one
the paper uses) is a grid over 10 ≤ T_d ≤ 40 K and 1.0 ≤ β_d ≤ 2.4. Each HFI band
gets a polynomial surface, quadratic in β_d − 1.5 and quartic in 20 K/T_d, fitted
once to the whole grid:

- it reproduces the table to ≤ 2.2×10⁻⁵, the same accuracy as the paper's
  polynomial at 19.6 K;
- at T_d = 19.6 K it equals the paper's polynomial to ≤ 8×10⁻⁸ for every β_d in
  the prior (0–6), extrapolation included, so the paper's model is exactly this
  one with T_d fixed (`check_model.py`);
- nothing is integrated over the bandpasses during the sampling, and the
  likelihood is about twice as fast as the paper's, because the bandpowers are
  processed together.

The prior of T_d is the range of the table. The QUIJOTE/WMAP/LFI dust corrections
keep the paper's power law at α = 2 + β_d: the curvature of the MBB would change
them by < 10⁻³ (at 70 GHz), and dust is negligible there.

**What to expect.** The polarized data stop at 353 GHz, on the Rayleigh–Jeans side
of the dust peak, where T_d and β_d are degenerate, so T_d can be set by its prior
where the dust is faint and in the single bandpowers of the bin-to-bin fit.

## Does T_d improve the fit? AIC and BIC

The two models are nested, since T_d = 19.6 K is inside the prior.
`compare_models.py` maximizes the likelihood of both with the same code: the
T_d-free maximum comes from the best MCMC samples, and the T_d-fixed one is
searched from that maximum and from the paper's solution. With D_ml the deviance
at the maximum of the likelihood, k the free parameters and N the independent
entries of the data matrices (945 for the QJ+WP+Pl power law, 105 per bandpower):

    AIC = D_ml + 2k,   BIC = D_ml + k ln N
    ΔAIC = AIC(T_d free) − AIC(19.6 K) = 2 − ΔD,   ΔBIC = ln N − ΔD,   ΔD = D_ml(19.6 K) − D_ml(free) ≥ 0

Negative values favour T_d free. The usual reading of |Δ| is: < 2 no real
preference, 2–6 positive, 6–10 strong, > 10 very strong (Kass & Raftery 1995;
Burnham & Anderson 2002). AIC asks whether the extra parameter improves the
predictions, and BIC, with its larger penalty, whether the data favour the larger
model. For the bin-to-bin fit, the nine bandpowers are also combined into one
model (ΔAIC summed, ΔBIC with N = 945).

Caveat: the Wishart fits of the paper have χ²_red ≈ 1.4–1.9. If part of that excess
comes from the likelihood rather than the sky model, every ΔD is inflated by about
the same factor, and AIC and BIC favour extra parameters too readily. Dividing ΔD
by the χ²_red of the larger model is the conservative reading.

## Running

From `t_dust/`, after the spectra and the Wishart fits of the paper. The noise term
is the HMDM spectra by default; with `SPECIFIC_REGIONS_NOISE=sims` it is the mean of
the noise simulations, and the outputs go to `results/wishart/`.

```bash
python make_cc_surface.py        # once: data/cc_dust_hfi_surface.fits (done)
python check_model.py            # T_d = 19.6 K reproduces the paper
python run_mcmc_power_law.py     # 28 fits, ~2 min each on one core
python run_mcmc_bin_to_bin.py    # 126 bandpowers, ~40 s each
python compare_models.py         # maxima of the likelihood, AIC, BIC
```

The MCMC scripts accept `--masks galcut10,fan,...` (all by default), `--modes EE,BB`,
`--nsteps` and `--nproc`; running one mask per process is the fastest way to use
several cores. The power law runs 64 walkers × 16 000 steps and the bin to bin
64 × 12 000 steps, the first half discarded. That is more than the paper (8000 and
5000), because T_d lengthens the autocorrelation time τ to ~80–150 steps, and more
where T_d is prior-dominated. Every result file stores τ.

Fits whose kept half is shorter than 30 τ can be run again with more steps. With
`--rerun-below 30`, the bin-to-bin script runs again only those bandpowers of an
existing file:

```bash
python run_mcmc_power_law.py --masks north --datasets QJ+WP+Pl --modes BB --nsteps 60000
python run_mcmc_bin_to_bin.py --masks south --modes BB --nsteps 60000 --rerun-below 30
```

## Files

| File | Contents |
|---|---|
| `tdconfig.py` | noise term (HMDM by default), paths to the paper's modules, outputs |
| `make_cc_surface.py` | HFI dust colour-correction surface → `data/cc_dust_hfi_surface.fits` |
| `td_model.py` | the model with T_d free, priors, maxima, emcee (also used by `../moments/`) |
| `check_model.py` | checks against the paper's likelihood |
| `run_mcmc_power_law.py` | → `results/wishart_hmdm/pl_wishart_td_<mask>_<mode><tag>_dl20.npz` |
| `run_mcmc_bin_to_bin.py` | → `results/wishart_hmdm/btb_wishart_td_<mask>_<mode>_dl20.npz` |
| `compare_models.py` | → `results/wishart_hmdm/model_comparison_td_dl20.npz` |

Power-law files: samples and log posterior (every 4th step, float32), `theta_map`
and `chi2_reduced` (maximum-posterior sample, as the paper), `theta_ml`,
`deviance_ml`, `aic`, `bic`, `tau`, `acceptance`. Bin-to-bin files: the 2.275,
15.865, 50, 84.135 and 97.725 percentiles of every parameter per bandpower, and the
same per-bandpower quantities.
