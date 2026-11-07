# Power-law fits with first-order moments of the SEDs

The Wishart power-law fits of `../t_dust/` (T_d free), extended with the first-order
moment expansion of the dust and synchrotron SEDs at the power-spectrum level, in
three combinations: synchrotron moment only, dust moments only, and both. The code
and results in `../python/` and `../results/` are only read.

## The model

Vinzl et al. (2026, arXiv:2607.20080, Sect. 4.4) replace their SED by Eq. 8 of
Vacher et al. (2022, A&A 660, A111), "including all auto- and cross-spectra of dust
**and synchrotron** moments up to first order".

For a component with SED e(ν; p) and spectral parameters p_k, the first-order
expansion of the emission around pivots p̄,

    S(ν, n) = A(n) e(ν; p̄) + Σ_k ω^k(n) ∂e/∂p_k,     ω^k(n) = A(n) [p_k(n) − p̄_k],

gives three kinds of terms in the cross-spectra:

- **A×A**: the model of the paper.
- **A×ω^k**: equivalent to a shift of the pivot p̄_k. Vinzl et al. update their
  pivots until these terms vanish (their Eq. 4.1). Here β_s, β_d and T_d are free
  parameters, which absorbs these terms.
- **ω^k×ω^k′**: what remains, and what is added here:

      C_ij += W_c(ℓ) Σ_kk′ w_c^{kk′} ∂e_c/∂p_k(ν_i) ∂e_c/∂p_k′(ν_j),     W_c(ℓ) = A_c (ℓ/80)^α_c

| Set | Parameters added to the 8 of `t_dust/` |
|---|---|
| `sync` | w_s^ββ |
| `dust` | w_d^ββ, w_d^TT, w_d^βT (both dust parameters β_d and T_d, and their cross term) |
| `both` | the four |

w^ββ is roughly the variance of β across the mask (weighted by the squared
amplitude), w_d^TT that of T_d (K²) and w_d^βT their covariance (K). Following
Vacher et al. (2022, Eq. 5), the dust expansion is in T, not 1/T; at first order the
two differ only by the factor −T².

**Derivatives** are those of the colour-corrected SEDs, i.e. of the band-integrated
emission. They include the derivatives of the colour corrections, analytic from the
polynomials and the HFI surface of `t_dust/` (`check_model.py` compares them with
finite differences):

    ∂ln e_s/∂β_s = ln(ν/23 GHz) − ∂ln cc_s/∂β_s
    ∂ln e_d/∂β_d = ln(ν/353 GHz) − ∂ln cc_d/∂β_d
    ∂ln e_d/∂T_d = Θ(ν, T_d) − Θ(353 GHz, T_d) − ∂ln cc_d/∂T_d,   Θ = x eˣ / [T (eˣ − 1)], x = hν/kT

**Multipole dependence.** Each w is a constant, so each moment spectrum has the
slope α_c of its component, with one parameter per term. An amplitude and a slope per
term (as Azzoni et al. 2021 do for the spectrum of β) would add unconstrained
parameters, since a slope is undefined when its amplitude is near 0.

**Priors.** Uniform, either sign: −1 ≤ w^ββ ≤ 1, −100 ≤ w_d^TT ≤ 100 K²,
−10 ≤ w_d^βT ≤ 10 K, and only the model matrix must stay positive definite. So the
model without moments (w = 0) is inside the prior, and a significantly negative auto
term would flag a mismodeling rather than a moment. There are no dust×synchrotron
moment terms; the paper's ρ term is kept as it is.

## Is a moment set justified? AIC and BIC of five nested models

`compare_models.py` compares, fit by fit, the maxima of the likelihood of

| Model | k |
|---|---|
| paper: T_d = 19.6 K, no moments | 7 |
| T_d free (`t_dust/`) | 8 |
| + synchrotron moment | 9 |
| + dust moments | 11 |
| + both | 12 |

using AIC = D_ml + 2k and BIC = D_ml + k ln N, as in `../t_dust/README.md`, where
the reading and the caveat on χ²_red > 1 are explained. The models are nested: paper
⊂ T_d free ⊂ {+sync, +dust} ⊂ +both. A larger model can never have a lower likelihood
maximum; if its optimizer ends below a nested model, it is restarted from that model's
maximum (extra w = 0).

Two checks to apply at the maximum of the likelihood, in `moments_model.py`:

- `at_prior_edge`: a w at the edge of its prior, where the likelihood still improves
  beyond it;
- `dust_moments_psd`: dust moments that are not a covariance matrix, w_d^ββ < 0,
  w_d^TT < 0 or (w_d^βT)² > w_d^ββ w_d^TT, which no distribution of β_d and T_d can
  give. The free sign of the prior allows it.

## Running

`../t_dust/` first, through `compare_models.py`: the moment fits start from its
T_d-free solutions and the comparison reads its maxima. Then, from `moments/`:

```bash
python check_model.py                        # w = 0 is t_dust's model; derivatives
python run_mcmc_power_law.py --moments all   # 3 x 28 fits
python compare_models.py
```

The script accepts `--masks`, `--datasets`, `--modes`, `--nproc` and `--nsteps` as in
`t_dust/`, plus `--moments sync,dust,both`. The noise term is the HMDM spectra unless
`SPECIFIC_REGIONS_NOISE=sims`. It runs 64 walkers × 24 000 steps, the first half
discarded. Fits whose kept half is shorter than 30 τ can be run again with more steps:

```bash
python run_mcmc_power_law.py --moments both --masks north --modes BB --datasets QJ+WP+Pl --nsteps 60000
```

## Files

| File | Contents |
|---|---|
| `mconfig.py` | paths (through `../t_dust/tdconfig.py`) and outputs |
| `moments_model.py` | moment terms, SED derivatives, the models |
| `check_model.py` | checks |
| `run_mcmc_power_law.py` | → `results/wishart_hmdm/pl_wishart_mom_<set>_<mask>_<mode><tag>_dl20.npz` |
| `compare_models.py` | → `results/wishart_hmdm/model_comparison_moments_dl20.npz` |

The result files have the same contents as the power-law files of `t_dust/` (see its
README).
