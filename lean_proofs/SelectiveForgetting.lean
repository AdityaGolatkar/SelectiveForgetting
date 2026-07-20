import Basic
import ReadoutContraction
import LocalForgettingBound
import InformationBound
import QuadraticScrubbing
import RobustScrubbing

/-!
# Eternal Sunshine of the Spotless Net — formalized kernel (umbrella module)

Lean 4 + Mathlib formalization of the self-contained mathematical results of

> A. Golatkar, A. Achille, S. Soatto,
> *Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks*,
> CVPR 2020, arXiv:1911.04933.

All five kernel results of Appendix C are fully proved and machine-checked:

* Log-sum inequality (proof engine): `log_sum_inequality` — `Basic`
* KL ≥ 0 (Gibbs): `klDiv_nonneg` — `Basic`
* **Lemma 1** (readout DPI): `klDiv_pushforward_le` — `ReadoutContraction`
* **Proposition 1** (information bound): `mutualInfo_le_expected_klDiv`,
  `information_bound` — `InformationBound`
* **Proposition 2** (local forgetting bound): `klDiv_mixture_le` — `LocalForgettingBound`
* **Proposition 3** (optimal quadratic scrubbing): `quadratic_scrubbing_flow_identity`,
  `quadratic_scrubbing_flow_identity_exp` — `QuadraticScrubbing`
* **Proposition 4** (robust isotropic scrubbing): `robust_scrubbing_isotropic_bound`,
  `robust_scrubbing_isotropic_optimality_condition`, `robust_scrubbing_isotropic_attains`
  — `RobustScrubbing`

The information-theoretic core (Lemma 1, Props 1–2) is built self-contained over finite types
from a single `log_sum_inequality` (Jensen for `x ↦ x log x`), faithful to the paper's own
discrete proofs. Props 3–4 are linear-algebra identities over `Matrix _ _ ℝ`. Two honest scope
notes are documented in the respective modules: Prop 3 formalizes the **corrected** middle-term
sign that the paper's *proof* derives (the printed statement has a sign typo), and Prop 4
formalizes the exact optimization of the paper's own quadratic/Gaussian-approximate objective,
with symmetric matrix square roots taken as (spectral-theorem-satisfiable) hypotheses.
-/
