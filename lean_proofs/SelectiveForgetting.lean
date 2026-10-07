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

The package proves the complete finite information-theory kernel and two explicitly delimited
linear-algebra subresults. It does not claim that every statement below is fully formalized:

* Log-sum inequality (proof engine): `log_sum_inequality` — `Basic`
* KL ≥ 0 (Gibbs): `klDiv_nonneg` — `Basic`
* **Lemma 1** (readout DPI): `klDiv_pushforward_le` — `ReadoutContraction`
* **Proposition 1** (information bound): `mutualInfo_le_expected_klDiv`,
  `markovMutualInfo_le`, `information_bound` — `InformationBound`
* **Proposition 2** (local forgetting bound): `klDiv_mixture_le` — `LocalForgettingBound`
* **Proposition 3, finite-time algebraic core**: `quadratic_scrubbing_flow_identity`,
  `quadratic_scrubbing_flow_identity_exp` — `QuadraticScrubbing`
* **Proposition 4, isotropic optimization only**: `robust_scrubbing_isotropic_bound`,
  `robust_scrubbing_isotropic_optimality_condition`, `robust_scrubbing_isotropic_attains`,
  `robust_scrubbing_isotropic_candidate_posDef`, `robust_scrubbing_isotropic_global_minimum`
  — `RobustScrubbing`

The information-theoretic core (Lemma 1, Props 1–2) is built self-contained over finite types
from a single `log_sum_inequality` (Jensen for `x ↦ x log x`), faithful to the paper's own
discrete proofs, under explicit full-support restrictions. Proposition 3 formalizes the
**corrected** finite-time substitution identity but not the ODE derivation, induced KL-zero
statement, or Eq. (7) limit. Proposition 4 formalizes the exact isotropic optimization of the
paper's quadratic/Gaussian-approximate objective, proves the candidate covariance is positive
definite, and proves global minimality over covariances represented by positive-definite square
roots; the general non-isotropic condition remains unformalized. The catalog also records
Corollary 1 and Example 1 as unformalized rather than omitting them.
-/
