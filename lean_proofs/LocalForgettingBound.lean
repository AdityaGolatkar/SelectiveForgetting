import Basic

open Real Finset

namespace SelectiveForgetting

/-!
# Proposition 2 — Local Forgetting Bound

Formalizes **Proposition 2** of "Eternal Sunshine of the Spotless Net" (catalog id
`prop2_local_forgetting_bound`). Writing the training seed as a discrete random variable
`e` with probability `p e`, the seed-averaged (mixture) distributions are
`x ↦ ∑ e, p e * Q e x`. The KL divergence between the two mixtures is bounded by the
average, over seeds, of the per-seed KL divergences — i.e. the joint convexity of KL.

The proof applies the log-sum inequality (`log_sum_inequality`, Basic) for each outcome
`x` over the seed index `e`, then swaps the order of summation. Only positivity of the
seed weights is used; `∑ e, p e = 1` records that `p` is the seed distribution so that the
mixtures are genuine distributions.
-/

/-- **Proposition 2 (Local Forgetting Bound).** `KL(∑ₑ pₑ Qₑ ‖ ∑ₑ pₑ Rₑ) ≤ ∑ₑ pₑ · KL(Qₑ ‖ Rₑ)`. -/
theorem klDiv_mixture_le {α σ : Type*} [Fintype α] [Fintype σ]
    (p : σ → ℝ) (Q R : σ → α → ℝ)
    (hp : ∀ e, 0 < p e) (_hp1 : ∑ e, p e = 1)
    (hQ : ∀ e x, 0 ≤ Q e x) (hR : ∀ e x, 0 < R e x) :
    klDiv (fun x => ∑ e, p e * Q e x) (fun x => ∑ e, p e * R e x)
      ≤ ∑ e, p e * klDiv (Q e) (R e) := by
  simp only [klDiv]
  calc ∑ x, (∑ e, p e * Q e x) * Real.log ((∑ e, p e * Q e x) / (∑ e, p e * R e x))
      ≤ ∑ x, ∑ e, (p e * Q e x) * Real.log ((p e * Q e x) / (p e * R e x)) := by
        apply Finset.sum_le_sum
        intro x _
        exact log_sum_inequality univ (fun e => p e * Q e x) (fun e => p e * R e x)
          (fun e _ => mul_nonneg (hp e).le (hQ e x)) (fun e _ => mul_pos (hp e) (hR e x))
    _ = ∑ x, ∑ e, (p e * Q e x) * Real.log (Q e x / R e x) := by
        refine Finset.sum_congr rfl (fun x _ => Finset.sum_congr rfl (fun e _ => ?_))
        rw [mul_div_mul_left _ _ (hp e).ne']
    _ = ∑ e, ∑ x, (p e * Q e x) * Real.log (Q e x / R e x) := Finset.sum_comm
    _ = ∑ e, p e * ∑ x, Q e x * Real.log (Q e x / R e x) := by
        refine Finset.sum_congr rfl (fun e _ => ?_)
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl (fun x _ => by ring)

end SelectiveForgetting
