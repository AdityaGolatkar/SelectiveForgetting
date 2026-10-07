import Mathlib

open Real Finset

namespace SelectiveForgetting

/-- **Log-sum inequality.** For nonnegative `a i` and positive `b i` over a finite index set,
`(∑ a) * log ((∑ a) / (∑ b)) ≤ ∑ a i * log (a i / b i)`.

This single analytic fact is the core behind Lemma 1, Proposition 1 and Proposition 2 of
"Eternal Sunshine of the Spotless Net": each of those information-theoretic inequalities is
an application of it. It is Jensen's inequality for the convex function `x ↦ x * log x`
applied to the ratios `a i / b i` with weights `b i / (∑ b)`. -/
theorem log_sum_inequality {ι : Type*} (t : Finset ι) (a b : ι → ℝ)
    (ha : ∀ i ∈ t, 0 ≤ a i) (hb : ∀ i ∈ t, 0 < b i) :
    (∑ i ∈ t, a i) * Real.log ((∑ i ∈ t, a i) / (∑ i ∈ t, b i))
      ≤ ∑ i ∈ t, a i * Real.log (a i / b i) := by
  rcases t.eq_empty_or_nonempty with rfl | hne
  · simp
  · set A := ∑ i ∈ t, a i with hA
    set B := ∑ i ∈ t, b i with hB
    have hBpos : 0 < B := by rw [hB]; exact Finset.sum_pos hb hne
    have hBne : B ≠ 0 := ne_of_gt hBpos
    -- Jensen for `x ↦ x * log x`, weights `b i / B`, points `a i / b i`.
    have hjensen := convexOn_mul_log.map_sum_le
      (t := t) (w := fun i => b i / B) (p := fun i => a i / b i)
      (fun i hi => div_nonneg (hb i hi).le hBpos.le)
      (by rw [← Finset.sum_div, ← hB]; exact div_self hBne)
      (fun i hi => Set.mem_Ici.mpr (div_nonneg (ha i hi) (hb i hi).le))
    simp only [smul_eq_mul] at hjensen
    -- Simplify the two sums appearing in Jensen.
    have hterm1 : ∀ i ∈ t, (b i / B) * (a i / b i) = a i / B := by
      intro i hi
      have hbi : b i ≠ 0 := (hb i hi).ne'
      field_simp
    have hsum1 : ∑ i ∈ t, (b i / B) * (a i / b i) = A / B := by
      rw [hA, Finset.sum_div]; exact Finset.sum_congr rfl hterm1
    have hterm2 : ∀ i ∈ t,
        (b i / B) * (a i / b i * Real.log (a i / b i))
          = a i * Real.log (a i / b i) / B := by
      intro i hi
      rw [← mul_assoc, hterm1 i hi, div_mul_eq_mul_div]
    have hsum2 : ∑ i ∈ t, (b i / B) * (a i / b i * Real.log (a i / b i))
        = (∑ i ∈ t, a i * Real.log (a i / b i)) / B := by
      rw [Finset.sum_div]; exact Finset.sum_congr rfl hterm2
    rw [hsum1, hsum2] at hjensen
    -- hjensen : A / B * log (A / B) ≤ (∑ ...) / B ; multiply by B > 0.
    calc A * Real.log (A / B)
        = A / B * Real.log (A / B) * B := by
          rw [mul_right_comm, div_mul_cancel₀ _ hBne]
      _ ≤ (∑ i ∈ t, a i * Real.log (a i / b i)) / B * B :=
          mul_le_mul_of_nonneg_right hjensen hBpos.le
      _ = ∑ i ∈ t, a i * Real.log (a i / b i) := div_mul_cancel₀ _ hBne

/-- Discrete Kullback–Leibler divergence of `Q` from `R` on a finite type,
`KL(Q ‖ R) = ∑ x, Q x * log (Q x / R x)`, exactly as the paper defines it for the
discrete random variables its proofs use. -/
noncomputable def klDiv {α : Type*} [Fintype α] (Q R : α → ℝ) : ℝ :=
  ∑ x, Q x * Real.log (Q x / R x)

/-- **Gibbs' inequality** (nonnegativity of KL divergence) for discrete probability
distributions. Immediate from the log-sum inequality with `a = Q`, `b = R`, since
`(∑ Q) * log ((∑ Q)/(∑ R)) = 1 * log (1/1) = 0`. Used in the proof of Proposition 1. -/
theorem klDiv_nonneg {α : Type*} [Fintype α] (Q R : α → ℝ)
    (hQ : ∀ x, 0 ≤ Q x) (hR : ∀ x, 0 < R x)
    (hQ1 : ∑ x, Q x = 1) (hR1 : ∑ x, R x = 1) :
    0 ≤ klDiv Q R := by
  have h := log_sum_inequality Finset.univ Q R (fun i _ => hQ i) (fun i _ => hR i)
  rw [hQ1, hR1, div_self (one_ne_zero), Real.log_one, mul_zero] at h
  exact h

end SelectiveForgetting
