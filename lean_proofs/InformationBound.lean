import Basic

open Real Finset

namespace SelectiveForgetting

/-!
# Proposition 1 — bound on the information a readout carries about the forgetting cohort

Formalizes **Proposition 1** of "Eternal Sunshine of the Spotless Net" (catalog id
`prop1_information_bound`, Eq. (2)). Modelling the forgetting cohort `D_f` as a discrete
random variable `X` with prior `pX`, and the readout of the scrubbed weights as `Z` with
channel `pZgX = p(z | x)`, the Shannon mutual information `I(X; Z)` is bounded by the
expected KL forgetting quantity against any reference distribution `q` (the paper's
`P(f(S_0(w)) | D_r)`).

The heart of the proof — the variational bound `I(X; Z) ≤ E_x[KL(p(·|x) ‖ q)]` — is proved
here in full from Gibbs' inequality (`klDiv_nonneg`, Basic): the difference equals exactly
`KL(marginal ‖ q) ≥ 0`. The paper's first step, the Data Processing Inequality
`I(Y; Z) ≤ I(D_f; Z)`, is invoked by the paper as a standard result; it enters here as the
hypothesis `hDPI`, matching the paper's own treatment.
-/

variable {X Z : Type*} [Fintype X] [Fintype Z] [Nonempty X]

/-- Marginal distribution of `Z` induced by prior `pX` and channel `pZgX`. -/
noncomputable def marginalZ (pX : X → ℝ) (pZgX : X → Z → ℝ) : Z → ℝ :=
  fun z => ∑ x, pX x * pZgX x z

/-- Discrete Shannon mutual information `I(X; Z) = ∑ₓ pX x · KL(p(·|x) ‖ p(·))`. -/
noncomputable def mutualInfo (pX : X → ℝ) (pZgX : X → Z → ℝ) : ℝ :=
  ∑ x, pX x * klDiv (pZgX x) (marginalZ pX pZgX)

/-- **Variational bound on mutual information** (the substance of Proposition 1). For any
reference distribution `q`, `I(X; Z) ≤ ∑ₓ pX x · KL(p(·|x) ‖ q)`, since the gap equals
`KL(marginal ‖ q) ≥ 0`. -/
theorem mutualInfo_le_expected_klDiv
    (pX : X → ℝ) (pZgX : X → Z → ℝ) (q : Z → ℝ)
    (hpX : ∀ x, 0 < pX x) (hpX1 : ∑ x, pX x = 1)
    (hpZgX : ∀ x z, 0 < pZgX x z) (hpZgX1 : ∀ x, ∑ z, pZgX x z = 1)
    (hq : ∀ z, 0 < q z) (hq1 : ∑ z, q z = 1) :
    mutualInfo pX pZgX ≤ ∑ x, pX x * klDiv (pZgX x) q := by
  have hmarg : ∀ z, marginalZ pX pZgX z = ∑ x, pX x * pZgX x z := fun _ => rfl
  have hpZpos : ∀ z, 0 < marginalZ pX pZgX z := by
    intro z; rw [hmarg]
    exact Finset.sum_pos (fun x _ => mul_pos (hpX x) (hpZgX x z)) Finset.univ_nonempty
  have hpZ1 : ∑ z, marginalZ pX pZgX z = 1 := by
    have h : ∑ z, marginalZ pX pZgX z = ∑ x, pX x := by
      simp_rw [hmarg]
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl (fun x _ => by rw [← Finset.mul_sum, hpZgX1 x, mul_one])
    rw [h, hpX1]
  -- per-x: KL(p(·|x) ‖ q) = KL(p(·|x) ‖ marginal) + ∑_z p(z|x) log(marginal z / q z)
  have hdiff : ∀ x, klDiv (pZgX x) q
      = klDiv (pZgX x) (marginalZ pX pZgX)
        + ∑ z, pZgX x z * Real.log (marginalZ pX pZgX z / q z) := by
    intro x
    unfold klDiv
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl (fun z _ => ?_)
    rw [← mul_add]
    congr 1
    rw [Real.log_div (hpZgX x z).ne' (hq z).ne',
        Real.log_div (hpZgX x z).ne' (hpZpos z).ne',
        Real.log_div (hpZpos z).ne' (hq z).ne']
    ring
  -- assemble: ∑_x pX x · KL(p(·|x) ‖ q) = I(X;Z) + KL(marginal ‖ q)
  have hsum : ∑ x, pX x * klDiv (pZgX x) q
      = mutualInfo pX pZgX + klDiv (marginalZ pX pZgX) q := by
    have e1 : ∑ x, pX x * klDiv (pZgX x) q
        = ∑ x, pX x * klDiv (pZgX x) (marginalZ pX pZgX)
          + ∑ x, pX x * ∑ z, pZgX x z * Real.log (marginalZ pX pZgX z / q z) := by
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl (fun x _ => by rw [hdiff x, mul_add])
    rw [e1]
    unfold mutualInfo
    congr 1
    unfold klDiv
    calc ∑ x, pX x * ∑ z, pZgX x z * Real.log (marginalZ pX pZgX z / q z)
        = ∑ x, ∑ z, pX x * (pZgX x z * Real.log (marginalZ pX pZgX z / q z)) := by
          exact Finset.sum_congr rfl (fun x _ => by rw [Finset.mul_sum])
      _ = ∑ z, ∑ x, pX x * (pZgX x z * Real.log (marginalZ pX pZgX z / q z)) := Finset.sum_comm
      _ = ∑ z, (∑ x, pX x * pZgX x z) * Real.log (marginalZ pX pZgX z / q z) := by
          refine Finset.sum_congr rfl (fun z _ => ?_)
          rw [Finset.sum_mul]
          exact Finset.sum_congr rfl (fun x _ => by ring)
      _ = ∑ z, marginalZ pX pZgX z * Real.log (marginalZ pX pZgX z / q z) := by
          exact Finset.sum_congr rfl (fun z _ => by rw [hmarg z])
  have hnn : 0 ≤ klDiv (marginalZ pX pZgX) q :=
    klDiv_nonneg (marginalZ pX pZgX) q (fun z => (hpZpos z).le) hq hpZ1 hq1
  linarith [hsum, hnn]

/-- **Proposition 1.** The information a readout `iYZ = I(Y; f(S(w)))` carries about any
attribute `Y` of the forgetting cohort is bounded by the expected KL forgetting quantity.
`hDPI` is the Data Processing Inequality `I(Y; Z) ≤ I(D_f; Z)`, which the paper invokes as a
standard result; the remaining, substantive inequality is `mutualInfo_le_expected_klDiv`. -/
theorem information_bound
    (pX : X → ℝ) (pZgX : X → Z → ℝ) (q : Z → ℝ) (iYZ : ℝ)
    (hpX : ∀ x, 0 < pX x) (hpX1 : ∑ x, pX x = 1)
    (hpZgX : ∀ x z, 0 < pZgX x z) (hpZgX1 : ∀ x, ∑ z, pZgX x z = 1)
    (hq : ∀ z, 0 < q z) (hq1 : ∑ z, q z = 1)
    (hDPI : iYZ ≤ mutualInfo pX pZgX) :
    iYZ ≤ ∑ x, pX x * klDiv (pZgX x) q :=
  hDPI.trans (mutualInfo_le_expected_klDiv pX pZgX q hpX hpX1 hpZgX hpZgX1 hq hq1)

end SelectiveForgetting
