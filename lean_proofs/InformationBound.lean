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

Both steps are proved here. The Data Processing Inequality `I(Y; Z) ≤ I(D_f; Z)` is derived
for the finite Markov model `Y ← D_f → Z` by applying the log-sum inequality while summing out
`D_f`. The variational bound `I(D_f; Z) ≤ E_x[KL(p(·|x) ‖ q)]` follows from Gibbs'
inequality (`klDiv_nonneg`, Basic): the gap equals exactly `KL(marginal ‖ q) ≥ 0`.

The discrete KL layer used in this file assumes full support: the prior, both conditional
channels, and the reference distribution are strictly positive. This is stronger than the
paper's unstated support conditions and is exposed explicitly in every theorem signature.
-/

variable {X Y Z : Type*} [Fintype X] [Fintype Y] [Fintype Z] [Nonempty X]

/-- Marginal distribution of `Z` induced by prior `pX` and channel `pZgX`. -/
noncomputable def marginalZ (pX : X → ℝ) (pZgX : X → Z → ℝ) : Z → ℝ :=
  fun z => ∑ x, pX x * pZgX x z

/-- Discrete Shannon mutual information `I(X; Z) = ∑ₓ pX x · KL(p(·|x) ‖ p(·))`. -/
noncomputable def mutualInfo (pX : X → ℝ) (pZgX : X → Z → ℝ) : ℝ :=
  ∑ x, pX x * klDiv (pZgX x) (marginalZ pX pZgX)

/-- Marginal distribution of an attribute `Y` induced by the cohort prior and the
channel `p(y | x)`. -/
noncomputable def marginalY (pX : X → ℝ) (pYgX : X → Y → ℝ) : Y → ℝ :=
  fun y => ∑ x, pX x * pYgX x y

/-- Joint distribution of `(Y,Z)` in the finite Markov model `Y ← X → Z`. Conditional
independence given `X` is encoded by the product `p(y | x) p(z | x)`. -/
noncomputable def markovJointYZ
    (pX : X → ℝ) (pYgX : X → Y → ℝ) (pZgX : X → Z → ℝ) : Y → Z → ℝ :=
  fun y z => ∑ x, pX x * pYgX x y * pZgX x z

/-- Mutual information `I(Y;Z)` for the finite Markov model `Y ← X → Z`, written as
`KL(p_{Y,Z} ‖ p_Y p_Z)`. -/
noncomputable def markovMutualInfo
    (pX : X → ℝ) (pYgX : X → Y → ℝ) (pZgX : X → Z → ℝ) : ℝ :=
  ∑ y, ∑ z, markovJointYZ pX pYgX pZgX y z *
    Real.log (markovJointYZ pX pYgX pZgX y z /
      (marginalY pX pYgX y * marginalZ pX pZgX z))

/-- **Data Processing Inequality for the paper's Markov chain.** In the finite model
`Y ← X → Z`, conditional independence of `Y` and `Z` given `X` implies
`I(Y;Z) ≤ I(X;Z)`. The proof is the log-sum inequality applied for every `(y,z)` while
summing out `X`; no information-theory result is assumed as a black box. -/
theorem markovMutualInfo_le
    (pX : X → ℝ) (pYgX : X → Y → ℝ) (pZgX : X → Z → ℝ)
    (hpX : ∀ x, 0 < pX x) (_hpX1 : ∑ x, pX x = 1)
    (hpYgX : ∀ x y, 0 < pYgX x y) (hpYgX1 : ∀ x, ∑ y, pYgX x y = 1)
    (hpZgX : ∀ x z, 0 < pZgX x z) (_hpZgX1 : ∀ x, ∑ z, pZgX x z = 1) :
    markovMutualInfo pX pYgX pZgX ≤ mutualInfo pX pZgX := by
  have hpZpos : ∀ z, 0 < marginalZ pX pZgX z := by
    intro z
    unfold marginalZ
    exact Finset.sum_pos (fun x _ => mul_pos (hpX x) (hpZgX x z)) Finset.univ_nonempty
  have hpoint : ∀ y z,
      markovJointYZ pX pYgX pZgX y z *
          Real.log (markovJointYZ pX pYgX pZgX y z /
            (marginalY pX pYgX y * marginalZ pX pZgX z))
        ≤ ∑ x, (pX x * pYgX x y * pZgX x z) *
            Real.log (pZgX x z / marginalZ pX pZgX z) := by
    intro y z
    have h := log_sum_inequality Finset.univ
      (fun x => pX x * pYgX x y * pZgX x z)
      (fun x => pX x * pYgX x y * marginalZ pX pZgX z)
      (fun x _ => (mul_pos (mul_pos (hpX x) (hpYgX x y)) (hpZgX x z)).le)
      (fun x _ => mul_pos (mul_pos (hpX x) (hpYgX x y)) (hpZpos z))
    have hden : (∑ x, pX x * pYgX x y * marginalZ pX pZgX z)
        = marginalY pX pYgX y * marginalZ pX pZgX z := by
      rw [← Finset.sum_mul]
      rfl
    rw [show (∑ x, pX x * pYgX x y * pZgX x z) =
        markovJointYZ pX pYgX pZgX y z by rfl, hden] at h
    convert h using 1
    refine Finset.sum_congr rfl (fun x _ => ?_)
    congr 1
    rw [show pX x * pYgX x y * pZgX x z =
          (pX x * pYgX x y) * pZgX x z by ring,
        show pX x * pYgX x y * marginalZ pX pZgX z =
          (pX x * pYgX x y) * marginalZ pX pZgX z by ring,
        mul_div_mul_left _ _ (mul_pos (hpX x) (hpYgX x y)).ne']
  calc
    markovMutualInfo pX pYgX pZgX
        ≤ ∑ y, ∑ z, ∑ x, (pX x * pYgX x y * pZgX x z) *
            Real.log (pZgX x z / marginalZ pX pZgX z) := by
          unfold markovMutualInfo
          exact Finset.sum_le_sum (fun y _ => Finset.sum_le_sum (fun z _ => hpoint y z))
    _ = ∑ y, ∑ x, ∑ z, (pX x * pYgX x y * pZgX x z) *
            Real.log (pZgX x z / marginalZ pX pZgX z) := by
          apply Finset.sum_congr rfl
          intro y _
          exact Finset.sum_comm
    _ = ∑ x, ∑ y, ∑ z, (pX x * pYgX x y * pZgX x z) *
            Real.log (pZgX x z / marginalZ pX pZgX z) := Finset.sum_comm
    _ = ∑ x, ∑ z, ∑ y, (pX x * pYgX x y * pZgX x z) *
            Real.log (pZgX x z / marginalZ pX pZgX z) := by
          apply Finset.sum_congr rfl
          intro x _
          exact Finset.sum_comm
    _ = ∑ x, ∑ z, pX x * pZgX x z *
            Real.log (pZgX x z / marginalZ pX pZgX z) := by
          apply Finset.sum_congr rfl
          intro x _
          apply Finset.sum_congr rfl
          intro z _
          calc
            ∑ y, (pX x * pYgX x y * pZgX x z) *
                Real.log (pZgX x z / marginalZ pX pZgX z)
                = ∑ y, pYgX x y * (pX x * pZgX x z *
                    Real.log (pZgX x z / marginalZ pX pZgX z)) := by
                    exact Finset.sum_congr rfl (fun y _ => by ring)
            _ = (∑ y, pYgX x y) * (pX x * pZgX x z *
                    Real.log (pZgX x z / marginalZ pX pZgX z)) := by
                    rw [Finset.sum_mul]
            _ = pX x * pZgX x z *
                    Real.log (pZgX x z / marginalZ pX pZgX z) := by
                    rw [hpYgX1 x, one_mul]
    _ = mutualInfo pX pZgX := by
          unfold mutualInfo klDiv
          apply Finset.sum_congr rfl
          intro x _
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl (fun z _ => by ring)

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

/-- **Proposition 1.** In the finite Markov model `Y ← X → Z`, where `X` is the forgetting
cohort and `Z = f(S(w))` is the readout, the information `I(Y;Z)` is bounded by the
expected KL forgetting quantity against the certificate distribution `q`.

Unlike the paper, which invokes Data Processing as a standard result, this theorem derives
that step from `markovMutualInfo_le` and then applies `mutualInfo_le_expected_klDiv`. -/
theorem information_bound
    (pX : X → ℝ) (pYgX : X → Y → ℝ) (pZgX : X → Z → ℝ) (q : Z → ℝ)
    (hpX : ∀ x, 0 < pX x) (hpX1 : ∑ x, pX x = 1)
    (hpYgX : ∀ x y, 0 < pYgX x y) (hpYgX1 : ∀ x, ∑ y, pYgX x y = 1)
    (hpZgX : ∀ x z, 0 < pZgX x z) (hpZgX1 : ∀ x, ∑ z, pZgX x z = 1)
    (hq : ∀ z, 0 < q z) (hq1 : ∑ z, q z = 1) :
    markovMutualInfo pX pYgX pZgX ≤ ∑ x, pX x * klDiv (pZgX x) q :=
  (markovMutualInfo_le pX pYgX pZgX hpX hpX1 hpYgX hpYgX1 hpZgX hpZgX1).trans
    (mutualInfo_le_expected_klDiv pX pZgX q hpX hpX1 hpZgX hpZgX1 hq hq1)

end SelectiveForgetting
