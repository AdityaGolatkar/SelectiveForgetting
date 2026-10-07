import Basic

open Real Finset Matrix

namespace SelectiveForgetting

/-!
# Proposition 3 — finite-time algebraic core of optimal quadratic scrubbing

Formalizes the **finite-time algebraic core of Proposition 3** of
"Eternal Sunshine of the Spotless Net" (catalog id `prop3_quadratic_scrubbing`, Appendix C).
Under a local quadratic approximation of the loss
and gradient-flow training, two networks trained on the full data `D` and on the retain set
`Dr` from the *same* random initialization `w₀` are related by an explicit affine map `h`,
so that applying `h` to the `D`-trained weights recovers exactly the `Dr`-trained weights:
`h(A_t(D, ε)) = A_t(Dr, ε)`.

The two gradient flows have the closed forms
`w_A(t) = w*_A + e^{-At}(w₀ - w*_A)` and `w_B(t) = w*_B + e^{-Bt}(w₀ - w*_B)`,
where `A = ∇²L_D`, `B = ∇²L_Dr`. Following the paper, we take these closed forms as given
(the paper asserts them as the standard solution of the linear gradient-flow ODE) and prove
the substitution identity that is the actual content of the proof.

## Faithfulness note

The paper's *printed* statement of `h` (Eq. (6) / appendix) has the middle term
`e^{-Bt}(d - d_r)`, but the proof's own final line derives `e^{-Bt}(d_r - d)`. The printed
sign makes the claimed identity fail (checked by an explicit numerical counterexample); we
formalize the **corrected sign `e^{-Bt}(d_r - d)`** that the proof actually establishes, and
keep the leading `w`-term present in the main-text/Eq. (6) form (the appendix statement drops
it, but the proof keeps it). See the local PLAN's Stage-2 typo list.

The only property of the matrix exponential used is that `e^{At}` inverts `e^{-At}`. The
headline `quadratic_scrubbing_flow_identity` abstracts `e^{-At}, e^{-Bt}, e^{At}` as matrices
`EA, EB, FA` with `FA * EA = 1`; the corollary `quadratic_scrubbing_flow_identity_exp`
instantiates them with the genuine matrix exponential, discharging the hypothesis from
`exp (MA) * exp (-MA) = exp 0 = 1`.

## Exact boundary of the formalization

This file does **not** derive the closed forms from quadratic-loss gradients or an ODE
existence/uniqueness theorem; it takes the two formulas displayed by the paper as hypotheses.
It also does not formalize the induced equality of probability laws/conditional KL zero, or
the analytic `t → ∞` Newton-update limit in Eq. (7). Those claims remain separately cataloged,
so the declarations below must not be described as a proof of the full proposition.
-/

/-- **Proposition 3 (finite-time flow substitution identity).** Let `wA = w*_A + EA (w₀ - w*_A)` and
`wB = w*_B + EB (w₀ - w*_B)` be the two gradient-flow paths started from the shared
initialization `w₀`, where `EA, EB` model `e^{-At}, e^{-Bt}` and `FA` models `e^{At}`, the
inverse of `EA` (`FA * EA = 1`). Writing `d = wA - w*_A` and `d_r = wA - w*_B`, the retain
path is recovered from the full-data path by
`wB = wA + EB (FA d) + EB (d_r - d) - d_r`. This is exactly `h(wA) = wB` with the corrected
middle-term sign. -/
theorem quadratic_scrubbing_flow_identity {n : Type*} [Fintype n] [DecidableEq n]
    (EA EB FA : Matrix n n ℝ) (wStarA wStarB w0 wA wB : n → ℝ)
    (hFA : FA * EA = 1)
    (hwA : wA = wStarA + EA *ᵥ (w0 - wStarA))
    (hwB : wB = wStarB + EB *ᵥ (w0 - wStarB)) :
    wB = wA + EB *ᵥ (FA *ᵥ (wA - wStarA))
             + EB *ᵥ ((wA - wStarB) - (wA - wStarA)) - (wA - wStarB) := by
  -- The sole use of the exponential: `FA` cancels `EA`.
  have hinv : ∀ v : n → ℝ, FA *ᵥ (EA *ᵥ v) = v := fun v => by
    rw [mulVec_mulVec, hFA, one_mulVec]
  subst hwA hwB
  simp only [Matrix.mulVec_add, Matrix.mulVec_sub, hinv]
  abel

/-- **Finite-time identity with the genuine matrix exponential.** Instantiates the flow identity
with
`EA = e^{-MA}`, `EB = e^{-MB}`, `FA = e^{MA}` (where `MA = At`, `MB = Bt`), showing the
abstract hypothesis `FA * EA = 1` is realized by the matrix exponential via
`e^{MA} · e^{-MA} = e^{MA + (-MA)} = e^0 = 1`. -/
theorem quadratic_scrubbing_flow_identity_exp {n : Type*} [Fintype n] [DecidableEq n]
    (MA MB : Matrix n n ℝ) (wStarA wStarB w0 wA wB : n → ℝ)
    (hwA : wA = wStarA + NormedSpace.exp (-MA) *ᵥ (w0 - wStarA))
    (hwB : wB = wStarB + NormedSpace.exp (-MB) *ᵥ (w0 - wStarB)) :
    wB = wA + NormedSpace.exp (-MB) *ᵥ (NormedSpace.exp MA *ᵥ (wA - wStarA))
             + NormedSpace.exp (-MB) *ᵥ ((wA - wStarB) - (wA - wStarA)) - (wA - wStarB) := by
  refine quadratic_scrubbing_flow_identity (NormedSpace.exp (-MA)) (NormedSpace.exp (-MB))
    (NormedSpace.exp MA) wStarA wStarB w0 wA wB ?_ hwA hwB
  have h0 : MA + -MA = 0 := by abel
  rw [← Matrix.exp_add_of_commute MA (-MA) ((Commute.refl MA).neg_right), h0,
      NormedSpace.exp_zero]

end SelectiveForgetting
