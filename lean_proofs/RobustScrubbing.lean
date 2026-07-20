import Basic

open Real Finset Matrix

namespace SelectiveForgetting

/-!
# Proposition 4 — Robust (isotropic) scrubbing: optimal noise covariance

Formalizes **Proposition 4** of "Eternal Sunshine of the Spotless Net" (catalog id
`prop4_robust_scrubbing`, Appendix C) in the isotropic case. After the paper's second-order
(Gaussian/quadratic) approximation of the Forgetting Lagrangian — which is the paper's own
stated hypothesis — the objective in the noise covariance `Σ` (a positive-definite matrix,
written `Cov` in the Lean code since `Σ` is reserved syntax) reduces, up to the additive
constant `L_Dr(h(w))`, to `½ tr(B Σ) + (λσ_h²/2) tr(Σ⁻¹)`, with `B = ∇²L_Dr`. Writing
`c = λσ_h²`, minimizing this is minimizing `g(Σ) = tr(B Σ) + c tr(Σ⁻¹)` over positive-definite
`Σ`.

We prove the **global** optimum by completing the square in the trace inner product — strictly
stronger than the paper's first-order (stationarity) derivation:

* `robust_scrubbing_isotropic_bound` : `2√c · tr(B^{1/2}) ≤ tr(B Σ) + c tr(Σ⁻¹)` for every PD
  `Σ` (the global lower bound; halving gives the objective's minimum value
  `√c · tr(B^{1/2}) = √(λσ_h²) · tr(B^{1/2})`).
* `robust_scrubbing_isotropic_attains` : the covariance `Σ* = √c · B^{-1/2}` attains it.
* `robust_scrubbing_isotropic_optimality_condition` : `Σ* B Σ* = c · I`, i.e. the paper's
  optimality condition `Σ B Σ = λ Σ_h` in the isotropic case `Σ_h = σ_h² I` (so
  `λ Σ_h = λσ_h² I = c I`), and hence `Σ* = √(λσ_h²) B^{-1/2}`, exactly Eq. of Prop 4.

## Disclosed hypotheses (honest scope)

1. **The quadratic/Gaussian approximation is the paper's hypothesis, not a theorem.** We
   formalize the *exact* optimization of the resulting approximate objective; the `≃` step in
   the paper (dropping `o(n²)`) is assumed, exactly as the paper assumes it.
2. **Symmetric square roots are taken as hypotheses.** Every real PD matrix has a unique PD
   (hence symmetric) square root by the spectral theorem; rather than construct it through the
   continuous functional calculus we hypothesize `Bsqrt, Ssqrt` symmetric with
   `Bsqrt*Bsqrt = B`, `Ssqrt*Ssqrt = Cov` (and invertibility). The hypotheses are satisfiable
   for exactly the matrices in the paper's scope (PD `B`, PD `Σ`).
-/

/-- **Completing the square in the Frobenius (trace) inner product.** For real square matrices
`P, Q`, `2 · tr(Pᵀ Q) ≤ tr(Pᵀ P) + tr(Qᵀ Q)`, from `0 ≤ tr((P - Q)ᵀ (P - Q))`. This is the
single analytic fact behind the global optimality in Proposition 4. -/
theorem two_mul_trace_le_of_transpose {n : Type*} [Fintype n] (P Q : Matrix n n ℝ) :
    2 * (Pᵀ * Q).trace ≤ (Pᵀ * P).trace + (Qᵀ * Q).trace := by
  have hpsd : 0 ≤ ((P - Q)ᵀ * (P - Q)).trace := by
    have h := (Matrix.posSemidef_conjTranspose_mul_self (P - Q)).trace_nonneg
    rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at h
  have hcross : (Qᵀ * P).trace = (Pᵀ * Q).trace := by
    rw [← Matrix.trace_transpose (Qᵀ * P), Matrix.transpose_mul, Matrix.transpose_transpose]
  have hexp : ((P - Q)ᵀ * (P - Q)).trace
      = (Pᵀ * P).trace - 2 * (Pᵀ * Q).trace + (Qᵀ * Q).trace := by
    rw [Matrix.transpose_sub, sub_mul, mul_sub, mul_sub, Matrix.trace_sub, Matrix.trace_sub,
        Matrix.trace_sub, hcross]
    ring
  rw [hexp] at hpsd
  linarith

/-- **Proposition 4 (isotropic global lower bound).** For PD `B = Bsqrt²`, `Cov = Ssqrt²` with
`Bsqrt, Ssqrt` symmetric square roots (`Ssqrt` invertible) and `c ≥ 0`,
`2√c · tr(Bsqrt) ≤ tr(B Cov) + c · tr(Cov⁻¹)`. Halving, the minimum of the (isotropic,
approximate) forgetting objective `½tr(BΣ) + (c/2)tr(Σ⁻¹)` over PD `Σ` is at least
`√c · tr(B^{1/2})`. (`Cov` denotes the noise covariance `Σ`.) -/
theorem robust_scrubbing_isotropic_bound {n : Type*} [Fintype n] [DecidableEq n]
    (B Cov Bsqrt Ssqrt : Matrix n n ℝ) (c : ℝ) (hc : 0 ≤ c)
    (hBs_sym : Bsqrt.IsSymm) (hBs : Bsqrt * Bsqrt = B)
    (hSs_sym : Ssqrt.IsSymm) (hSs : Ssqrt * Ssqrt = Cov)
    (hSs_unit : IsUnit Ssqrt.det) :
    2 * Real.sqrt c * Bsqrt.trace ≤ (B * Cov).trace + c * Cov⁻¹.trace := by
  set s := Real.sqrt c with hs
  have hss : s * s = c := Real.mul_self_sqrt hc
  -- P = Bsqrt·Ssqrt,  Q = s • Ssqrt⁻¹
  have hPtP : ((Bsqrt * Ssqrt)ᵀ * (Bsqrt * Ssqrt)).trace = (B * Cov).trace := by
    rw [Matrix.transpose_mul, hBs_sym, hSs_sym]
    calc (Ssqrt * Bsqrt * (Bsqrt * Ssqrt)).trace
        = (Ssqrt * (Bsqrt * Bsqrt * Ssqrt)).trace := by congr 1; noncomm_ring
      _ = (Bsqrt * Bsqrt * Ssqrt * Ssqrt).trace := Matrix.trace_mul_comm _ _
      _ = (B * Cov).trace := by rw [mul_assoc, hBs, hSs]
  have hQtQ : ((s • Ssqrt⁻¹)ᵀ * (s • Ssqrt⁻¹)).trace = c * Cov⁻¹.trace := by
    rw [Matrix.transpose_smul, Matrix.transpose_nonsing_inv, hSs_sym, smul_mul_assoc,
        mul_smul_comm, smul_smul, hss, Matrix.trace_smul, smul_eq_mul, ← Matrix.mul_inv_rev, hSs]
  have hPtQ : ((Bsqrt * Ssqrt)ᵀ * (s • Ssqrt⁻¹)).trace = s * Bsqrt.trace := by
    rw [Matrix.transpose_mul, hBs_sym, hSs_sym, mul_smul_comm, Matrix.trace_smul, smul_eq_mul]
    congr 1
    calc (Ssqrt * Bsqrt * Ssqrt⁻¹).trace
        = (Ssqrt * (Bsqrt * Ssqrt⁻¹)).trace := by rw [mul_assoc]
      _ = (Bsqrt * Ssqrt⁻¹ * Ssqrt).trace := Matrix.trace_mul_comm _ _
      _ = (Bsqrt * (Ssqrt⁻¹ * Ssqrt)).trace := by rw [mul_assoc]
      _ = (Bsqrt * 1).trace := by rw [Matrix.nonsing_inv_mul Ssqrt hSs_unit]
      _ = Bsqrt.trace := by rw [mul_one]
  have hmain := two_mul_trace_le_of_transpose (Bsqrt * Ssqrt) (s • Ssqrt⁻¹)
  rw [hPtP, hQtQ, hPtQ, ← mul_assoc] at hmain
  linarith

/-- **Proposition 4 (optimality condition).** The optimal isotropic covariance
`Σ* = √c · B^{-1/2}` satisfies `Σ* B Σ* = c · I`, i.e. the paper's condition `Σ B Σ = λ Σ_h`
with `Σ_h = σ_h² I` and `c = λσ_h²`. -/
theorem robust_scrubbing_isotropic_optimality_condition {n : Type*} [Fintype n] [DecidableEq n]
    (B Bsqrt : Matrix n n ℝ) (c : ℝ) (hc : 0 ≤ c)
    (hBs : Bsqrt * Bsqrt = B) (hBs_unit : IsUnit Bsqrt.det) :
    (Real.sqrt c • Bsqrt⁻¹) * B * (Real.sqrt c • Bsqrt⁻¹) = c • (1 : Matrix n n ℝ) := by
  have hss : Real.sqrt c * Real.sqrt c = c := Real.mul_self_sqrt hc
  rw [← hBs, smul_mul_assoc, smul_mul_assoc, mul_smul_comm, smul_smul, hss]
  congr 1
  rw [← mul_assoc, Matrix.nonsing_inv_mul Bsqrt hBs_unit, Matrix.one_mul,
      Matrix.mul_nonsing_inv Bsqrt hBs_unit]

/-- **Proposition 4 (attainment).** With `c > 0`, the covariance `Σ* = √c · B^{-1/2}` attains
the global lower bound of `robust_scrubbing_isotropic_bound`:
`tr(B Σ*) + c · tr(Σ*⁻¹) = 2√c · tr(B^{1/2})`. Together with the bound, `Σ*` is a global
minimizer of the isotropic forgetting objective. -/
theorem robust_scrubbing_isotropic_attains {n : Type*} [Fintype n] [DecidableEq n]
    (B Bsqrt : Matrix n n ℝ) (c : ℝ) (hc : 0 < c)
    (hBs : Bsqrt * Bsqrt = B) (hBs_unit : IsUnit Bsqrt.det) :
    (B * (Real.sqrt c • Bsqrt⁻¹)).trace + c * (Real.sqrt c • Bsqrt⁻¹)⁻¹.trace
      = 2 * Real.sqrt c * Bsqrt.trace := by
  set s := Real.sqrt c with hs
  have hss : s * s = c := Real.mul_self_sqrt hc.le
  have hs_ne : s ≠ 0 := (Real.sqrt_pos.mpr hc).ne'
  -- tr(B Σ*) = s · tr(Bsqrt)
  have hBCov : (B * (s • Bsqrt⁻¹)).trace = s * Bsqrt.trace := by
    rw [← hBs, mul_smul_comm, Matrix.trace_smul, smul_eq_mul, mul_assoc,
        Matrix.mul_nonsing_inv Bsqrt hBs_unit, mul_one]
  -- (Σ*)⁻¹ = s⁻¹ • Bsqrt
  have hCovInv : (s • Bsqrt⁻¹)⁻¹ = s⁻¹ • Bsqrt := by
    apply Matrix.inv_eq_right_inv
    rw [smul_mul_assoc, mul_smul_comm, smul_smul, mul_inv_cancel₀ hs_ne,
        Matrix.nonsing_inv_mul Bsqrt hBs_unit, one_smul]
  have hcs : c * s⁻¹ = s := by rw [← hss, mul_assoc, mul_inv_cancel₀ hs_ne, mul_one]
  rw [hBCov, hCovInv, Matrix.trace_smul, smul_eq_mul, ← mul_assoc, hcs]
  ring

end SelectiveForgetting
