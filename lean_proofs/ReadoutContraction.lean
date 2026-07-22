import Basic

open Real Finset

namespace SelectiveForgetting

/-!
# Lemma 1 — a readout function can only decrease the KL divergence

Formalizes **Lemma 1** of "Eternal Sunshine of the Spotless Net" (catalog id
`lemma1_readout_dpi`). For any readout `f`, the KL divergence between the pushed-forward
distributions is bounded by the KL divergence on the full weights, so controlling the
latter guarantees robustness to every readout an attacker might apply.

The pushforward of a discrete distribution `Q` along `f` is `c ↦ ∑_{x : f x = c} Q x`.
The proof applies the log-sum inequality (`log_sum_inequality`, Basic) on each fiber
`f⁻¹(c)` and sums over `c` (recombining the fibers with `Finset.sum_fiberwise`).

The theorem uses the paper's elementary finite-sum formula under the explicit full-support
assumption `R x > 0`. A support-aware extended-real KL would remove this restriction; the
present statement exposes it rather than silently identifying it with the unrestricted lemma.
-/

/-- **Lemma 1.** For any readout `f`, `KL(f_* Q ‖ f_* R) ≤ KL(Q ‖ R)`, where `f_*` denotes
pushforward. -/
theorem klDiv_pushforward_le {α β : Type*} [Fintype α] [Fintype β] [DecidableEq β]
    (Q R : α → ℝ) (f : α → β)
    (hQ : ∀ x, 0 ≤ Q x) (hR : ∀ x, 0 < R x) :
    klDiv (fun c => ∑ x ∈ univ.filter (fun x => f x = c), Q x)
          (fun c => ∑ x ∈ univ.filter (fun x => f x = c), R x)
      ≤ klDiv Q R := by
  have hle : klDiv (fun c => ∑ x ∈ univ.filter (fun x => f x = c), Q x)
              (fun c => ∑ x ∈ univ.filter (fun x => f x = c), R x)
      ≤ ∑ c : β, ∑ x ∈ univ.filter (fun x => f x = c), Q x * Real.log (Q x / R x) := by
    unfold klDiv
    refine Finset.sum_le_sum (fun c _ => ?_)
    exact log_sum_inequality (univ.filter (fun x => f x = c)) Q R
      (fun i _ => hQ i) (fun i _ => hR i)
  exact hle.trans_eq (Finset.sum_fiberwise univ f fun x => Q x * Real.log (Q x / R x))

end SelectiveForgetting
