# Lean 4 / Mathlib formalization of the Appendix C results

Machine-checked (Lean 4 + [Mathlib](https://github.com/leanprover-community/mathlib4)) proofs of the self-contained mathematical results of *Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks* (Golatkar, Achille, Soatto, CVPR 2020) — the four propositions and one lemma of Appendix C ("Proofs").

Everything builds with zero `sorry`, and `#print axioms` reports only the standard classical axioms (`propext`, `Classical.choice`, `Quot.sound`) for every result. The files are:

- `Basic.lean` — the log-sum inequality and discrete KL divergence (shared engine);
- `ReadoutContraction.lean` — Lemma 1 (a readout can only decrease KL);
- `LocalForgettingBound.lean` — Proposition 2 (the Local Forgetting Bound);
- `InformationBound.lean` — Proposition 1 / Eq. (2) (the information bound);
- `QuadraticScrubbing.lean` — Proposition 3 (optimal quadratic scrubbing);
- `RobustScrubbing.lean` — Proposition 4 (robust isotropic scrubbing);
- `SelectiveForgetting.lean` — the umbrella import.

## How to build

Requires a Lean 4 toolchain (`elan`/`lake`). Run `lake build` in this folder. The Mathlib version is pinned in `lakefile.toml` / `lake-manifest.json`.

## What is proved

The information-theoretic results (Lemma 1, Propositions 1–2) are modelled over finite types, faithful to the paper's own discrete proofs ("we will consider the random variables to be discrete"). A discrete distribution is a function $Q : \alpha \to \mathbb{R}$ with $Q \ge 0$; KL divergence is $\mathrm{KL}(Q \Vert R) = \sum_x Q(x)\, \log\!\big(Q(x)/R(x)\big)$. All three rest on a single lemma.

- **`Basic.lean`** — the **log-sum inequality**: for $a_i \ge 0$ and $b_i > 0$ over a finite index set,
  $$\Big(\textstyle\sum_i a_i\Big)\, \log\frac{\sum_i a_i}{\sum_i b_i} \le \sum_i a_i\, \log\frac{a_i}{b_i},$$
  proved as Jensen's inequality for the convex function $t \mapsto t\,\log t$. From it, `klDiv_nonneg` gives Gibbs' inequality $\mathrm{KL}(Q \Vert R) \ge 0$.

- **`ReadoutContraction.lean`** — **Lemma 1**. For any readout $f$, with the pushforward $f_* Q(c) = \sum_{x : f(x) = c} Q(x)$, `klDiv_pushforward_le` proves $\mathrm{KL}(f_* Q \Vert f_* R) \le \mathrm{KL}(Q \Vert R)$, by the log-sum inequality on each level set $\{x : f(x)=c\}$ summed over $c$.

- **`LocalForgettingBound.lean`** — **Proposition 2**. For a finite seed distribution $p$ and per-seed distributions $Q_e, R_e$, `klDiv_mixture_le` proves the joint convexity of KL,
  $$\mathrm{KL}\Big(\textstyle\sum_e p_e Q_e \,\Big\Vert\, \sum_e p_e R_e\Big) \le \sum_e p_e\, \mathrm{KL}(Q_e \Vert R_e),$$
  the discrete analogue of the paper's $\mathbb{E}_\epsilon$ mixture, again from the log-sum inequality.

- **`InformationBound.lean`** — **Proposition 1** / Eq. (2). With a prior $p_X$ and channel $p(z \mid x)$, `mutualInfo_le_expected_klDiv` proves the variational bound $I(X; Z) \le \sum_x p_X(x)\, \mathrm{KL}(p(\cdot \mid x) \Vert q)$ for **any** reference $q$ — the gap equals $\mathrm{KL}(\text{marginal} \Vert q) \ge 0$ (Gibbs). `information_bound` then bounds $I(Y; f(S(w)))$; the Data Processing Inequality $I(Y; f(S(w))) \le I(D_f; f(S(w)))$ enters as a hypothesis, exactly as the paper invokes it as a standard result.

- **`QuadraticScrubbing.lean`** — **Proposition 3**. Two gradient flows from the same initialization, $w_A(t) = w^\*_A + e^{-At}(w_0 - w^\*_A)$ and $w_B(t) = w^\*_B + e^{-Bt}(w_0 - w^\*_B)$, satisfy the scrubbing identity $h(w_A(t)) = w_B(t)$. `quadratic_scrubbing_flow_identity` proves this as a linear-algebra identity whose only use of the matrix exponential is that $e^{At}$ inverts $e^{-At}$; `quadratic_scrubbing_flow_identity_exp` instantiates it with the genuine matrix exponential. The formalized map uses the middle-term sign $e^{-Bt}(d_r - d)$ that the paper's *proof* derives (the printed statement has $e^{-Bt}(d - d_r)$; see the note below). The $t \to \infty$ Newton update (Eq. (7)) is an analytic limit taken informally by the paper and is not formalized.

- **`RobustScrubbing.lean`** — **Proposition 4**, isotropic case. Under the paper's own second-order (Gaussian/quadratic) approximation, the noise covariance minimizes $\tfrac12\,\mathrm{tr}(B\Sigma) + \tfrac{c}{2}\,\mathrm{tr}(\Sigma^{-1})$ with $c = \lambda\sigma_h^2$. `robust_scrubbing_isotropic_bound` proves the **global** lower bound $2\sqrt{c}\,\mathrm{tr}(B^{1/2}) \le \mathrm{tr}(B\Sigma) + c\,\mathrm{tr}(\Sigma^{-1})$ over all PD $\Sigma$ by completing the square in the trace inner product (stronger than the paper's first-order argument); `robust_scrubbing_isotropic_attains` shows $\Sigma^\* = \sqrt{c}\,B^{-1/2}$ attains it, so it is a global minimizer; and `robust_scrubbing_isotropic_optimality_condition` verifies the paper's condition $\Sigma^\* B \Sigma^\* = c\,I$ (i.e. $\Sigma B \Sigma = \lambda\Sigma_h$ with $\Sigma_h = \sigma_h^2 I$). The symmetric square roots $B^{1/2}, \Sigma^{1/2}$ are taken as hypotheses; every real PD matrix has such a root by the spectral theorem.

The `catalog.json` in this folder maps each result to the paper's statement, equations, and proof lines.
