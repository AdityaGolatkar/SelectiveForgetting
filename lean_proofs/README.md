# Lean 4 / Mathlib formalization of the paper's finite mathematical kernel

Machine-checked (Lean 4 + [Mathlib](https://github.com/leanprover-community/mathlib4)) proofs associated with *Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks* (Golatkar, Achille, Soatto, CVPR 2020). Coverage is complete for Lemma 1 and Propositions 1–2, and deliberately partial for Propositions 3–4: the finite-time algebraic core of Proposition 3 and the isotropic optimization subclaim of Proposition 4 are proved. The exact boundaries are stated below and in `catalog.json`.

Everything builds with zero `sorry`, and `#print axioms` reports only the standard classical axioms (`propext`, `Classical.choice`, `Quot.sound`) for every listed Lean declaration. The files are:

- `Basic.lean` — the log-sum inequality and discrete KL divergence (shared engine);
- `ReadoutContraction.lean` — Lemma 1 (a readout can only decrease KL);
- `LocalForgettingBound.lean` — Proposition 2 (the Local Forgetting Bound);
- `InformationBound.lean` — Proposition 1 / Eq. (2) (the information bound);
- `QuadraticScrubbing.lean` — the finite-time algebraic core of Proposition 3;
- `RobustScrubbing.lean` — the isotropic optimization subclaim of Proposition 4;
- `SelectiveForgetting.lean` — the umbrella import.

## How to build

Requires a Lean 4 toolchain (`elan`/`lake`). Run `lake build` in this folder. The Mathlib version is pinned in `lakefile.toml` / `lake-manifest.json`.

## What is proved

The information-theoretic results (Lemma 1, Propositions 1–2) are modelled over finite types, following the paper's discrete proofs ("we will consider the random variables to be discrete"). A discrete distribution is a function $Q : \alpha \to \mathbb{R}$ with $Q \ge 0$; KL divergence is $\mathrm{KL}(Q \Vert R) = \sum_x Q(x)\, \log\!\big(Q(x)/R(x)\big)$. The Lean signatures impose full support on reference distributions and, where used, priors/channels/seed weights; these strict-positivity assumptions are stronger than the paper states and avoid extended-real zero/infinity conventions. All three results rest on one log-sum lemma.

- **`Basic.lean`** — the **log-sum inequality**: for $a_i \ge 0$ and $b_i > 0$ over a finite index set,
  $$\Big(\textstyle\sum_i a_i\Big)\, \log\frac{\sum_i a_i}{\sum_i b_i} \le \sum_i a_i\, \log\frac{a_i}{b_i},$$
  proved as Jensen's inequality for the convex function $t \mapsto t\,\log t$. From it, `klDiv_nonneg` gives Gibbs' inequality $\mathrm{KL}(Q \Vert R) \ge 0$.

- **`ReadoutContraction.lean`** — **Lemma 1**. For any readout $f$, with the pushforward $f_* Q(c) = \sum_{x : f(x) = c} Q(x)$, `klDiv_pushforward_le` proves $\mathrm{KL}(f_* Q \Vert f_* R) \le \mathrm{KL}(Q \Vert R)$, by the log-sum inequality on each level set $\{x : f(x)=c\}$ summed over $c$.

- **`LocalForgettingBound.lean`** — **Proposition 2**. For a finite seed distribution $p$ and per-seed distributions $Q_e, R_e$, `klDiv_mixture_le` proves the joint convexity of KL,
  $$\mathrm{KL}\Big(\textstyle\sum_e p_e Q_e \,\Big\Vert\, \sum_e p_e R_e\Big) \le \sum_e p_e\, \mathrm{KL}(Q_e \Vert R_e),$$
  the discrete analogue of the paper's $\mathbb{E}_\epsilon$ mixture, again from the log-sum inequality.

- **`InformationBound.lean`** — **Proposition 1** / Eq. (2). `markovMutualInfo_le` models $Y \leftarrow D_f \to Z$ by two conditionally independent finite channels and proves $I(Y;Z) \le I(D_f;Z)$ directly from the log-sum inequality. `mutualInfo_le_expected_klDiv` proves $I(D_f;Z) \le \sum_x p_X(x)\,\mathrm{KL}(p(\cdot\mid x)\Vert q)$ for any reference $q$, with gap $\mathrm{KL}(p_Z\Vert q)\ge 0$. `information_bound` composes the two proved steps; Data Processing is no longer a hypothesis.

- **`QuadraticScrubbing.lean`** — the **finite-time algebraic core of Proposition 3**. Assuming the two displayed closed-form paths from the same initialization, `quadratic_scrubbing_flow_identity` proves the substitution $h(w_A(t))=w_B(t)$; `quadratic_scrubbing_flow_identity_exp` instantiates the inverse factors with the genuine matrix exponential. The map uses the sign $e^{-Bt}(d_r-d)$ derived by the paper's proof, rather than the opposite sign printed in the statement. The file does not derive the paths from quadratic-loss ODEs, formalize the resulting equality of probability laws/conditional KL zero, or prove the $t\to\infty$ Newton update in Eq. (7).

- **`RobustScrubbing.lean`** — **Proposition 4, isotropic case only**. Under the paper's second-order Gaussian/quadratic approximation, the reduced objective is $\tfrac12\,\mathrm{tr}(B\Sigma)+\tfrac{c}{2}\,\mathrm{tr}(\Sigma^{-1})$, with $c=\lambda\sigma_h^2>0$. The proof completes the square in the trace inner product. `robust_scrubbing_isotropic_candidate_posDef` proves that $\Sigma^\*=\sqrt c\,B^{-1/2}$ is positive definite when $B^{1/2}$ is supplied as a positive-definite root, and `robust_scrubbing_isotropic_global_minimum` proves $g(\Sigma^\*)\le g(\Sigma)$ for every competitor supplied with a positive-definite square-root witness. The remaining declarations expose the lower bound, attainment, and $\Sigma^\*B\Sigma^\*=cI$. The general non-isotropic condition $\Sigma B\Sigma=\lambda\Sigma_h$ is not formalized.

The `catalog.json` maps the proved declarations to the paper and also records five unformalized claims explicitly: Proposition 3's conditional-KL consequence and Eq. (7) limit, the general non-isotropic part of Proposition 4, Example 1's infinite-noise limit, and Corollary 1's Gaussian specialization.
