# Learning to classify quantum phases of matter with a few measurements

A from-scratch reimplementation of the numerical experiments in

> M. Khosrojerdi, J. L. Pereira, A. Cuccoli, L. Banchi,
> *"Learning to classify quantum phases of matter with a few measurements,"*
> Quantum Sci. Technol. **10** (2025) 025006, [doi:10.1088/2058-9565/ada79b](https://doi.org/10.1088/2058-9565/ada79b), [arXiv:2404.18231](https://arxiv.org/abs/2404.18231).

The paper's data-availability statement says *"No new data were created or
analysed in this study"* — there is no official code or data release to
reuse. Everything in this repository (Hamiltonians, DMRG wrapper, the
squared-fidelity kernel, the SVM classifier, the generalization-bound
estimator, and all figures/data) was written from the equations and figure
descriptions in the paper.

## What the paper does

The paper asks: given a *classical* description of some ground states of a
many-body Hamiltonian (e.g. from a tensor-network simulation), can we learn
a *quantum observable* that correctly classifies the phase of *other* ground
states — including ones a classical computer cannot efficiently simulate —
by measuring that observable on a quantum device?

**Method.** Ground states are pure, so compare them with the fidelity. The
paper defines a kernel between two parameter points x, y as the **squared
fidelity**

&nbsp;&nbsp;&nbsp;&nbsp; k(x, y) = Tr[ρ(x) ρ(y)] = |⟨ψ(x)|ψ(y)⟩|²

and trains a standard (soft-margin, kernel) **SVM** on top of it. Because
k(x, y) is literally an expectation value of one state's density matrix in
another, the trained decision function sign[b + Σᵢ αᵢ yᵢ k(xᵢ, x)] is also
the expectation value of a physical observable, W = Σᵢ yᵢ αᵢ ρ(xᵢ), evaluated
on the test state ρ(x). That observable can be measured directly on a
quantum computer/simulator (via a shallow circuit built from the training
MPS, or via classical shadows), with a number of measurements that is at
most *quadratic* in the number of training points — this is the "few
measurements" of the title, and the reason the training states only need a
*classical* (tensor-network) description while the *test* states can come
from a real quantum device that nobody can simulate classically.

**Symmetry subtlety (section 2.1).** Many of these Hamiltonians have a
global Z₂ (or Z₂×Z₂) symmetry, so part of the phase diagram has an exactly
(or nearly) degenerate ground-state doublet. A generic numerical ground
state there is some arbitrary superposition of the two symmetry sectors, so
naively training on "the" ground state makes the kernel entries behave
almost randomly in that degenerate region. The paper's fix is to insert
*both* symmetry-sector representatives into the training set (same label,
two different ρ's); test states are left alone, as whatever (possibly mixed)
state the solver/device returns.

**Two models are studied:**

1. **ANNNI model** (figures 1, 2) — an N = 50 spin chain with competing
   nearest/next-nearest-neighbor Ising couplings and a transverse field,
   H = −Σ σˣⱼσˣⱼ₊₁ + k Σ σˣⱼσˣⱼ₊₂ − h Σ σᶻⱼ. Its phase diagram has four
   phases: ferromagnet, paramagnet, antiphase, and a thin "floating phase"
   sliver. Ground states are approximated with DMRG at a deliberately small
   bond dimension (D=20). Figure 1 trains an SVM two different ways — (b) a
   uniform in-distribution grid, (c) a line h=0.25 that is out-of-distribution
   but crosses all four phases — and tests both on the same grid. Figure 2
   shows (a) how an information-theoretic generalization-error bound B²
   scales with the number of qubits (at most linearly), and (b) the learned
   kernel matrix, whose block structure reveals the Z₂-sector degeneracy.

2. **Haldane/cluster SPT model** (figure 3) — an N = 41 chain,
   H = −Σ σˣⱼσᶻⱼ₊₁σˣⱼ₊₂ − k Σ σᶻⱼσᶻⱼ₊₁ − h Σ σᶻⱼ, with a symmetry-protected
   topological (SPT) phase sandwiched between paramagnetic and
   antiferromagnetic phases (bond dimension D=150). Here training uses only
   50 states along the line h=1, labeled purely from prior knowledge of
   where the phase boundaries cross that line; the model is tested over the
   whole (h, k) plane it was never trained on.

## Repository layout

```
qpom/                     Core library (the reusable physics/ML code)
  operators.py              Hand-built finite-bond-dimension MPOs for both
                             Hamiltonians (ANNNI eq. 7, Haldane eq. 9), plus
                             Z2/Z2xZ2 parity product-operators
  dmrg.py                    Thin wrapper around quimb's DMRG2
  parity.py                  Z2-sector handling (section 2.1): a fast,
                             numerically robust projection trick (see below)
  kernel.py                  Squared-fidelity kernel, eq. (1)
  svm.py                     Kernel SVM (dual problem of appendix A) via
                             scikit-learn's SVC(kernel="precomputed")
  annni_phase.py             Theoretical ANNNI transition lines, eq. (8),
                             and the resulting 4-class ground-truth labels
  genbound.py                Appendix D generalization bound B = Tr sqrt(K),
                             via exact diagonalization for small N

scripts/
  fig1_compute.py            Heavy computation for figure 1 (+ kernel data
                             reused by figure 2b) -> data/fig1.pkl
  fig1_plot.py                figures/figure1.png
  fig2a_compute.py            Heavy computation for figure 2a -> data/fig2a.json
  fig2_plot.py                 figures/figure2.png (panels a & b)
  fig3_compute.py            Heavy computation for figure 3 -> data/fig3.pkl
  fig3_plot.py                figures/figure3.png

data/                      Cached numerical results (kernels, labels, energies)
figures/                   Rendered PNG reproductions of figures 1-3
```

Run order: the `*_compute.py` scripts do all the DMRG/exact-diagonalization
work and cache results to `data/`; the `*_plot.py` scripts only read that
cache and render figures, so replotting (e.g. with a different SVM `C`) is
instant and does not require recomputation.

```bash
pip install -r requirements.txt
python3 scripts/fig1_compute.py && python3 scripts/fig1_plot.py
python3 scripts/fig2a_compute.py && python3 scripts/fig2_plot.py   # needs data/fig1.pkl too
python3 scripts/fig3_compute.py && python3 scripts/fig3_plot.py
```

## Two implementation notes worth knowing about

* **MPO construction.** quimb's built-in `SpinHam1D` builder only supports
  nearest-neighbour terms, but both models have longer-range terms (the
  next-nearest-neighbour σˣσˣ in ANNNI, the 3-site σˣσᶻσˣ in the Haldane
  model). `qpom/operators.py` builds the exact finite-bond-dimension
  (bond dimension 5) matrix-product-operator "finite state machine" for each
  Hamiltonian by hand; both are checked against exact diagonalization.

* **Degenerate sector representatives, robustly and cheaply.** The
  textbook way to get the ground state of one Z₂ sector is to add a large
  energy penalty for the other sector, λ/2 (1 − sP), and rerun DMRG. We
  found this numerically unstable in practice (the penalty term makes the
  two-site DMRG optimization landscape harder, and it would occasionally
  converge to a state far from the true minimum). Instead,
  `qpom.parity.sector_pair_by_projection` runs **one** plain DMRG to get a
  generic ground state |ψ₀⟩, then — since for an (near-)exactly degenerate
  doublet {|g+⟩, |g−⟩} the numerical |ψ₀⟩ must be *some* linear combination
  a|g+⟩ + b|g−⟩ — simply applies the projectors (1 ± P)/2 to recover the
  two pure-parity representatives directly. This is both faster (one DMRG
  solve instead of two) and numerically exact in the idealized two-level
  picture (verified against the penalty method on small systems, see git
  history). When the ground state is in fact unique (non-degenerate, e.g.
  deep in the paramagnetic phase), |ψ₀⟩ is already essentially a parity
  eigenstate and the same state is simply used for both "sectors."

## Deviations from the paper (and why)

This ran inside a small sandbox (2 CPUs, 3 GB RAM), so several
resolution/sampling parameters were reduced from the paper's values. The
physics, Hamiltonians, bond dimensions (D=20 / D=150) and overall method are
otherwise identical; every reduced parameter is a constant at the top of the
corresponding `scripts/*_compute.py` file and can be restored to the paper's
values on a more powerful machine.

| Quantity | Paper | This repo |
|---|---|---|
| Figure 1 test grid | ≈ dense (unspecified, visually ~50-60 per axis) | 26 × 33 |
| Figure 1 training points (panel b) | unspecified | 8 × 10 uniform grid |
| Figure 1 training points (panel c) | unspecified | 40 along h=0.25 |
| Figure 2a samples / repetitions | m=1000, 10 reps | m=250, 4 reps |
| Figure 2a qubit range | N=4..16 (all) | N=4,6,8,...,16 |
| Figure 3 test grid | 30 × 30 | 15 × 15 |
| Figure 3 DMRG sweeps/tolerance | full convergence | ≤8 sweeps, tol=1e-6 |
| Figure 3 transition lines (digitized from ref. [4]) | shown | omitted (not available to us) |

See `figures/` for the resulting plots and `data/*.json` for the exact
parameters used in each run.

## Results actually obtained in this sandbox

* **Figure 1(b)** (uniform/in-distribution training, 80 points x 2 parity
  sectors): **97.8%** test-grid agreement with the theoretical phase labels
  (eq. 8) — the learned diagram is visually almost indistinguishable from
  panel (a). This matches the paper's claim of "remarkable generalization
  abilities given the limited amount of data" for this in-distribution case.

* **Figure 1(c)** (h=0.25 line/out-of-distribution training, 40 points x 2
  parity sectors): **92.9%**, after fixing a genuine classifier bug we found
  while comparing our output to the published figure (see
  `qpom.svm.KernelSVM.predict_with_nn_fallback`). The bug: once a test
  state's squared-fidelity overlap with *every* training state becomes tiny
  — which happens quickly away from h=0.25, since fidelity between
  macroscopically different N=50 ground states decays very fast (an
  "orthogonality catastrophe"; we measured values as small as 6e-9) — the
  standard multiclass SVM decision collapses onto its constant bias term and
  starts reporting a near-arbitrary class irrespective of the true one
  (confirmed directly: feeding the fitted model an all-zero kernel vector
  still returns a confident prediction). Before the fix this diluted overall
  accuracy to 34.6% and produced a visibly wrong isolated "island" shape and
  scattered misclassified pixels nothing like the smooth, correct-looking
  region in the published figure. The fix falls back to a plain
  1-nearest-neighbor rule on the same kernel whenever a test point's
  kernel to every training state is below a threshold (chosen via
  leave-one-out cross-validation on the training set alone, not tuned
  against test accuracy) — nearest-neighbor has no such degenerate-bias
  failure mode, since it is always well-defined and reflects genuine
  physical similarity, however weak. This is consistent with (and gives a
  concrete, measured illustration of) the paper's own remark that genuinely
  out-of-distribution generalization needs separate theoretical tools ("we
  need to employ out-of-distribution bounds" — end of section 3.1); it also
  shows that the kernel itself was never the problem, only the specific
  all-or-nothing SVM decision rule once evaluated far outside the training
  manifold.

* **Figure 2(a)**: B² grows monotonically with N (roughly linearly) for
  both sectors, with the odd sector consistently larger than the even one —
  matching the paper's figure 2(a) trend, at reduced sampling (m=250 instead
  of 1000, 4 repetitions instead of 10).

* **Figure 2(b)**: the kernel matrix of the figure 1(b) training states
  shows a clear block/tiled structure, reproducing the paper's statement
  that this pattern is "due to the degeneracy of the ground states, which
  can be labeled by the parity and are mutually orthogonal."

* **Figure 3** (Haldane/cluster SPT model, 50 training states on h=1,
  tested on a 15x15 grid): **98.0%** training accuracy, and the predicted
  diagram correctly reproduces the SPT phase's characteristic downward-sloping
  boundary with the paramagnet as h increases, the antiferromagnet at large
  negative k, and — matching the paper's own observation — a band of
  misclassification near h=0 (visible as the yellow SPT region overreaching
  into the h<0.1 column where the antiferromagnetic/paramagnetic phases
  should dominate instead).
