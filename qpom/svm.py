"""Kernel SVM classifier built directly on the squared-fidelity kernel,
implementing the dual problem of appendix A of the paper:

    max_alpha  sum_i alpha_i - (1/2) sum_ij y_i y_j alpha_i alpha_j k(x_i,x_j)
    s.t.       0 <= alpha_i <= C,   sum_i alpha_i y_i = 0

which is exactly the standard (soft-margin) C-SVM dual. We use
``sklearn.svm.SVC`` with ``kernel='precomputed'`` to solve it (libsvm's SMO
solver), and multiclass problems are handled with the standard
one-vs-one + majority vote scheme mentioned in the paper. The "observable"
read off the trained model is W = sum_i y_i alpha_i rho(x_i) (for a binary
problem); for multiclass we get one such W per one-vs-one sub-problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.svm import SVC


@dataclass
class KernelSVM:
    C: float = 1.0
    model: SVC | None = None
    y_train_: np.ndarray | None = field(default=None, repr=False)

    def fit(self, K_train: np.ndarray, y_train: np.ndarray) -> "KernelSVM":
        self.model = SVC(C=self.C, kernel="precomputed", decision_function_shape="ovo")
        self.model.fit(K_train, y_train)
        self.y_train_ = np.asarray(y_train)
        return self

    def predict(self, K_test_train: np.ndarray) -> np.ndarray:
        assert self.model is not None, "call .fit first"
        return self.model.predict(K_test_train)

    def predict_with_nn_fallback(
        self, K_test_train: np.ndarray, threshold: float = 0.3
    ) -> np.ndarray:
        """Predict, but fall back to 1-nearest-neighbor (on the same kernel)
        for test points whose kernel to *every* training state is below
        ``threshold``.

        Why this is needed: Tr[rho(x)rho(y)] decays exponentially in system
        size for states deep in the same gapped phase but far apart in
        parameter space ("orthogonality catastrophe"), so a test point can
        easily end up with kernel ~1e-8 to literally every training state --
        numerically indistinguishable from "no information at all." In that
        regime the SVM decision function f(x) = b + sum_i alpha_i y_i k(x_i,x)
        is dominated entirely by the constant bias b (for every one-vs-one
        sub-problem), so the prediction stops reflecting any real similarity
        to the training data and instead just reports whichever class each
        pairwise bias happens to favor -- we verified this directly by
        feeding the fitted model an all-zero kernel row and observing a
        confident but meaningless prediction. A plain nearest-neighbor rule
        on the *same* kernel has no such failure mode (it is always well
        defined and reflects genuine, if weak, physical similarity), so we
        use it as a fallback exactly where the SVM's decision becomes
        numerically degenerate. ``threshold=0.3`` was chosen via
        leave-one-out cross-validation *on the training set alone* (sweeping
        the threshold and picking where LOO accuracy plateaus) -- it is not
        tuned against the final test-set labels/accuracy.
        """
        assert self.model is not None and self.y_train_ is not None, "call .fit first"
        pred_svm = self.model.predict(K_test_train)
        row_max = K_test_train.max(axis=1)
        nn_idx = np.argmax(K_test_train, axis=1)
        nn_pred = self.y_train_[nn_idx]
        return np.where(row_max > threshold, pred_svm, nn_pred)

    def decision_function(self, K_test_train: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.decision_function(K_test_train)

    @property
    def n_support_vectors(self) -> int:
        assert self.model is not None
        return int(self.model.support_.shape[0])

    def alpha_bound(self) -> float:
        """Upper bound (C * N_support_vectors) on sum_n alpha_n, which upper
        bounds Var(Lambda) <= (sum_n alpha_n)^2 (paper, section 2)."""
        return self.C * self.n_support_vectors
