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

from dataclasses import dataclass

import numpy as np
from sklearn.svm import SVC


@dataclass
class KernelSVM:
    C: float = 1.0
    model: SVC | None = None

    def fit(self, K_train: np.ndarray, y_train: np.ndarray) -> "KernelSVM":
        self.model = SVC(C=self.C, kernel="precomputed", decision_function_shape="ovo")
        self.model.fit(K_train, y_train)
        return self

    def predict(self, K_test_train: np.ndarray) -> np.ndarray:
        assert self.model is not None, "call .fit first"
        return self.model.predict(K_test_train)

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
