import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


class MetadataGatedLogitFusion:
    """Metadata-conditioned fusion over model probabilities."""

    def __init__(
        self,
        *,
        prior_weights: np.ndarray,
        regularization: float,
        class_weight_balanced: bool,
        max_iterations: int = 500,
        epsilon: float = 1e-7,
    ) -> None:
        prior_weights = np.asarray(
            prior_weights,
            dtype=np.float64,
        )

        if prior_weights.ndim != 1:
            raise ValueError("Prior weights must be one-dimensional.")

        if (prior_weights <= 0).any():
            raise ValueError("Prior weights must be positive.")

        if regularization < 0:
            raise ValueError("Regularization must be non-negative.")

        self.prior_weights = prior_weights / prior_weights.sum()
        self.regularization = regularization
        self.class_weight_balanced = class_weight_balanced
        self.max_iterations = max_iterations
        self.epsilon = epsilon

        self.metadata_mean_: np.ndarray | None = None
        self.metadata_scale_: np.ndarray | None = None
        self.coefficients_: np.ndarray | None = None
        self.optimization_success_: bool | None = None
        self.optimization_message_: str | None = None

    def _prepare_metadata(
        self,
        metadata: np.ndarray,
        *,
        fit: bool,
    ) -> np.ndarray:
        metadata = np.asarray(
            metadata,
            dtype=np.float64,
        )

        if metadata.ndim != 2:
            raise ValueError("Metadata must be two-dimensional.")

        if metadata.shape[1] != 2:
            raise ValueError("Expected age and sex metadata.")

        if not np.isfinite(metadata).all():
            raise ValueError("Metadata contains NaN or infinity.")

        if fit:
            self.metadata_mean_ = metadata.mean(axis=0)

            self.metadata_scale_ = metadata.std(
                axis=0,
                ddof=0,
            )

            self.metadata_scale_[self.metadata_scale_ < 1e-12] = 1.0

        if self.metadata_mean_ is None or self.metadata_scale_ is None:
            raise RuntimeError("Metadata scaler is not fitted.")

        standardized = (metadata - self.metadata_mean_) / self.metadata_scale_

        intercept = np.ones(
            (
                len(metadata),
                1,
            ),
            dtype=np.float64,
        )

        return np.column_stack(
            [
                standardized,
                intercept,
            ]
        )

    def _prepare_probabilities(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if probabilities.ndim != 3:
            raise ValueError(
                "Probabilities must have shape (patients, streams, classes)."
            )

        if probabilities.shape[1] != len(self.prior_weights):
            raise ValueError("Probability stream count does not match prior weights.")

        if not np.isfinite(probabilities).all():
            raise ValueError("Probabilities contain NaN or infinity.")

        return np.clip(
            probabilities,
            self.epsilon,
            1.0,
        )

    def _sample_weights(
        self,
        labels: np.ndarray,
    ) -> np.ndarray:
        if not self.class_weight_balanced:
            return np.ones(
                len(labels),
                dtype=np.float64,
            )

        class_count = int(labels.max()) + 1

        counts = np.bincount(
            labels,
            minlength=class_count,
        ).astype(np.float64)

        if (counts <= 0).any():
            raise ValueError("A class is absent from the gating training data.")

        class_weights = len(labels) / (class_count * counts)

        return class_weights[labels]

    def fit(
        self,
        *,
        metadata: np.ndarray,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> "MetadataGatedLogitFusion":
        probabilities = self._prepare_probabilities(probabilities)

        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        if labels.shape != (len(probabilities),):
            raise ValueError("Label count mismatch.")

        metadata_design = self._prepare_metadata(
            metadata,
            fit=True,
        )

        log_probabilities = np.log(probabilities)

        sample_weights = self._sample_weights(labels)

        sample_weight_sum = float(sample_weights.sum())

        prior_logits = np.log(self.prior_weights)

        parameter_shape = (
            metadata_design.shape[1],
            probabilities.shape[1],
        )

        initial_parameters = np.zeros(
            parameter_shape,
            dtype=np.float64,
        )

        def objective(
            flat_parameters: np.ndarray,
        ) -> tuple[
            float,
            np.ndarray,
        ]:
            coefficients = flat_parameters.reshape(parameter_shape)

            gate_logits = metadata_design @ coefficients + prior_logits

            gates = softmax(
                gate_logits,
                axis=1,
            )

            fused_logits = np.einsum(
                "ns,nsc->nc",
                gates,
                log_probabilities,
            )

            fused_probabilities = softmax(
                fused_logits,
                axis=1,
            )

            row_indices = np.arange(len(labels))

            negative_log_likelihood = (
                -np.sum(
                    sample_weights
                    * np.log(
                        np.clip(
                            fused_probabilities[
                                row_indices,
                                labels,
                            ],
                            self.epsilon,
                            1.0,
                        )
                    )
                )
                / sample_weight_sum
            )

            penalty = 0.5 * self.regularization * np.sum(coefficients**2)

            loss = negative_log_likelihood + penalty

            probability_gradient = fused_probabilities.copy()

            probability_gradient[
                row_indices,
                labels,
            ] -= 1.0

            probability_gradient *= (
                sample_weights[
                    :,
                    None,
                ]
                / sample_weight_sum
            )

            gate_gradient = np.einsum(
                "nc,nsc->ns",
                probability_gradient,
                log_probabilities,
            )

            centered_gate_gradient = gates * (
                gate_gradient
                - np.sum(
                    gates * gate_gradient,
                    axis=1,
                    keepdims=True,
                )
            )

            coefficient_gradient = (
                metadata_design.T @ centered_gate_gradient
                + self.regularization * coefficients
            )

            return (
                float(loss),
                coefficient_gradient.ravel(),
            )

        result = minimize(
            objective,
            initial_parameters.ravel(),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": (self.max_iterations),
                "ftol": 1e-9,
            },
        )

        self.coefficients_ = result.x.reshape(parameter_shape)

        self.optimization_success_ = bool(result.success)
        self.optimization_message_ = str(result.message)

        if not np.isfinite(self.coefficients_).all():
            raise RuntimeError("Gating optimization produced invalid coefficients.")

        return self

    def predict_gates(
        self,
        metadata: np.ndarray,
    ) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("Gating model is not fitted.")

        metadata_design = self._prepare_metadata(
            metadata,
            fit=False,
        )

        gate_logits = metadata_design @ self.coefficients_ + np.log(self.prior_weights)

        return softmax(
            gate_logits,
            axis=1,
        ).astype(np.float32)

    def predict_proba(
        self,
        *,
        metadata: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        probabilities = self._prepare_probabilities(probabilities)

        gates = self.predict_gates(metadata).astype(np.float64)

        fused_logits = np.einsum(
            "ns,nsc->nc",
            gates,
            np.log(probabilities),
        )

        fused_probabilities = softmax(
            fused_logits,
            axis=1,
        )

        return fused_probabilities.astype(np.float32)
