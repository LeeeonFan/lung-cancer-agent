from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    macro_auroc: float
    balanced_accuracy: float
    per_class_auroc: dict[str, float]


def calculate_classification_metrics(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> ClassificationMetrics:
    expected_classes = np.arange(len(class_names))

    if probabilities.shape != (
        len(y_true),
        len(class_names),
    ):
        raise ValueError(
            f"Probability shape does not match labels/classes: {probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or infinity.")

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5,
    ):
        raise ValueError("Probability rows do not sum to one.")

    predicted_labels = probabilities.argmax(axis=1)

    macro_auroc = roc_auc_score(
        y_true,
        probabilities,
        labels=expected_classes,
        average="macro",
        multi_class="ovr",
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predicted_labels,
    )

    per_class_auroc = {}

    for class_index, class_name in enumerate(class_names):
        binary_target = (y_true == class_index).astype(np.int32)

        per_class_auroc[class_name] = float(
            roc_auc_score(
                binary_target,
                probabilities[:, class_index],
            )
        )

    return ClassificationMetrics(
        macro_auroc=float(macro_auroc),
        balanced_accuracy=float(balanced_accuracy),
        per_class_auroc=per_class_auroc,
    )
