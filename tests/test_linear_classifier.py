import numpy as np
import torch

from lung_fusion_agent.training import (
    LinearTrainingConfig,
    predict_probabilities,
    train_linear_classifier,
)


def test_linear_classifier_outputs_valid_probabilities() -> None:
    generator = np.random.default_rng(42)

    train_features = generator.normal(size=(42, 8)).astype(np.float32)

    train_labels = np.repeat(
        np.arange(7),
        6,
    )

    validation_features = generator.normal(size=(14, 8)).astype(np.float32)

    validation_labels = np.repeat(
        np.arange(7),
        2,
    )

    result = train_linear_classifier(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=(validation_features),
        validation_labels=(validation_labels),
        number_of_classes=7,
        config=LinearTrainingConfig(
            epochs=3,
            learning_rate=1e-3,
            weight_decay=1e-3,
            class_weight="balanced",
            seed=42,
        ),
        device=torch.device("cpu"),
    )

    probabilities = predict_probabilities(
        result.model,
        torch.from_numpy(validation_features),
    )

    assert probabilities.shape == (
        14,
        7,
    )

    assert np.isfinite(probabilities).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    )

    assert len(result.history) == 3

    assert 1 <= result.best_epoch <= 3
