"""Tests for the small MLP classifier."""

import numpy as np
import torch

from lung_fusion_agent.training.mlp_classifier import (
    MLPClassifier,
    MLPTrainingConfig,
    train_mlp_classifier,
)


def test_mlp_output_shape() -> None:
    model = MLPClassifier(
        input_dimension=16,
        number_of_classes=7,
        hidden_dimension=8,
        dropout=0.3,
    )

    features = torch.zeros(
        (5, 16),
        dtype=torch.float32,
    )

    logits = model(features)

    assert logits.shape == (5, 7)
    assert torch.isfinite(logits).all()


def test_mlp_training_returns_finite_history() -> None:
    generator = np.random.default_rng(42)

    train_features = generator.normal(
        size=(70, 16)
    ).astype(np.float32)

    validation_features = generator.normal(
        size=(35, 16)
    ).astype(np.float32)

    train_labels = np.tile(
        np.arange(7),
        10,
    ).astype(np.int64)

    validation_labels = np.tile(
        np.arange(7),
        5,
    ).astype(np.int64)

    config = MLPTrainingConfig(
        epochs=3,
        learning_rate=1e-3,
        weight_decay=1e-3,
        class_weight="balanced",
        hidden_dimension=8,
        dropout=0.3,
        seed=42,
        restore_best_state=True,
    )

    result = train_mlp_classifier(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        number_of_classes=7,
        config=config,
        device=torch.device("cpu"),
    )

    assert len(result.history) == 3
    assert 1 <= result.best_epoch <= 3

    for row in result.history:
        assert np.isfinite(
            row["training_loss"]
        )
        assert np.isfinite(
            row["validation_loss"]
        )
        assert np.isfinite(
            row["validation_macro_auroc"]
        )
        assert np.isfinite(
            row["validation_balanced_accuracy"]
        )
        assert np.isfinite(
            row["validation_composite"]
        )