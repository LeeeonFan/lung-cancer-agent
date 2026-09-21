"""Small MLP classifier for frozen slide embeddings."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.nn import functional

from lung_fusion_agent.training.linear_classifier import (
    balanced_class_weights,
    calculate_macro_auroc,
    choose_device,
    seed_everything,
)


@dataclass(frozen=True)
class MLPTrainingConfig:
    """Hyperparameters for the small MLP classifier."""

    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    class_weight: str = "none"
    hidden_dimension: int = 32
    dropout: float = 0.3
    seed: int = 42
    early_stopping_patience: int | None = None
    minimum_improvement: float = 1e-3
    restore_best_state: bool = True

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")

        if self.class_weight not in {
            "none",
            "balanced",
        }:
            raise ValueError(
                "class_weight must be 'none' or 'balanced'"
            )

        if self.hidden_dimension < 1:
            raise ValueError(
                "hidden_dimension must be at least 1"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in the interval [0, 1)"
            )

        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 1
        ):
            raise ValueError(
                "early_stopping_patience must be positive"
            )


class MLPClassifier(nn.Module):
    """One-hidden-layer MLP producing multiclass logits."""

    def __init__(
        self,
        input_dimension: int,
        number_of_classes: int,
        hidden_dimension: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dimension = input_dimension
        self.number_of_classes = number_of_classes
        self.hidden_dimension = hidden_dimension
        self.dropout_probability = dropout

        self.network = nn.Sequential(
            nn.Linear(
                input_dimension,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dimension,
                number_of_classes,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(features)


@dataclass
class MLPTrainingResult:
    """Trained MLP model and epoch-level history."""

    model: MLPClassifier
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_composite: float
    stopped_early: bool
    config: MLPTrainingConfig

    def checkpoint(
        self,
    ) -> dict[str, object]:
        return {
            "model_state_dict": self.model.state_dict(),
            "input_dimension": self.model.input_dimension,
            "number_of_classes": (
                self.model.number_of_classes
            ),
            "hidden_dimension": (
                self.model.hidden_dimension
            ),
            "dropout": (
                self.model.dropout_probability
            ),
            "best_epoch": self.best_epoch,
            "best_validation_composite": (
                self.best_validation_composite
            ),
            "stopped_early": self.stopped_early,
            "config": asdict(self.config),
        }


def train_mlp_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    number_of_classes: int,
    config: MLPTrainingConfig,
    device: torch.device | None = None,
) -> MLPTrainingResult:
    """Train a full-batch one-hidden-layer MLP."""

    config.validate()
    seed_everything(config.seed)

    selected_device = (
        device
        if device is not None
        else choose_device()
    )

    train_features_tensor = torch.from_numpy(
        np.asarray(
            train_features,
            dtype=np.float32,
        )
    ).to(selected_device)

    validation_features_tensor = torch.from_numpy(
        np.asarray(
            validation_features,
            dtype=np.float32,
        )
    ).to(selected_device)

    train_labels_tensor = torch.from_numpy(
        np.asarray(
            train_labels,
            dtype=np.int64,
        )
    ).to(selected_device)

    validation_labels_tensor = torch.from_numpy(
        np.asarray(
            validation_labels,
            dtype=np.int64,
        )
    ).to(selected_device)

    model = MLPClassifier(
        input_dimension=train_features.shape[1],
        number_of_classes=number_of_classes,
        hidden_dimension=config.hidden_dimension,
        dropout=config.dropout,
    ).to(selected_device)

    weight_tensor = None

    if config.class_weight == "balanced":
        class_weights = balanced_class_weights(
            train_labels,
            number_of_classes,
        )

        weight_tensor = torch.from_numpy(
            class_weights.astype(np.float32)
        ).to(selected_device)

    training_criterion = nn.CrossEntropyLoss(
        weight=weight_tensor
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float | int]] = []

    best_epoch = 0
    best_composite = float("-inf")
    best_state = copy.deepcopy(model.state_dict())

    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(
        1,
        config.epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(set_to_none=True)

        logits = model(train_features_tensor)

        training_loss = training_criterion(
            logits,
            train_labels_tensor,
        )

        training_loss.backward()
        optimizer.step()

        model.eval()

        with torch.inference_mode():
            updated_train_logits = model(
                train_features_tensor
            )

            validation_logits = model(
                validation_features_tensor
            )

            updated_training_loss = (
                training_criterion(
                    updated_train_logits,
                    train_labels_tensor,
                )
            )

            validation_loss = functional.cross_entropy(
                validation_logits,
                validation_labels_tensor,
            )

            validation_probabilities = (
                torch.softmax(
                    validation_logits,
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )

        validation_predictions = (
            validation_probabilities.argmax(axis=1)
        )

        validation_auroc = calculate_macro_auroc(
            validation_labels,
            validation_probabilities,
            number_of_classes,
        )

        validation_ba = float(
            balanced_accuracy_score(
                validation_labels,
                validation_predictions,
            )
        )

        validation_composite = (
            validation_auroc + validation_ba
        ) / 2

        history.append(
            {
                "epoch": epoch,
                "training_loss": float(
                    updated_training_loss.cpu()
                ),
                "validation_loss": float(
                    validation_loss.cpu()
                ),
                "validation_macro_auroc": (
                    validation_auroc
                ),
                "validation_balanced_accuracy": (
                    validation_ba
                ),
                "validation_composite": (
                    validation_composite
                ),
            }
        )

        improved = (
            validation_composite
            > best_composite
            + config.minimum_improvement
        )

        if improved:
            best_composite = validation_composite
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        patience = config.early_stopping_patience

        if (
            patience is not None
            and epochs_without_improvement
            >= patience
        ):
            stopped_early = True
            break

    if config.restore_best_state:
        model.load_state_dict(best_state)

    return MLPTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_composite=best_composite,
        stopped_early=stopped_early,
        config=config,
    )