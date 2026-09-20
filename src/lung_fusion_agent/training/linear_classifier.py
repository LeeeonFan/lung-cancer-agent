"""Epoch-based linear softmax classifier for frozen slide embeddings."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class LinearTrainingConfig:
    """Hyperparameters for the linear classifier."""

    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    class_weight: str = "none"
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
            raise ValueError("class_weight must be 'none' or 'balanced'")

        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be positive")


class LinearClassifier(nn.Module):
    """One affine layer producing 7-class logits."""

    def __init__(
        self,
        input_dimension: int,
        number_of_classes: int,
    ) -> None:
        super().__init__()

        self.input_dimension = input_dimension
        self.number_of_classes = number_of_classes

        self.linear = nn.Linear(
            input_dimension,
            number_of_classes,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(features)


@dataclass
class LinearTrainingResult:
    """Trained model and epoch-level history."""

    model: LinearClassifier
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_composite: float
    stopped_early: bool
    config: LinearTrainingConfig

    def checkpoint(
        self,
    ) -> dict[str, object]:
        return {
            "model_state_dict": (self.model.state_dict()),
            "input_dimension": (self.model.input_dimension),
            "number_of_classes": (self.model.number_of_classes),
            "best_epoch": self.best_epoch,
            "best_validation_composite": (self.best_validation_composite),
            "stopped_early": (self.stopped_early),
            "config": asdict(self.config),
        }


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def balanced_class_weights(
    labels: np.ndarray,
    number_of_classes: int,
) -> np.ndarray:
    """Compute inverse-frequency class weights."""

    counts = np.bincount(
        labels,
        minlength=number_of_classes,
    )

    if np.any(counts == 0):
        raise ValueError(
            "Balanced weighting requires "
            "every class in training. "
            f"Counts: {counts.tolist()}"
        )

    return len(labels) / (number_of_classes * counts)


def calculate_macro_auroc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    number_of_classes: int,
) -> float:
    """Calculate macro one-vs-rest AUROC."""

    try:
        return float(
            roc_auc_score(
                labels,
                probabilities,
                labels=np.arange(number_of_classes),
                multi_class="ovr",
                average="macro",
            )
        )

    except ValueError:
        return float("nan")


def predict_probabilities(
    model: nn.Module,
    features: torch.Tensor,
) -> np.ndarray:
    """Return softmax probabilities."""

    model.eval()

    with torch.inference_mode():
        probabilities = torch.softmax(
            model(features),
            dim=1,
        )

    return probabilities.detach().cpu().numpy()


def train_linear_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    number_of_classes: int,
    config: LinearTrainingConfig,
    device: torch.device | None = None,
) -> LinearTrainingResult:
    """Train a full-batch linear classifier."""

    config.validate()
    seed_everything(config.seed)

    selected_device = device if device is not None else choose_device()

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

    model = LinearClassifier(
        input_dimension=(train_features.shape[1]),
        number_of_classes=number_of_classes,
    ).to(selected_device)

    weight_tensor = None

    if config.class_weight == "balanced":
        class_weights = balanced_class_weights(
            train_labels,
            number_of_classes,
        )

        weight_tensor = torch.from_numpy(class_weights.astype(np.float32)).to(
            selected_device
        )

    training_criterion = nn.CrossEntropyLoss(weight=weight_tensor)

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
            updated_train_logits = model(train_features_tensor)

            validation_logits = model(validation_features_tensor)

            updated_training_loss = training_criterion(
                updated_train_logits,
                train_labels_tensor,
            )

            # Validation loss is unweighted.
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

        validation_predictions = validation_probabilities.argmax(axis=1)

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

        validation_composite = (validation_auroc + validation_ba) / 2

        history.append(
            {
                "epoch": epoch,
                "training_loss": float(updated_training_loss.cpu()),
                "validation_loss": float(validation_loss.cpu()),
                "validation_macro_auroc": (validation_auroc),
                "validation_balanced_accuracy": (validation_ba),
                "validation_composite": (validation_composite),
            }
        )

        improved = validation_composite > best_composite + config.minimum_improvement

        if improved:
            best_composite = validation_composite

            best_epoch = epoch

            best_state = copy.deepcopy(model.state_dict())

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        patience = config.early_stopping_patience

        if patience is not None and epochs_without_improvement >= patience:
            stopped_early = True
            break

    if config.restore_best_state:
        model.load_state_dict(best_state)

    return LinearTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_composite=(best_composite),
        stopped_early=stopped_early,
        config=config,
    )
