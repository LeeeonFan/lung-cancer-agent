"""Compare Linear and small-MLP classifier families."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LINEAR_ROOT = Path("artifacts/results/linear_cv")

MLP_ROOT = Path("artifacts/results/mlp_cv")

OUTPUT_DIRECTORY = Path("artifacts/results/classifier_comparison")

REPRESENTATIONS = (
    "metadata",
    "uni2",
    "virchow2",
    "prism2",
    "prism2_metadata",
    "early_concat",
    "early_l2_concat",
)

IMAGE_REPRESENTATIONS = (
    "uni2",
    "virchow2",
    "prism2",
    "prism2_metadata",
    "early_concat",
    "early_l2_concat",
)

MINIMUM_MEAN_IMPROVEMENT = 0.01
MINIMUM_REPRESENTATION_WINS = 4
MAXIMUM_VARIABILITY_RATIO = 1.25


def load_selected_config(
    root: Path,
    representation: str,
) -> dict[str, object]:
    path = root / representation / "selected_config.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing selected config: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def load_best_trial(
    root: Path,
    representation: str,
) -> pd.Series:
    path = root / representation / "cv_trials.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing CV trials: {path}")

    trials = pd.read_csv(path)

    required_columns = {
        "selection_score",
        "macro_auroc_mean",
        "macro_auroc_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "composite_mean",
        "composite_std",
        "median_best_epoch",
    }

    missing_columns = required_columns - set(trials.columns)

    if missing_columns:
        raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")

    best_index = trials["selection_score"].idxmax()

    return trials.loc[best_index]


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_records: list[dict[str, object]] = []

    for representation in REPRESENTATIONS:
        linear_config = load_selected_config(
            LINEAR_ROOT,
            representation,
        )

        mlp_config = load_selected_config(
            MLP_ROOT,
            representation,
        )

        linear_trial = load_best_trial(
            LINEAR_ROOT,
            representation,
        )

        mlp_trial = load_best_trial(
            MLP_ROOT,
            representation,
        )

        linear_score = float(linear_trial["selection_score"])

        mlp_score = float(mlp_trial["selection_score"])

        comparison_records.append(
            {
                "representation": (representation),
                "linear_macro_auroc": float(linear_trial["macro_auroc_mean"]),
                "mlp_macro_auroc": float(mlp_trial["macro_auroc_mean"]),
                "auroc_difference": (
                    float(mlp_trial["macro_auroc_mean"])
                    - float(linear_trial["macro_auroc_mean"])
                ),
                "linear_balanced_accuracy": (
                    float(linear_trial["balanced_accuracy_mean"])
                ),
                "mlp_balanced_accuracy": float(mlp_trial["balanced_accuracy_mean"]),
                "balanced_accuracy_difference": (
                    float(mlp_trial["balanced_accuracy_mean"])
                    - float(linear_trial["balanced_accuracy_mean"])
                ),
                "linear_selection_score": (linear_score),
                "mlp_selection_score": (mlp_score),
                "selection_score_difference": (mlp_score - linear_score),
                "linear_composite_std": float(linear_trial["composite_std"]),
                "mlp_composite_std": float(mlp_trial["composite_std"]),
                "linear_fixed_epochs": int(linear_config["fixed_epochs"]),
                "mlp_fixed_epochs": int(mlp_config["fixed_epochs"]),
                "mlp_wins": (mlp_score > linear_score),
            }
        )

    comparison = pd.DataFrame(comparison_records)

    comparison.to_csv(
        OUTPUT_DIRECTORY / "classifier_comparison.csv",
        index=False,
    )

    image_comparison = comparison.loc[
        comparison["representation"].isin(IMAGE_REPRESENTATIONS)
    ].reset_index(drop=True)

    mean_improvement = float(image_comparison["selection_score_difference"].mean())

    representation_wins = int(image_comparison["mlp_wins"].sum())

    prism_family_improved = bool(
        image_comparison.loc[
            image_comparison["representation"].isin(
                (
                    "prism2",
                    "prism2_metadata",
                )
            ),
            "mlp_wins",
        ].any()
    )

    linear_mean_variability = float(image_comparison["linear_composite_std"].mean())

    mlp_mean_variability = float(image_comparison["mlp_composite_std"].mean())

    if linear_mean_variability > 0:
        variability_ratio = mlp_mean_variability / linear_mean_variability
    else:
        variability_ratio = float("inf")

    variability_acceptable = variability_ratio <= MAXIMUM_VARIABILITY_RATIO

    selection_checks = {
        "minimum_mean_improvement": {
            "threshold": (MINIMUM_MEAN_IMPROVEMENT),
            "observed": mean_improvement,
            "passed": (mean_improvement >= MINIMUM_MEAN_IMPROVEMENT),
        },
        "minimum_representation_wins": {
            "threshold": (MINIMUM_REPRESENTATION_WINS),
            "observed": (representation_wins),
            "passed": (representation_wins >= MINIMUM_REPRESENTATION_WINS),
        },
        "prism_family_improved": {
            "required": True,
            "observed": (prism_family_improved),
            "passed": (prism_family_improved),
        },
        "variability_ratio": {
            "maximum": (MAXIMUM_VARIABILITY_RATIO),
            "observed": (variability_ratio),
            "passed": (variability_acceptable),
        },
    }

    select_mlp = all(bool(check["passed"]) for check in selection_checks.values())

    selected_classifier = "small_mlp" if select_mlp else "linear"

    summary = {
        "selected_classifier_family": (selected_classifier),
        "selection_rule_predeclared": True,
        "image_representations": list(IMAGE_REPRESENTATIONS),
        "mean_selection_score_improvement": (mean_improvement),
        "mlp_representation_wins": (representation_wins),
        "total_image_representations": (len(IMAGE_REPRESENTATIONS)),
        "prism_family_improved": (prism_family_improved),
        "linear_mean_composite_std": (linear_mean_variability),
        "mlp_mean_composite_std": (mlp_mean_variability),
        "variability_ratio": (variability_ratio),
        "checks": selection_checks,
        "validation_set_evaluated": False,
        "test_set_evaluated": False,
    }

    (OUTPUT_DIRECTORY / "selection_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    display_columns = [
        "representation",
        "linear_macro_auroc",
        "mlp_macro_auroc",
        "linear_balanced_accuracy",
        "mlp_balanced_accuracy",
        "linear_selection_score",
        "mlp_selection_score",
        "selection_score_difference",
        "mlp_wins",
    ]

    print()
    print("=== Classifier-family comparison ===")

    print(comparison[display_columns].to_string(index=False))

    print()
    print("=== Predeclared selection checks ===")

    for name, check in selection_checks.items():
        print(f"{name}: {'PASS' if check['passed'] else 'FAIL'} ({check})")

    print()
    print(
        "Selected classifier family:",
        selected_classifier,
    )

    print()
    print("Fixed validation was not evaluated.")

    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
