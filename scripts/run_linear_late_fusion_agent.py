"""Run late-fusion search on PyTorch Linear OOF probabilities."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from lung_fusion_agent.agent import (
    LateFusionSearchAgent,
)

INPUT_DIRECTORY = Path("artifacts/results/linear_fusion_agent")

CACHE_PATH = INPUT_DIRECTORY / "linear_oof_split_predictions.npz"

STREAM_SUMMARY_PATH = INPUT_DIRECTORY / "oof_stream_summary.csv"

MAX_TRIALS = 8
PATIENCE = 4
MINIMUM_IMPROVEMENT = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fusion-method",
        choices=(
            "probability",
            "logit",
        ),
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with np.load(
        CACHE_PATH,
        allow_pickle=False,
    ) as cache:
        labels = cache["labels"].astype(np.int64)

        class_names = cache["class_names"].astype(str).tolist()

        stream_names = cache["stream_names"].astype(str).tolist()

        validation_masks = cache["validation_masks"].astype(bool)

        probabilities = cache["probabilities"].astype(np.float32)

    expected_probability_shape = (
        15,
        141,
        4,
        7,
    )

    if probabilities.shape != (expected_probability_shape):
        raise ValueError(f"Unexpected probability shape: {probabilities.shape}")

    if validation_masks.shape != (
        15,
        141,
    ):
        raise ValueError(f"Unexpected validation-mask shape: {validation_masks.shape}")

    stream_summary = pd.read_csv(STREAM_SUMMARY_PATH)

    standalone_scores = {
        str(row["stream"]): float(row["composite_mean"])
        for _, row in (stream_summary.iterrows())
    }

    if set(standalone_scores) != set(stream_names):
        raise ValueError("Stream-summary names do not match OOF cache stream names")

    agent = LateFusionSearchAgent(
        stream_names=stream_names,
        standalone_scores=(standalone_scores),
        max_trials=MAX_TRIALS,
        patience=PATIENCE,
        minimum_improvement=(MINIMUM_IMPROVEMENT),
        fusion_method=(args.fusion_method),
    )

    trials = agent.run(
        probability_cache=probabilities,
        validation_masks=(validation_masks),
        labels=labels,
        class_names=class_names,
    )

    trial_frame = pd.DataFrame(agent.trials_as_dicts())

    for stream_index, stream_name in enumerate(stream_names):
        trial_frame[f"weight_{stream_name}"] = [
            trial.weights[stream_index] for trial in trials
        ]

    trial_frame = trial_frame.drop(columns=["weights"])

    method = args.fusion_method

    trials_path = INPUT_DIRECTORY / f"late_fusion_{method}_trials.csv"

    trial_frame.to_csv(
        trials_path,
        index=False,
    )

    decision_log_path = INPUT_DIRECTORY / f"{method}_decision_log.jsonl"

    with decision_log_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for trial in trials:
            handle.write(
                json.dumps(
                    asdict(trial),
                    sort_keys=True,
                )
                + "\n"
            )

    best_trial = agent.best_trial()

    stopping_reason = (
        "trial_budget"
        if len(trials) >= MAX_TRIALS
        else ("patience_or_no_untried_neighbors")
    )

    result_summary = {
        "agent": ("deterministic feedback-driven late-fusion search"),
        "classifier": ("PyTorch Linear"),
        "fusion_method": method,
        "max_trials": MAX_TRIALS,
        "completed_trials": len(trials),
        "patience": PATIENCE,
        "minimum_material_improvement": (MINIMUM_IMPROVEMENT),
        "stopping_reason": (stopping_reason),
        "stream_order": stream_names,
        "best_trial": asdict(best_trial),
        "validation_set_evaluated": (False),
        "test_set_evaluated": False,
    }

    best_path = INPUT_DIRECTORY / f"late_fusion_{method}_best.json"

    best_path.write_text(
        json.dumps(
            result_summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"=== Linear {method} fusion trials ===")

    display_columns = [
        "trial_index",
        "weight_metadata",
        "weight_uni2",
        "weight_virchow2",
        "weight_prism2",
        "macro_auroc_mean",
        "balanced_accuracy_mean",
        "selection_score",
        "improved_incumbent",
        "material_improvement",
    ]

    print(trial_frame[display_columns].to_string(index=False))

    print()
    print("=== Best late-fusion trial ===")

    print(
        json.dumps(
            asdict(best_trial),
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("Fixed validation was not evaluated.")

    print("Test set was not evaluated.")

    print(
        "Decision log:",
        decision_log_path,
    )


if __name__ == "__main__":
    main()
