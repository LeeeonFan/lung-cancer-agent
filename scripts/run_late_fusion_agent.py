import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from lung_fusion_agent.agent import (
    LateFusionSearchAgent,
)

CACHE_PATH = Path("artifacts/results/fusion_agent/oof_probabilities.npz")

STREAM_SUMMARY_PATH = Path("artifacts/results/fusion_agent/oof_stream_summary.csv")

OUTPUT_DIR = Path("artifacts/results/fusion_agent")

MAX_TRIALS = 8
PATIENCE = 4
MINIMUM_IMPROVEMENT = 0.001


def main() -> None:
    with np.load(
        CACHE_PATH,
        allow_pickle=False,
    ) as cache:
        labels = cache["labels"].astype(np.int64)

        class_names = cache["class_names"].astype(str).tolist()

        stream_names = cache["stream_names"].astype(str).tolist()

        validation_masks = cache["validation_masks"].astype(bool)

        probabilities = cache["probabilities"].astype(np.float32)

    stream_summary = pd.read_csv(STREAM_SUMMARY_PATH)

    standalone_scores = {
        str(row["stream"]): float(row["composite_mean"])
        for _, row in (stream_summary.iterrows())
    }

    agent = LateFusionSearchAgent(
        stream_names=stream_names,
        standalone_scores=(standalone_scores),
        max_trials=MAX_TRIALS,
        patience=PATIENCE,
        minimum_improvement=(MINIMUM_IMPROVEMENT),
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

    trial_frame.to_csv(
        OUTPUT_DIR / "late_fusion_trials.csv",
        index=False,
    )

    decision_log_path = OUTPUT_DIR / "decision_log.jsonl"

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
        else "patience_or_no_untried_neighbors"
    )

    result_summary = {
        "agent": ("deterministic feedback-driven late-fusion search"),
        "max_trials": MAX_TRIALS,
        "completed_trials": len(trials),
        "patience": PATIENCE,
        "minimum_material_improvement": (MINIMUM_IMPROVEMENT),
        "stopping_reason": (stopping_reason),
        "stream_order": stream_names,
        "best_trial": asdict(best_trial),
        "validation_set_evaluated": False,
        "test_set_evaluated": False,
    }

    (OUTPUT_DIR / "late_fusion_best.json").write_text(
        json.dumps(
            result_summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=== Agent trial table ===")

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
    print("Validation set was not evaluated.")
    print("Test set was not evaluated.")
    print(
        "Decision log:",
        decision_log_path,
    )


if __name__ == "__main__":
    main()
