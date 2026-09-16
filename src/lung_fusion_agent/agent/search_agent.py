from dataclasses import asdict, dataclass

import numpy as np

from lung_fusion_agent.fusion.late import (
    evaluate_late_fusion,
)

MIN_IMAGE_WEIGHT = 0.05
MIN_METADATA_WEIGHT = 0.025
NEIGHBOR_STEP = 0.05


@dataclass(frozen=True)
class FusionProposal:
    weights: tuple[float, ...]
    rationale: str


@dataclass(frozen=True)
class FusionTrial:
    trial_index: int
    weights: tuple[float, ...]
    rationale: str
    macro_auroc_mean: float
    macro_auroc_std: float
    balanced_accuracy_mean: float
    balanced_accuracy_std: float
    composite_mean: float
    composite_sem: float
    selection_score: float
    improved_incumbent: bool
    material_improvement: bool


class LateFusionSearchAgent:
    def __init__(
        self,
        *,
        stream_names: list[str],
        standalone_scores: dict[str, float],
        max_trials: int = 8,
        patience: int = 4,
        minimum_improvement: float = 0.001,
        fusion_method: str = "probability",
    ) -> None:
        if stream_names != [
            "metadata",
            "uni2",
            "virchow2",
            "prism2",
        ]:
            raise ValueError("Unexpected stream order.")

        if fusion_method not in {
            "probability",
            "logit",
        }:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.stream_names = stream_names
        self.standalone_scores = standalone_scores
        self.max_trials = max_trials
        self.patience = patience
        self.minimum_improvement = minimum_improvement
        self.fusion_method = fusion_method

        self.trials: list[FusionTrial] = []
        self._attempted_weights: set[tuple[float, ...]] = set()

        self._best_score = -np.inf
        self._best_weights: tuple[float, ...] | None = None
        self._trials_without_material_gain = 0

    @staticmethod
    def _canonicalize_weights(
        weights: np.ndarray,
    ) -> tuple[float, ...]:
        normalized = weights / weights.sum()

        return tuple(round(float(value), 6) for value in normalized)

    def _initial_proposal(
        self,
    ) -> FusionProposal:
        weights = self._canonicalize_weights(
            np.asarray(
                [
                    0.05,
                    0.3166667,
                    0.3166667,
                    0.3166666,
                ]
            )
        )

        return FusionProposal(
            weights=weights,
            rationale=(
                "Begin with nearly equal weights "
                "across the three image streams "
                "and a small non-zero metadata "
                "contribution."
            ),
        )

    def _second_proposal(
        self,
    ) -> FusionProposal:
        weights = self._canonicalize_weights(
            np.asarray(
                [
                    0.05,
                    0.25,
                    0.15,
                    0.55,
                ]
            )
        )

        return FusionProposal(
            weights=weights,
            rationale=(
                "The standalone CV ranking places "
                "Prism2 first, UNI2 second and "
                "Virchow2 third, so test a "
                "Prism2-heavy fusion while "
                "retaining all streams."
            ),
        )

    def _generate_neighbors(
        self,
        base_weights: tuple[float, ...],
    ) -> list[tuple[float, ...]]:
        base = np.asarray(
            base_weights,
            dtype=np.float64,
        )

        candidates = []

        for donor_index in range(len(base)):
            for receiver_index in range(len(base)):
                if donor_index == receiver_index:
                    continue

                candidate = base.copy()

                candidate[donor_index] -= NEIGHBOR_STEP
                candidate[receiver_index] += NEIGHBOR_STEP

                if candidate[0] < MIN_METADATA_WEIGHT:
                    continue

                if (candidate[1:] < MIN_IMAGE_WEIGHT).any():
                    continue

                canonical = self._canonicalize_weights(candidate)

                if canonical in self._attempted_weights:
                    continue

                candidates.append(canonical)

        return candidates

    def _heuristic_score(
        self,
        weights: tuple[float, ...],
    ) -> float:
        return float(
            sum(
                weight * self.standalone_scores[stream_name]
                for weight, stream_name in zip(
                    weights,
                    self.stream_names,
                    strict=True,
                )
            )
        )

    def propose(
        self,
    ) -> FusionProposal | None:
        if len(self.trials) >= self.max_trials:
            return None

        if self._trials_without_material_gain >= self.patience:
            return None

        if not self.trials:
            return self._initial_proposal()

        if len(self.trials) == 1:
            return self._second_proposal()

        if self._best_weights is None:
            raise RuntimeError("Agent has no incumbent weights.")

        candidates = self._generate_neighbors(self._best_weights)

        if not candidates:
            return None

        candidate = max(
            candidates,
            key=self._heuristic_score,
        )

        best_trial_number = (
            max(
                self.trials,
                key=lambda trial: trial.selection_score,
            ).trial_index
            + 1
        )

        rationale = (
            "Generate a local weight-transfer "
            "neighbor around the current best "
            f"trial {best_trial_number}. "
            "Among untried neighbors, prefer the "
            "candidate supported by standalone "
            "train-CV stream scores."
        )

        return FusionProposal(
            weights=candidate,
            rationale=rationale,
        )

    def observe(
        self,
        *,
        proposal: FusionProposal,
        result: object,
    ) -> FusionTrial:
        weights = self._canonicalize_weights(np.asarray(proposal.weights))

        if weights in self._attempted_weights:
            raise ValueError("Agent proposed duplicate weights.")

        previous_best = self._best_score

        improved_incumbent = result.selection_score > previous_best

        material_improvement = result.selection_score >= (
            previous_best + self.minimum_improvement
        )

        if improved_incumbent:
            self._best_score = result.selection_score
            self._best_weights = weights

        if material_improvement:
            self._trials_without_material_gain = 0
        else:
            self._trials_without_material_gain += 1

        self._attempted_weights.add(weights)

        trial = FusionTrial(
            trial_index=len(self.trials),
            weights=weights,
            rationale=proposal.rationale,
            macro_auroc_mean=(result.macro_auroc_mean),
            macro_auroc_std=(result.macro_auroc_std),
            balanced_accuracy_mean=(result.balanced_accuracy_mean),
            balanced_accuracy_std=(result.balanced_accuracy_std),
            composite_mean=(result.composite_mean),
            composite_sem=(result.composite_sem),
            selection_score=(result.selection_score),
            improved_incumbent=(improved_incumbent),
            material_improvement=(material_improvement),
        )

        self.trials.append(trial)

        return trial

    def run(
        self,
        *,
        probability_cache: np.ndarray,
        validation_masks: np.ndarray,
        labels: np.ndarray,
        class_names: list[str],
    ) -> list[FusionTrial]:
        while True:
            proposal = self.propose()

            if proposal is None:
                break

            print()
            print("=" * 70)
            print(f"Agent trial {len(self.trials) + 1}/{self.max_trials}")
            print(
                "Weights:",
                proposal.weights,
            )
            print(
                "Reason:",
                proposal.rationale,
            )

            result = evaluate_late_fusion(
                probability_cache=probability_cache,
                validation_masks=validation_masks,
                labels=labels,
                class_names=class_names,
                weights=np.asarray(proposal.weights),
                fusion_method=(self.fusion_method),
            )

            trial = self.observe(
                proposal=proposal,
                result=result,
            )

            print(
                "Macro AUROC:",
                round(
                    trial.macro_auroc_mean,
                    6,
                ),
            )
            print(
                "Balanced accuracy:",
                round(
                    trial.balanced_accuracy_mean,
                    6,
                ),
            )
            print(
                "Selection score:",
                round(
                    trial.selection_score,
                    6,
                ),
            )
            print(
                "New incumbent:",
                trial.improved_incumbent,
            )
            print(
                "Material improvement:",
                trial.material_improvement,
            )

        return self.trials

    def best_trial(
        self,
    ) -> FusionTrial:
        if not self.trials:
            raise RuntimeError("Agent has no completed trials.")

        return max(
            self.trials,
            key=lambda trial: trial.selection_score,
        )

    def trials_as_dicts(
        self,
    ) -> list[dict[str, object]]:
        return [asdict(trial) for trial in self.trials]
