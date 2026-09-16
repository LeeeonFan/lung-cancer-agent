import numpy as np

from lung_fusion_agent.fusion.late import (
    fuse_log_probabilities,
    fuse_probabilities,
)


def test_equal_weight_probability_fusion() -> None:
    probabilities = np.asarray(
        [
            [
                [0.8, 0.2],
                [0.4, 0.6],
            ],
            [
                [0.1, 0.9],
                [0.3, 0.7],
            ],
        ],
        dtype=np.float32,
    )

    fused = fuse_probabilities(
        probabilities,
        np.asarray([0.5, 0.5]),
    )

    expected = np.asarray(
        [
            [0.6, 0.4],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        fused,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_weights_are_automatically_normalized() -> None:
    probabilities = np.asarray(
        [
            [
                [0.8, 0.2],
                [0.4, 0.6],
            ],
        ],
        dtype=np.float32,
    )

    fused_a = fuse_probabilities(
        probabilities,
        np.asarray([0.75, 0.25]),
    )

    fused_b = fuse_probabilities(
        probabilities,
        np.asarray([3.0, 1.0]),
    )

    np.testing.assert_allclose(
        fused_a,
        fused_b,
        rtol=1e-6,
        atol=1e-6,
    )

def test_equal_predictions_remain_unchanged_with_logit_fusion() -> None:
    probabilities = np.asarray(
        [
            [
                [0.7, 0.2, 0.1],
                [0.7, 0.2, 0.1],
            ],
        ],
        dtype=np.float32,
    )

    fused = fuse_log_probabilities(
        probabilities,
        np.asarray(
            [0.5, 0.5]
        ),
    )

    np.testing.assert_allclose(
        fused,
        probabilities[:, 0, :],
        rtol=1e-6,
        atol=1e-6,
    )


def test_logit_fusion_returns_valid_probabilities() -> None:
    probabilities = np.asarray(
        [
            [
                [0.8, 0.2],
                [0.4, 0.6],
            ],
            [
                [0.1, 0.9],
                [0.3, 0.7],
            ],
        ],
        dtype=np.float32,
    )

    fused = fuse_log_probabilities(
        probabilities,
        np.asarray(
            [0.5, 0.5]
        ),
    )

    assert np.isfinite(
        fused
    ).all()

    np.testing.assert_allclose(
        fused.sum(axis=1),
        np.ones(2),
        rtol=1e-6,
        atol=1e-6,
    )