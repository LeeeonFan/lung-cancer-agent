import numpy as np

from lung_fusion_agent.fusion.gated import (
    MetadataGatedLogitFusion,
)


def test_zero_coefficients_reproduce_prior_weights() -> None:
    model = MetadataGatedLogitFusion(
        prior_weights=np.asarray([0.05, 0.25, 0.05, 0.65]),
        regularization=1.0,
        class_weight_balanced=False,
    )

    model.metadata_mean_ = np.asarray([50.0, 0.5])
    model.metadata_scale_ = np.asarray([10.0, 0.5])
    model.coefficients_ = np.zeros((3, 4))

    metadata = np.asarray(
        [
            [40.0, 0.0],
            [60.0, 1.0],
        ]
    )

    gates = model.predict_gates(metadata)

    expected = np.tile(
        np.asarray([0.05, 0.25, 0.05, 0.65]),
        (2, 1),
    )

    np.testing.assert_allclose(
        gates,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_gated_predictions_are_valid_probabilities() -> None:
    model = MetadataGatedLogitFusion(
        prior_weights=np.asarray([0.05, 0.25, 0.05, 0.65]),
        regularization=10.0,
        class_weight_balanced=False,
    )

    metadata = np.asarray(
        [
            [45.0, 0.0],
            [60.0, 1.0],
            [55.0, 0.0],
            [70.0, 1.0],
        ]
    )

    probabilities = np.asarray(
        [
            [
                [0.6, 0.4],
                [0.7, 0.3],
                [0.5, 0.5],
                [0.8, 0.2],
            ],
            [
                [0.4, 0.6],
                [0.3, 0.7],
                [0.5, 0.5],
                [0.2, 0.8],
            ],
            [
                [0.55, 0.45],
                [0.65, 0.35],
                [0.45, 0.55],
                [0.75, 0.25],
            ],
            [
                [0.45, 0.55],
                [0.35, 0.65],
                [0.55, 0.45],
                [0.25, 0.75],
            ],
        ]
    )

    labels = np.asarray([0, 1, 0, 1])

    model.fit(
        metadata=metadata,
        probabilities=probabilities,
        labels=labels,
    )

    fused = model.predict_proba(
        metadata=metadata,
        probabilities=probabilities,
    )

    assert fused.shape == (
        4,
        2,
    )

    assert np.isfinite(fused).all()

    np.testing.assert_allclose(
        fused.sum(axis=1),
        np.ones(4),
        rtol=1e-6,
        atol=1e-6,
    )
