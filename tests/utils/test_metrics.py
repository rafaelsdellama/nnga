from nnga.utils.metrics import cross_validation_statistics, compute_metrics
import pandas as pd
import numpy as np


def test_compute_metrics():
    metrics = compute_metrics(
        [0, 0, 1],
        [1, 0, 0],
        np.array(
            [np.array([0.1, 0.9]), np.array([0.8, 0.2]), np.array([0.7, 0.3])]
        ),
        ["Class 0", "Class 1"],
        ["0", "1", "2"],
    )

    assert type(metrics) == dict
    assert all(
        [
            a == b
            for a, b in zip(
                metrics.keys(),
                [
                    "accuracy_score",
                    "balanced_accuracy_score",
                    "confusion_matrix",
                    "classification_report",
                    "predict_table",
                    "medical_metrics",
                ],
            )
        ]
    )


def test_cross_validation_statistics():
    evaluate_results = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
    cv_statistic = cross_validation_statistics(evaluate_results)
    assert type(cv_statistic) == pd.DataFrame
    assert len(cv_statistic) == len(evaluate_results) + 4
    assert all(
        [a == b for a, b in zip(cv_statistic.columns.values, ["loss", "acc"])]
    )
