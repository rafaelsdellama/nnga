import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)


def cross_validation_statistics(evaluate_results):
    """
    Parameters
    ----------
        evaluate_results : list of list
            List of list with cross validation
            loss value & metrics values for each fold
    Returns
    -------
        Pandas DataFrame with cross validation results,
        and max, min, mean and std metrics

    """
    df = pd.DataFrame(evaluate_results, columns=["loss", "acc"])

    mean_result = df.mean()
    min_result = df.min()
    max_result = df.max()
    std_result = df.std()
    df.loc["mean"] = mean_result
    df.loc["min"] = min_result
    df.loc["max"] = max_result
    df.loc["std"] = std_result

    return df


def compute_metrics(lbl, predict, predict_encoded, target_names, idx):
    """
    Compute metris to evaluate the model
    Parameters
    ----------
        lbl : list
            List of true label index
        predict : list
            List of predict label index
        predict_encoded: list
            List of predict proba label
        target_names: list
            List of class target
        idx: List
            List of idx of samples
    Returns
    -------
        Dict with all metrics
            - accuracy_score
            - balanced_accuracy_score
            - confusion_matrix
            - classification_report
            - predict_table
            - medical_metrics
    """
    metrics = dict()
    metrics["accuracy_score"] = accuracy_score(lbl, predict)
    metrics["balanced_accuracy_score"] = balanced_accuracy_score(lbl, predict)
    metrics["confusion_matrix"] = confusion_matrix(lbl, predict)
    metrics["classification_report"] = classification_report(
        lbl, predict, target_names=target_names
    )

    predict_table = {
        "label": idx,
        "y_test": [target_names[i] for i in lbl],
        "predict": [target_names[i] for i in predict],
    }

    for i in range(len(target_names)):
        predict_table[str("percentage " + target_names[i])] = predict_encoded[
            :, i
        ]

    # Medical metrics
    FP = metrics["confusion_matrix"].sum(axis=0) - np.diag(
        metrics["confusion_matrix"]
    )
    FN = metrics["confusion_matrix"].sum(axis=1) - np.diag(
        metrics["confusion_matrix"]
    )
    TP = np.diag(metrics["confusion_matrix"])
    TN = np.sum(metrics["confusion_matrix"]) - (FP + FN + TP)

    np.seterr(divide="ignore", invalid="ignore")
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = np.divide(TP, TP + FN)
    # Specificity or true negative rate
    TNR = np.divide(TN, TN + FP)
    # Precision or positive predictive value
    PPV = np.divide(TP, TP + FP)
    # Negative predictive value
    NPV = np.divide(TN, TN + FN)
    # Fall out or false positive rate
    FPR = np.divide(FP, FP + TN)
    # False negative rate
    FNR = np.divide(FN, TP + FN)
    # False discovery rate
    FDR = np.divide(FP, TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    # Medical metrics
    medical_metrics = []
    for i in range(metrics["confusion_matrix"].shape[0]):
        medical_metrics.append(
            {
                "Class": target_names[i],
                "TP": TP[i],
                "TN": TN[i],
                "FP": FP[i],
                "FN": FN[i],
                "Specificity": TNR[i],
                "Sensitivity": TPR[i],
                "Positive Predictive Value": PPV[i],
                "Negative Predictive Value": NPV[i],
                "False Positive Rate": FPR[i],
                "False Negative Rate": FNR[i],
                "False Discovery Rate": FDR[i],
                "Accuracy": ACC[i],
            }
        )

        metrics["predict_table"] = pd.DataFrame(predict_table)
        metrics["medical_metrics"] = pd.DataFrame(medical_metrics)

        metrics["accuracy_score"] = str(metrics["accuracy_score"])
        metrics["balanced_accuracy_score"] = str(
            metrics["balanced_accuracy_score"]
        )
        metrics["confusion_matrix"] = str(metrics["confusion_matrix"])
        metrics["classification_report"] = str(
            metrics["classification_report"]
        )
        metrics["medical_metrics"] = metrics["medical_metrics"].to_string(
            index=False
        )
        metrics["predict_table"] = metrics["predict_table"].to_string(
            index=False
        )

    return metrics
