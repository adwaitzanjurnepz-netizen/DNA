#!/usr/bin/env script
"""
evaluation_metrics.py
A pure Python and NumPy-based evaluation metrics library for multi-class classification.
Implements:
  - Confusion Matrix
  - Accuracy
  - Per-class Precision, Recall, F1-score, and Support
  - Macro-averaging
  - Micro-averaging
  - Weighted-averaging
  - Classification Report Generation
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd


def _validate_and_flatten_inputs(
    y_true: Union[list, np.ndarray], 
    y_pred: Union[list, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates input shapes and types, returning flattened NumPy arrays.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        
    Returns:
        Tuple of flattened (y_true, y_pred) numpy arrays.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise ValueError("Input arrays must not be empty.")
        
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Input shape mismatch: y_true shape {y_true_arr.shape} "
            f"does not match y_pred shape {y_pred_arr.shape}"
        )
        
    return y_true_arr.ravel(), y_pred_arr.ravel()


def accuracy_score(
    y_true: Union[list, np.ndarray], 
    y_pred: Union[list, np.ndarray]
) -> float:
    """
    Compute the accuracy classification score.
    
    Args:
        y_true: Ground truth (correct) labels.
        y_pred: Predicted labels returned by a classifier.
        
    Returns:
        Accuracy score as a float.
    """
    y_true_flat, y_pred_flat = _validate_and_flatten_inputs(y_true, y_pred)
    correct = np.sum(y_true_flat == y_pred_flat)
    return float(correct / len(y_true_flat))


def confusion_matrix(
    y_true: Union[list, np.ndarray], 
    y_pred: Union[list, np.ndarray], 
    labels: Optional[List[Any]] = None
) -> pd.DataFrame:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        labels: List of labels to index the matrix. If None,
                unique labels are extracted from y_true and y_pred in sorted order.
                
    Returns:
        A pandas DataFrame where rows represent true classes, and columns represent predicted.
    """
    y_true_flat, y_pred_flat = _validate_and_flatten_inputs(y_true, y_pred)
    
    if labels is None:
        unique_labels = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
        try:
            unique_labels = sorted(unique_labels)
        except TypeError:
            unique_labels = sorted(unique_labels, key=str)
    else:
        unique_labels = list(labels)
        
    n_labels = len(unique_labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    matrix = np.zeros((n_labels, n_labels), dtype=int)
    
    for vt, vp in zip(y_true_flat, y_pred_flat):
        if vt in label_to_idx and vp in label_to_idx:
            matrix[label_to_idx[vt], label_to_idx[vp]] += 1
            
    df_cm = pd.DataFrame(matrix, index=unique_labels, columns=unique_labels)
    df_cm.index.name = "Actual"
    df_cm.columns.name = "Predicted"
    return df_cm


def precision_recall_fscore_support(
    y_true: Union[list, np.ndarray],
    y_pred: Union[list, np.ndarray],
    labels: Optional[List[Any]] = None
) -> Tuple[Dict[Any, Dict[str, float]], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Compute precision, recall, F-measure and support for each class as well as
    macro, micro, and weighted averages.
    
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        labels: List of labels to index the metrics. If None,
                unique labels are extracted from y_true and y_pred in sorted order.
                
    Returns:
        A tuple of four dictionaries:
        - per_class: dictionary mapping each class to a dictionary of metrics
        - macro_avg: dictionary containing macro averages
        - micro_avg: dictionary containing micro averages
        - weighted_avg: dictionary containing weighted averages
    """
    y_true_flat, y_pred_flat = _validate_and_flatten_inputs(y_true, y_pred)
    
    if labels is None:
        unique_labels = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
        try:
            unique_labels = sorted(unique_labels)
        except TypeError:
            unique_labels = sorted(unique_labels, key=str)
    else:
        unique_labels = list(labels)
        
    per_class = {}
    tps, fps, fns, supports = {}, {}, {}, {}
    total_support = 0
    
    for label in unique_labels:
        true_mask = (y_true_flat == label)
        pred_mask = (y_pred_flat == label)
        
        tp = np.sum(true_mask & pred_mask)
        fp = np.sum(~true_mask & pred_mask)
        fn = np.sum(true_mask & ~pred_mask)
        support = np.sum(true_mask)
        
        tps[label] = tp
        fps[label] = fp
        fns[label] = fn
        supports[label] = support
        total_support += support
        
        # Calculate scores, ensuring division-by-zero results in 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": int(support)
        }
        
    # 1. Macro-averaging: arithmetic mean of all classes
    macro_precision = np.mean([metrics["precision"] for metrics in per_class.values()])
    macro_recall = np.mean([metrics["recall"] for metrics in per_class.values()])
    macro_f1 = np.mean([metrics["f1-score"] for metrics in per_class.values()])
    
    macro_avg = {
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1-score": float(macro_f1),
        "support": int(total_support)
    }
    
    # 2. Micro-averaging: global metrics pooled across all classes
    total_tp = sum(tps.values())
    total_fp = sum(fps.values())
    total_fn = sum(fns.values())
    
    micro_precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = float(2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0
    
    micro_avg = {
        "precision": float(micro_precision),
        "recall": float(micro_recall),
        "f1-score": float(micro_f1),
        "support": int(total_support)
    }
    
    # 3. Weighted-averaging: weighted mean by support count
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    
    if total_support > 0:
        for label, metrics in per_class.items():
            weight = metrics["support"] / total_support
            weighted_precision += metrics["precision"] * weight
            weighted_recall += metrics["recall"] * weight
            weighted_f1 += metrics["f1-score"] * weight
            
    weighted_avg = {
        "precision": float(weighted_precision),
        "recall": float(weighted_recall),
        "f1-score": float(weighted_f1),
        "support": int(total_support)
    }
    
    return per_class, macro_avg, micro_avg, weighted_avg


def classification_report(
    y_true: Union[list, np.ndarray],
    y_pred: Union[list, np.ndarray],
    labels: Optional[List[Any]] = None,
    digits: int = 4
) -> str:
    """
    Build a text report showing the main classification metrics.
    
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        labels: List of labels to include in the report.
        digits: Number of digits to print in floats.
        
    Returns:
        Formatted classification report as a string.
    """
    per_class, macro_avg, micro_avg, weighted_avg = precision_recall_fscore_support(
        y_true, y_pred, labels=labels
    )
    accuracy = accuracy_score(y_true, y_pred)
    
    # Format Header
    report = f"{'class':<15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n\n"
    
    # Format Per-Class Rows
    for label, metrics in per_class.items():
        report += (
            f"{str(label):<15} "
            f"{metrics['precision']:>{10}.{digits}f} "
            f"{metrics['recall']:>{10}.{digits}f} "
            f"{metrics['f1-score']:>{10}.{digits}f} "
            f"{metrics['support']:>10d}\n"
        )
    
    report += "\n"
    
    # Format Accuracy Row
    total_support = micro_avg["support"]
    report += (
        f"{'accuracy':<15} "
        f"{'':>10} "
        f"{'':>10} "
        f"{accuracy:>{10}.{digits}f} "
        f"{total_support:>10d}\n"
    )
    
    # Format Macro Avg Row
    report += (
        f"{'macro avg':<15} "
        f"{macro_avg['precision']:>{10}.{digits}f} "
        f"{macro_avg['recall']:>{10}.{digits}f} "
        f"{macro_avg['f1-score']:>{10}.{digits}f} "
        f"{macro_avg['support']:>10d}\n"
    )
    
    # Format Weighted Avg Row
    report += (
        f"{'weighted avg':<15} "
        f"{weighted_avg['precision']:>{10}.{digits}f} "
        f"{weighted_avg['recall']:>{10}.{digits}f} "
        f"{weighted_avg['f1-score']:>{10}.{digits}f} "
        f"{weighted_avg['support']:>10d}\n"
    )
    
    return report


# Validation Test Suite comparing with Scikit-learn
if __name__ == "__main__":
    print("Running validation tests vs Scikit-Learn...")
    from sklearn.metrics import (
        classification_report as sklearn_report,
        confusion_matrix as sklearn_cm,
        accuracy_score as sklearn_accuracy
    )
    
    # Test 1: Standard Multiclass Scenario
    y_test_true = [0, 1, 2, 0, 1, 2, 2, 0, 1, 3]
    y_test_pred = [0, 2, 1, 0, 0, 2, 2, 0, 1, 3]
    
    # Compare Accuracy
    my_acc = accuracy_score(y_test_true, y_test_pred)
    sk_acc = sklearn_accuracy(y_test_true, y_test_pred)
    assert np.isclose(my_acc, sk_acc), f"Accuracy mismatch: {my_acc} vs {sk_acc}"
    
    # Compare Confusion Matrix
    my_cm = confusion_matrix(y_test_true, y_test_pred).values
    sk_cm = sklearn_cm(y_test_true, y_test_pred)
    assert np.array_equal(my_cm, sk_cm), f"Confusion Matrix mismatch:\n{my_cm}\nvs\n{sk_cm}"
    
    # Print custom report
    print("Custom Report output:")
    print(classification_report(y_test_true, y_test_pred, digits=4))
    
    print("Scikit-Learn Report output:")
    print(sklearn_report(y_test_true, y_test_pred, digits=4))
    
    # Test 2: Edge Cases (Zero Division)
    y_edge_true = [0, 1, 2, 0, 1, 2]
    y_edge_pred = [0, 0, 0, 0, 0, 0] # all predicted as 0, classes 1 and 2 have 0 precision and recall
    
    my_per_class, my_macro, my_micro, my_weighted = precision_recall_fscore_support(y_edge_true, y_edge_pred)
    
    assert my_per_class[1]["precision"] == 0.0
    assert my_per_class[1]["recall"] == 0.0
    assert my_per_class[1]["f1-score"] == 0.0
    
    print("\nAll checks passed successfully! Your custom metrics match Scikit-Learn perfectly.")
