import pandas as pd
import numpy as np
from prts import ts_precision, ts_recall
from utils import set_index

def calculate_metrics(predictions_path, ground_truth_path, al=0, cardinality='reciprocal', bias='front'):
    # Read the files
    predictions = pd.read_csv(predictions_path)
    ground_truth = pd.read_csv(ground_truth_path)
    
    # Ensure both dataframes have same index
    predictions = set_index(predictions)
    ground_truth = set_index(ground_truth)
    
    # Initialize result containers
    metrics = {
        'Rr': [],  # Recall
        'Pr': [],  # Precision
        'F1r': [], # F1 score
        'TP': [],  # True Positives
        'FP': [],  # False Positives
        'TN': [],  # True Negatives
        'FN': []   # False Negatives
    }
    
    # Calculate metrics for each column
    for col in predictions.columns:
        if col in ground_truth.columns:
            y_true = ground_truth[col].values
            y_pred = predictions[col].values
            
            # Calculate basic counts
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            if np.allclose(np.unique(y_pred), np.array([0, 1])) or np.allclose(np.unique(y_pred), np.array([1])):
                # Calculate custom metrics
                pre_value = ts_precision(y_true, y_pred, al, cardinality, bias)
                rec_value = ts_recall(y_true, y_pred, al, cardinality, bias)
                f1_value = 2*(pre_value*rec_value)/(pre_value+rec_value+1e-6)
            else:
                pre_value = 0
                rec_value = 0
                f1_value = 0
            
            # Store results
            metrics['Rr'].append(rec_value * 100)
            metrics['Pr'].append(pre_value * 100)
            metrics['F1r'].append(f1_value * 100)
            metrics['TP'].append(tp)
            metrics['FP'].append(fp)
            metrics['TN'].append(tn)
            metrics['FN'].append(fn)
    
    # Create results DataFrame
    results_df = pd.DataFrame(metrics)
    print("\nMetrics for each time series:")
    print(results_df)
    
    # Calculate mean scores
    print("\nMean scores:")
    print(f"Average Recall (Rr): {np.mean(metrics['Rr']):.2f}%")
    print(f"Average Precision (Pr): {np.mean(metrics['Pr']):.2f}%")
    print(f"Average F1-score (F1r): {np.mean(metrics['F1r']):.2f}%")
    print("\nTotal counts:")
    print(f"Total TP: {np.sum(metrics['TP'])}")
    print(f"Total FP: {np.sum(metrics['FP'])}")
    print(f"Total TN: {np.sum(metrics['TN'])}")
    print(f"Total FN: {np.sum(metrics['FN'])}")
    
    return results_df

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python evaluate_metrics.py <predictions_csv> <ground_truth_csv>")
        sys.exit(1)
        
    predictions_path = sys.argv[1]
    ground_truth_path = sys.argv[2]
    
    results = calculate_metrics(predictions_path, ground_truth_path)
    # Save results
    results.to_csv('evaluation_metrics.csv')