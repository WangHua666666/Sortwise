import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    """
    Calculate various metrics for model evaluation
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary containing various metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics

def calculate_per_class_metrics(y_true, y_pred, class_names):
    """
    Calculate per-class metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        dict: Dictionary containing per-class metrics
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, average=None),
        'recall': recall_score(y_true, y_pred, average=None),
        'f1': f1_score(y_true, y_pred, average=None)
    }
    
    # Create per-class metrics dictionary
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': metrics['precision'][i],
            'recall': metrics['recall'][i],
            'f1': metrics['f1'][i]
        }
    
    return per_class_metrics 