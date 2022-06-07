import numpy as np
import pandas as pd

# took from: https://www.kaggle.com/code/rohanrao/amex-competition-metric-implementations

def compute_recall_at4(y_true: np.array, y_pred: np.array) -> float:
    
    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos
    
    # desc sorting by prediction values
    indices = np.argsort(y_pred)[::-1]
    target = y_true[indices]
    
    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    
    # default rate captured at 4%
    d = target[four_pct_mask].sum() / n_pos
    
    return d

def compute_normalized_gini(y_true: np.array, y_pred: np.array) -> float:
    
    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting desc by prediction values
    indices = np.argsort(y_pred)[::-1]
    target = y_true[indices]

    # weighted gini coefficient
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()

    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max
    
    return g
    
def compute_amex_metric(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting desc by prediction values
    indices = np.argsort(y_pred)[::-1]
    target = y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)