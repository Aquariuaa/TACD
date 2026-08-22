from castle.common import GraphDAG
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import roc_auc_score
from castle.metrics import MetricsDAG


def metric_dag_res(est_causal_matrix, dag_matrix):
    auto_result = MetricsDAG(est_causal_matrix, dag_matrix).metrics
    TP = []
    FP = []
    FN = []
    TN = []
    for i in range(len(est_causal_matrix)):
        for j in range(len(est_causal_matrix)):
            if est_causal_matrix[i][j]==1 and dag_matrix[i][j]==1:
                TP.append((i, j))
            if est_causal_matrix[i][j]==1 and dag_matrix[i][j]==0:
                FP.append((i, j))
            if est_causal_matrix[i][j]==0 and dag_matrix[i][j]==1:
                FN.append((i, j))
            if est_causal_matrix[i][j]==0 and dag_matrix[i][j]==0:
                TN.append((i, j))
    TP, FP, TN, FN = len(TP), len(FP), len(TN), len(FN)
    return TP, FN, TN, FP, auto_result['precision'], auto_result['recall'], auto_result['F1'], auto_result['gscore'], auto_result['shd']


def find_optimal_threshold_and_matrix_curve(prob_matrix, true_matrix, metric='f1'):
    y_true = true_matrix.flatten()
    y_prob = prob_matrix.flatten()

    n = prob_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool).flatten()
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    # print("y_true", y_true)
    # print("y_prob", y_prob)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_scores = f1_scores[:-1]

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    est_causal_matrix = (prob_matrix > optimal_threshold).astype(int)

    return optimal_threshold, est_causal_matrix