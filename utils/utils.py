import numpy as np
from sklearn import metrics

def write_to_tb(writer, labels, scalars, iteration, phase='train'):
    for scalar, label in zip(scalars, labels):
        writer.add_scalar(f'{phase}/{label}', scalar, iteration)

def calculate_metrics_sigmoid(y_true, y_pred):

    if type(y_true) == list:
        y_true = np.concatenate(y_true, axis=0)
    if type(y_pred) == list:
        y_pred = np.concatenate(y_pred, axis=0)

    roc_auc = metrics.roc_auc_score(y_true.ravel(), y_pred.ravel())
    ap = metrics.average_precision_score(y_true.ravel(), y_pred.ravel())

    dict_metrics = {'ROC AUC': roc_auc, 'AP': ap} #, 'mF1': mf1, 'MF1': Mf1}

    return dict_metrics





