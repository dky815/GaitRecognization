import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def plot_AUC(df):
    labels = df['label']
    scores = df['score']
    labels = [int(e)   for e in labels]
    scores = [float(e) for e in scores]
    auc_value = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='black', lw=lw, label='AUC = %0.4f, EER= %0.4f' % (auc_value, eer))
    plt.plot([0, 1], [0, 1], color='darkorange', lw=lw, linestyle='--')
    plt.xlim([-0.005, 1.0])
    plt.ylim([-0.005, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")
    plt.show()
    return