import numpy as np

def accuracy_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], '实际值和预测值长度必须相等'
    return np.sum(y_true == y_predict)/len(y_true)