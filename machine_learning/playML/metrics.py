import numpy as np


def accuracy_score(y_true, y_predict):
    """
    计算分类准确度
    y_true: 真值
    y_predict: 预测值
    """
    assert y_true.shape[0] == y_predict.shape[0], '实际值和预测值长度必须相等'
    return np.sum(y_true == y_predict)/len(y_true)


def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的MSE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum((y_true - y_predict)**2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的RMSE"""

    return np.sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """计算y_true和y_predict之间的MAE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"
    
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """计算y_true和y_predict之间的R Square"""

    return 1 - mean_squared_error(y_true, y_predict)/np.var(y_true)