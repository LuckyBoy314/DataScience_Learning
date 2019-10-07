import numpy as np
from collections import Counter

def KNN_classify(k, X_train, y_train, x):

    # 计算x与X_train中各个样本的距离，距离采用欧氏距离
    # 注意广播计算的计算方法
    distances = np.sqrt(np.sum(np.square(X_train-x),axis=1))

    # 计算与x距离最近的前K样本的标签
    # 注意argsort排序后返回的不是值，而是值在原来数组中的索引
    topK_y = y_train[np.argsort(distances)][:k]

    # 计算前K样本标签中最多的标签
    # 注意collections.Counter的用法
    votes = Counter(topK_y)
    predict_y = votes.most_common(1)[0][0]

    return predict_y

# 模拟Scikit-learn中的KNN算法
class KNN_classifier:
    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None
        
    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        
        return self
    
    def _pre(self,x):
        
        # 计算x与X_train中各个样本的距离，距离采用欧氏距离
        # 注意广播计算的计算方法
        distances = np.sqrt(np.sum(np.square(self._X_train-x),axis=1))

        # 计算与x距离最近的前K样本的标签
        # 注意argsort排序后返回的不是值，而是值在原来数组中的索引
        topK_y = self._y_train[np.argsort(distances)][:self.k]

        # 计算前K样本标签中最多的标签
        # 注意collections.Counter的用法
        votes = Counter(topK_y)
        predict_y = votes.most_common(1)[0][0]

        return predict_y

    def predict(self, x_predict): # 注意x_predict是二维矩阵
        y_predict = [self._pre(x) for x in x_predict]
        return np.array(y_predict)
        
    def __repr__(self):
        return 'KNN(k=%d)' % self.k 