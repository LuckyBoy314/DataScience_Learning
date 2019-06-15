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
        """初始化kNN分类器"""
        
        assert k >= 1, "k must be valid"
        
        self.k = k
        self._X_train = None
        self._y_train = None
        
    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."
        
        self._X_train = X_train
        self._y_train = y_train
        
        return self
    
    def _pre(self,x):
        """给定单个待预测数据x，返回x的预测结果值"""
        
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"
        
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

    def predict(self, X_predict): # 注意x_predict是二维矩阵
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        
        assert self._X_train is not None and self._y_train is not None, \
                "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
                "the feature number of X_predict must be equal to X_train"
        
        y_predict = [self._pre(x) for x in X_predict]
        return np.array(y_predict)
    
    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
            
        y_predict = self.predict(X_test)
        return np.sum(y_test == y_predict)/len(y_test)
        
    def __repr__(self):
        return 'KNN(k=%d)' % self.k 