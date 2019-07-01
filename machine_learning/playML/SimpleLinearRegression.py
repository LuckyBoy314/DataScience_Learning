import numpy as np


class SimpleLinearRegression1():
    def __init__(self):
        self.a_ = None
        self.b_ = None
        
    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"
        
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        
        numerator = 0.0   # 分子
        denominator = 0.0    # 分母
        for x_i, y_i in zip(x_train, y_train):
            numerator += (x_i - x_mean) * (y_i - y_mean)
            denominator += (x_i - x_mean) * (x_i - x_mean)

        self.a_ = numerator / denominator
        self.b_ = y_mean - self.a_ * x_mean
        
        return self
        
    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
        
        return np.array([self._predict(i) for i in x_predict])
        
    def _predict(self, x_single):
        
        y_hat = self.a_ * x_single + self.b_
        return y_hat
    
    def __repr__(self):
        if self.a_ is None and  self.b_ is None:
            return  'LinearRegression1() to be fitted'
        else:
            return 'LinearRegression1(a = %s, b = %s)'%(self.a_, self.b_)
        
        
class SimpleLinearRegression2():
    def __init__(self):
        self.a_ = None
        self.b_ = None
        
    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"
        
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        
        # numerator = 0.0   # 分子
        # denominator = 0.0    # 分母
        #for x_i, y_i in zip(x_train, y_train):
        #    numerator += (x_i - x_mean) * (y_i - y_mean)
        #    denominator += (x_i - x_mean) * (x_i - x_mean)
        numerator = (x_train - x_mean).dot(y_train - y_mean)
        denominator = (x_train - x_mean).dot(x_train - x_mean)
        
        self.a_ = numerator / denominator
        self.b_ = y_mean - self.a_ * x_mean
        
        return self
        
    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
        
        return np.array([self._predict(i) for i in x_predict])
        
    def _predict(self, x_single):
        
        y_hat = self.a_ * x_single + self.b_
        return y_hat
    
    def __repr__(self):
        if self.a_ is None and  self.b_ is None:
            return  'LinearRegression2() to be fitted'
        else:
            return 'LinearRegression2(a = %s, b = %s)'%(self.a_, self.b_)