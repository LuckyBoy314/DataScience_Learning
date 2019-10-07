import numpy as np



class StandardScaler():
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        """根据训练数据集X获得数据的均值和方差"""
        assert X.ndim == 2, "The dimension of X must be 2"

        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        
        return self
    
    def transform(self, X):
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
               "must fit before transform!"
        assert X.shape[1] == len(self.mean_), \
               "the feature number of X must be equal to mean_ and std_"
        
        return (X - self.mean_)/self.scale_
    

    
class MinMaxScaler():
    def __init__(self):
        self.min_ = None
        self.max_ = None
        
    def fit(self, X):
        """根据训练数据集X获得数据的均值和方差"""
        assert X.ndim == 2, "The dimension of X must be 2"

        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        
        return self
    
    def transform(self, X):
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.min_ is not None and self.max_ is not None, \
               "must fit before transform!"
        assert X.shape[1] == len(self.min_), \
               "the feature number of X must be equal to mean_ and std_"
        
        return (X - self.min_)/(self.max_ - self.min_)