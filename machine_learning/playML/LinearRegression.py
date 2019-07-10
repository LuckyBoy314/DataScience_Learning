import numpy as np
from .metrics import r2_score


class LinearRegression():
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None
        
    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        Xb = np.hstack([np.ones((len(X_train),1)), X_train])
        self._theta = np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self
    
    def fit_gd(self, X_train, y_train,initial_theta = None, eta = 0.01, n_iters = 1e4, epsilon=1e-8):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        def J(theta, Xb, y):
            try:
                return np.sum((y - Xb@theta)**2)/len(Xb)
            except:
                return float('inf')

        def DJ(theta, Xb, y):
            n = len(theta)  # 特征个数
            m = Xb.shape[0]
            gradient = np.empty(n)
            gradient[0] = np.sum(Xb@theta - y)
            for i in range(1, n):
                gradient[i] = (Xb@theta - y)@Xb[:,i]
            return gradient*2/m
        
        def gradient_descent(Xb, y, initial_theta, eta, n_iters = 1e4, epsilon=1e-8):
    
            theta = initial_theta
            i_iter = 0

            while i_iter < n_iters:
                gradient = DJ(theta, Xb, y)
                last_theta = theta
                theta = theta - eta * gradient

                if(abs(J(theta, Xb, y) - J(last_theta, Xb, y)) < epsilon):
                    break

                i_iter += 1
            
            return theta
        
        Xb = np.hstack([np.ones((len(X_train),1)), X_train])
        if not initial_theta:
            initial_theta = np.zeros(Xb.shape[1])
        self._theta = gradient_descent(Xb, y_train, initial_theta, eta, n_iters, epsilon)
        
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
        return self
    
        
    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        
        Xb = np.hstack([np.ones((len(X_predict),1)), X_predict])
        return Xb.dot(self._theta)
    
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)
    
    def __repr__(self):
        return 'LinearRegression_MyOwnVersion'
    
    