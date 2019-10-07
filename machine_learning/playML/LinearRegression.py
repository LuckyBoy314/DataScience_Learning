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
    
    
    def fit_gd0(self, X_train, y_train,initial_theta = None, eta = 0.01, n_iters = 1e4, epsilon=1e-8):
        """
            循环的方式计算
            根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        # 注意theta是待求解的未知数，Xb和y是目标函数的参数
        def J(theta, Xb, y):
            try:
                # return np.sum((y - Xb@theta)**2)/len(Xb)
                # 使用完全向量化计算更快一点
                tmp = y - Xb@theta
                return (tmp@tmp)/len(Xb)
            except:
                return float('inf')

        def DJ(theta, Xb, y):
            n = len(theta)  # 特征个数
            m = Xb.shape[0] # 样本个数
            gradient = np.empty(n)
            gradient[0] = np.sum(Xb@theta - y)
            for j in range(1, n):
                gradient[j] = (Xb@theta - y)@Xb[:,j]
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
    
    
    def fit_gd(self, X_train, y_train,initial_theta = None, eta = 0.01, n_iters = 1e4, epsilon=1e-8):
        """
            向量化计算
            根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        # 注意theta是待求解的未知数，Xb和y是目标函数的系数
        def J(theta, Xb, y):
            try:
                # return np.sum((y - Xb@theta)**2)/len(Xb)
                # 使用完全向量化计算更快一点
                tmp = y - Xb@theta
                return (tmp@tmp)/len(Xb)
            except:
                return float('inf')

        def DJ(theta, Xb, y):
            
            return Xb.T@(Xb@theta - y)*2/len(y)
        
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
    
    fit_bgd = fit_gd
    
    def fit_sgd(self, X_train, y_train, initial_theta=None,  n_iters=5, t0=5, t1=50):
        """
            根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1
        
        def dJ_sgd(theta, Xb_i, y_i):
            return 2 * Xb_i.T.dot(Xb_i.dot(theta) - y_i)

        def sgd(Xb, y, initial_theta, n_iters, t0, t1):

            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(Xb)
            
            for i_iter in range(n_iters):
                indexes = np.random.permutation(m)
                Xb_new = Xb[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, Xb_new[i], y_new[i])
                    theta = theta - learning_rate(i_iter * m + i) * gradient
                    
            return theta
        
        Xb = np.hstack([np.ones((len(X_train),1)), X_train])
        if not initial_theta:
            initial_theta = np.random.randn(Xb.shape[1])
            
        self._theta = sgd(Xb, y_train, initial_theta, n_iters, t0, t1)
        
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
    
    