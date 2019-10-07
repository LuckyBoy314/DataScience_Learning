import numpy as np
    
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    
    # 通过重排索引达到数据随机化的目的
    all_size = len(X)
    shuffled_indexes = np.random.permutation(all_size)
    
    test_size = int(all_size * test_size)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]
    
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    
    return X_train, X_test, y_train, y_test