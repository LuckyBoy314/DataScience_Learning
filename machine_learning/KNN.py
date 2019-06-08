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