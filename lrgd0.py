import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rc
from mylib.adaline import AdalineGD
import numpy as np
from sklearn.preprocessing import StandardScaler

style.use('seaborn-talk')

rc('font', family = 'AppleGothic', weight = 'bold', size = 10)
matplotlib.rcParams['axes.unicode_minus'] = False


class LinearRegressionGD():
    def __init__ (self, eta = 0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):  # 이부분이 다르다! 
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors) # X.T는 X의 전치행렬을 의미.
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)   # 비용 (2차함수, 하지만 w의 변수가 배열로 되어있는)
            print(self.w_)

        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return self.net_input()
