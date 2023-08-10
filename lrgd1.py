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


df = pd.read_csv('/Users/leejimin/Desktop/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']


X = df[['RM']].values
y = df[['MEDV']].values

#y.shape = (506, )이며, 이는 scikit-learn의 함수에서 아래와 같은 경고를 유발
#warnings..
#따라서 임시로 차원을 (506, 1)ㄹ ㅗ바꿈 

y = y.reshape((-1, 1))

sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
y_std = sc_x.fit_transform(y)

#변환한 y의 차원 (506, )로 복원
y_std = y_std.reshape((-1, ))

lr = LinearRegressionGD()
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

#RM-MEDV의 표준화된 값을 이용하여, x축은 비용함수의 가중치를 업데이트하는 반복 횟수,
# y축은 LinearRegressionGD에 의해 계삳뇌는 비용함수의 값. 