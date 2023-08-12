import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rc
from mylib.adaline import AdalineGD
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import RANSACRegressor

style.use('seaborn-talk')

rc('font', family = 'AppleGothic', weight = 'bold', size = 10)
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('/Users/leejimin/Desktop/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

X = df.iloc[:, :-1].values
y = df[['MEDV']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
## train은 70%, test는 30%

lr = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50,
                         residual_threshold=5.0, random_state=0)

lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

plt.scatter(y_train_pred, y_train_pred-y_train, c = 'blue', marker='o', label='트레이닝데이터')
plt.scatter(y_test_pred, y_test_pred-y_test, c = 'lightgreen', marker='s', label='트레이닝데이터')


plt.xlabel('예측값')
plt.ylabel('잔차')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.title('[다중설명변수-주택가격] 잔차 분석 그림\n')
plt.legend(loc=2)
plt.show()

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print('MSE-트레이닝데이터 : %.2f, 테스트데이터 : %.2f' %(mse_train, mse_test))

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print('R2 - 트레이닝데이터: %.2f, 테스트데이터 : %.2f' %(r2_train, r2_test))

