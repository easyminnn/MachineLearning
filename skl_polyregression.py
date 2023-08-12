import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rc
from mylib.adaline import AdalineGD
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


style.use('seaborn-talk')

rc('font', family = 'AppleGothic', weight = 'bold', size = 10)
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('/Users/leejimin/Desktop/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

X = df[['RM']].values
y = df[['MEDV']].values

lr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree = 3)

X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

#단순 회귀 계산
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
lr.fit(X, y)
y_lin_fit = lr.predict(X_fit)
l_r2 = r2_score(y, lr.predict(X))

#2차 다항 회귀 모델 적용 후 계산
lr.fit(X_quad, y)
y_quad_fit = lr.predict(quadratic.fit_transform(X_fit))
q_r2 = r2_score(y, lr.predict(X_quad))

#3차 다항 회귀 모델 적용 후 계산
lr.fit(X_cubic, y)
y_cubic_fit = lr.predict(cubic.fit_transform(X_fit))
c_r2 = r2_score(y, lr.predict(X_cubic))


plt.scatter(X, y, c='lightgray', marker='o', label = '트레이닝 데이터')


plt.plot(X_fit, y_lin_fit, linestyle =':', label='linear fit(d=1), $R^2=%.2f$' %l_r2,
         c='blue', lw=3)
plt.plot(X_fit, y_quad_fit, linestyle ='-', label='quad fit(d=2), $R^2=%.2f$' %q_r2,
         c='red', lw=3)
plt.plot(X_fit, y_cubic_fit, linestyle ='--', label='cubic fit(d=3), $R^2=%.2f$' %c_r2,
         c='green', lw=3)
plt.xlabel('인구의 낮은 백분율')
plt.ylabel('1000달러 단위 집값 [MEDV]')
plt.legend(loc=1)
plt.show()

