import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rc
from mylib.adaline import AdalineGD
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

style.use('seaborn-talk')

rc('font', family = 'AppleGothic', weight = 'bold', size = 10)
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('/Users/leejimin/Desktop/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

X = df[['RM']].values
y = df[['MEDV']].values

ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50,
                         residual_threshold=5.0, random_state=123)

ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label = 'Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label = 'Outliers')

plt.plot(line_X, line_y_ransac, c='red')
plt.xlabel('평균 방수 [RM]')
plt.ylabel('1000달러 단위 집값 [MEDV]')
plt.title('1978년 미국 보스턴 외곽지역 [방 수 - 주택가격] 츄아 \n')
plt.show()

slope = ransac.estimator_.coef_[0]
intercept = ransac.estimator_.intercept_
print('회귀선 기울기: %.3f\n절편 : %.3f' %(slope, intercept))