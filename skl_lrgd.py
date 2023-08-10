import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rc
from mylib.adaline import AdalineGD
import numpy as np
from sklearn.linear_model import LinearRegression

style.use('seaborn-talk')

rc('font', family = 'AppleGothic', weight = 'bold', size = 10)
matplotlib.rcParams['axes.unicode_minus'] = False

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='b')
    plt.plot(X, model.predict(X), c='r')

#단, 표준화된 값이므로 이것을 원래 값으로 변환해주어야 그래프 해석에 쉽다
# 예를 들어, 평균 방 수에따른 1000달러 단위 집값의 비례관계는 당연히 표준화된 값보다는 수치화된 값이 유리하다.
#scikitlearn은 표준화된값을 막 변환하고 그럴 필요가 없다! 이거 사용하는게 낫다.

df = pd.read_csv('/Users/leejimin/Desktop/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

X = df[['RM']].values
y = df[['MEDV']].values

slr = LinearRegression()
slr = slr.fit(X, y)

slope = slr.coef_[0]
intercept = slr.intercept_

print('회귀선 기울기: %.3f\n절편: %.3f' %(slope, intercept))

lin_regplot(X, y, slr)
plt.xlabel('평균 방수 [RM]')
plt.ylabel('1000달러 단위 집값 [MEDV]')
plt.title('1978년 미국 보스턴 외곽지역 [방 수 - 주택가격] 츄아 \n')
plt.show()

num_rooms = 5.0


house_val = slr.predict(num_rooms)

print('방이 [%d]개인 주택가격은 약 [%.2f] 달러입니다.' %(int(num_rooms), house_val * 1000))
