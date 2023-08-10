import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/leejimin/Desktop/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

sns.set(style = 'whitegrid', context = 'notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()
sns.reset_orig() #원래 matplot 으로 복귀
'''
#이거는 변수 간의 상관관계를 보기에 매우 좋은 방법이다 !
#결과 :
    1. LSTAT와 MEDV는  반비례 그래프가 나와 약간의 상관관계가 있다. 
    2. RM과 MEDV도 비례의 상관관계가 있다. 
''' 
#끝