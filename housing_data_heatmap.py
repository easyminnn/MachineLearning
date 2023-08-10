import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('/Users/leejimin/Desktop/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size' : 15}, yticklabels=cols, xticklabels=cols)

plt.show()

'''
#이거는 변수 간의 상관관계를 보기에 매우 좋은 방법이다 !
#결과 :
    1. LSTAT와 MEDV는  반비례 그래프가 나와 약간의 상관관계가 있다. 
    2. RM과 MEDV도 비례의 상관관계가 있다. 
''' 
