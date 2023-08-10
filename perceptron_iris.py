import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rc
from mylib.perceptron import Perceptron
import numpy as np

style.use('seaborn-talk')

##krfont = {'fmaily' : '', 'weight' : 'bold', 'size' : 10}
rc('font', family = 'AppleGothic', weight = 'bold', size = 10)
matplotlib.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':
    style.use('seaborn-talk')

    df = pd.read_csv('iris.data.csv', header=None)
    #csv형식으로 저장된 iris데이터를 읽고, pandas의 dataframe 객체로 변환. 

    y = df.iloc[0:100, 4].values #0~99라인까지 5번쨰 column(아이리스 품종)의 데이터값을 
                                    #numpy배열로 리턴받아 y에 대입
    y = np.where(y=='Iris-setosa', -1, 1)
    "품종이 iris-setosa인 경우 -1, 아닌 경우 1로 바꾼 numpy배열을 y에 다시 대입"
    X = df.iloc[0:100, [0, 2]].values
    #iris데이터를 저장한 dataframe에서 1번째(꽃받침길이), 3번째 컬럼(꽃잎길이)의 데이터값을
                            # numpy배열로 리턴받아 X에 대입. 

    #즉, 이 코드는 꽃받침길이와 꽃잎길이에 따라 품종을 머신러닝으로 학습하는 내용이 될 것.


    '''해당 plt는 X에 저장된 꽃받침길이와 꽃잎길이에 대한 데이터의 상관관계를 파악하기 위해
        Matplotlib의 산점도를 이용해 화면에 그려보는 코드임.(데이터 시각화 프로그래밍)'''
    plt.scatter(X[:50, 0], X[:50, 1], color = 'r', marker = 'o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color ='b', marker='x', label = 'versicolor')
    plt.xlabel('꽃잎 길이(cm)')
    plt.ylabel('꽃받침 길이(cm)')
    plt.legend(loc = 4)
    plt.show()


    ppn1 = Perceptron(eta = 0.1)
    ppn1.fit(X, y)
    print(ppn1.errors_)