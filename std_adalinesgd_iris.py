import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rc
from mylib.adalinesgd import AdalineSGD
import numpy as np

style.use('seaborn-talk')

rc('font', family = 'AppleGothic', weight = 'bold', size = 10)
matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    df = pd.read_csv('iris.data.csv', header = None)
    y = df.iloc[0:100, 4].values
    y = np.where(y=='Iris-setosa', -1 , 1)
    X = df.iloc[0:100, [0,2]].values

    X_std = np.copy(X)    # X를 하나 복사하여 X_std로 둔다.
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:, 0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:, 1].std()
    # numpy의 mean()은 numpy 배열에 있는 값들의 평균을, std()는 numpy 
    # 배열에 있는 값들의 표준편차를 구해주는 함수이다. 

    adal = AdalineSGD(eta = 0.01, n_iter = 15, random_state=1).fit(X_std , y)
    plt.plot(range(1, len(adal.cost_) + 1), adal.cost_, marker= 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.title('Adaline Stochastic GD - Learning rate 0.01')

    plt.show()

    #총 두 개의 비용함수 J(w)를 비교한다. (이 두 함수의 다른 점은 eta값이다)
    #eta = 0.01 인 함수의 경우, 그래프가 수렴하지 않고 발산함. 
    #eta = 0.0001 인 함수의 경우, 수렴.
    # 따라서, eta 값(learning rate)를 매우 작게 해야 아달라인의 J(w) 값이 작아지고 우리가
    # 원하는 머신러닝 값을 얻을 수 있다. 