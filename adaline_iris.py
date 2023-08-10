import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from mylib.adaline import AdalineGD
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv('iris.data.csv', header = None)
    y = df.iloc[0:100, 4].values
    y = np.where(y=='Iris-setosa', -1 , 1)
    X = df.iloc[0:100, [0,2]].values

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))

    adal = AdalineGD(eta = 0.01, n_iter = 10).fit(X , y)
    ax[0].plot(range(1, len(adal.cost_) + 1), np.log10(adal.cost_), marker= 'o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(SQE)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    adal2 = AdalineGD(eta = 0.0001, n_iter = 10).fit(X , y)
    ax[1].plot(range(1, len(adal2.cost_) + 1), np.log10(adal2.cost_), marker = 'o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(SQE)')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.subplots_adjust(left = 0.1, bottom = 0.2, right = 0.95, top = 0.9, 
                        wspace = 0.5, hspace=0.5)
    plt.show()

    #총 두 개의 비용함수 J(w)를 비교한다. (이 두 함수의 다른 점은 eta값이다)
    #eta = 0.01 인 함수의 경우, 그래프가 수렴하지 않고 발산함. 
    #eta = 0.0001 인 함수의 경우, 수렴.
    # 따라서, eta 값(learning rate)를 매우 작게 해야 아달라인의 J(w) 값이 작아지고 우리가
    # 원하는 머신러닝 값을 얻을 수 있다. 