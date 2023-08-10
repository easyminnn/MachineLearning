import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib import rc

rc('font', family = 'AppleGothic', weight = 'bold', size = 10)
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_decision_region(X, y, classifier, test_idx=None, resolution = 0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))]) #인자로 주어진 색상을 그래프상에 표시하기 위함
    #np.unique(y) : y에 있는 고유한 값을 작은 값 순으로 나열함. 
    # ex) y=[0,1,2,2,1,2,0]일 때는 [0,1,2]로 중복없이 나열됨.  

    #decision surface 그리기 
    x1_min, x1_max  =X[:, 0].min() - 1, X[:, 0].max() + 1 #꽃잎길이
    x2_min, x2_max  =X[:, 1].min() - 1, X[:, 1].max() + 1 #꽃잎너비
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    #np.meshgird는 교차점 좌표를 편리하게 다룰 수 있도록 값을 리턴하는 함수. 
    #트레이닝 데이터의 꽃잎길이, 꽃잎너비가 분포하는 좌표 격자 교차점을 resolution 간격으로 만들어줌.
    
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)

    #ravel() : 다차원 배열을 일차원 배열로 만들어줌. 그리고 .T (전치행렬로의 변환)
    # 이렇게 변환한 것을 predict()의 인자로 입력하여 계산된 예측값을 Z라고 둠 .
    # Z는 reshape을 이용해 원래 배열 모양으로 복원함. (ravel과 반대)

    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap) 
    # Z를 xx,yy가 축인 그래프상에 cmap을 이용해 등고선을 그림. 
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx],
                    label=cl)
        
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],  linewidth=1, marker='o', s=80,
                    label ='테스트셋')
        
    plt.legend(loc = 2)
    plt.title(title)
    plt.show()

