from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import style
from mylib.plotdregion import plot_decision_region
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#scikit-learn은 파이썬으로 구현한 머신러닝ㅇ르 위한 확장 라이브러리. 
#우리가 이전에 다루었던  퍼셉트론은 이 API에 담겨있다!\

style.use('seaborn-talk') 

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:,[2, 3]] #iris의 꽃잎길이 와 꽃잎너비 데이터 대입. 
    y = iris.target # 품종 데이터 대입. 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # test_size = 0.3 의 의미 :
    # X와 y를 무작위로 섞은 후(train_test_split) 70%는 X_train과 y_train으로, 
    # 나머지 30%는 X_test, y_test로 둔다. random_state 의 값은 난수 발생을 위한 seed의 인자값. 

    # 70%의 트레이닝 데이터로 머신러닝을 수행하여 학습이 마무리되면, 나머지 30%의 데이터에 대한 예측값과 
    # 실제 결과값을 비교함으로써 머신러닝이 얼마나 잘되었는지 확인하기 위함임. 

    sc = StandardScaler()
    sc.fit(X_train) # X_train의 평균과 표준편차를 구함
    X_train_std = sc.transform(X_train) #트레이닝 데이터를 표준화
    X_test_std = sc.transform(X_test) # 테스트 데이터를 표준화

    ml = RandomForestClassifier(criterion='entropy', n_estimators= 10, n_jobs = 2, random_state=1)
    # n_estimators : 의사결정트리의 개수
    # n_jobs : CPU 코어 n개를 병렬적으로 활용. 
    ml.fit(X_train_std, y_train)
    y_pred = ml.predict(X_test_std)

    print('총 테스트 개수: %d, 오류개수 : %d' %(len(y_test), (y_test != y_pred).sum()))
    # 테스트값에 대한 실제 결괏값 : y_test
    # 머신러닝을 수행한 퍼셉트론에 의해 예측된 값: y_pred 
    # 이를 비교하여 다른 값을 가진 개수를 출력. 
    
    print('정확도 : %.2f' %accuracy_score(y_test, y_pred))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_region(X=X_combined_std, y=y_combined, classifier=ml,
                         test_idx=range(105, 150), title= 'scikit-learn 랜덤 포레스트')

