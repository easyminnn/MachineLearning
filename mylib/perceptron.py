import numpy as np

class Perceptron():
    def __init__(self, thresholds = 0.0, eta = 0.01, n_iter = 10):
        self.thresholds = thresholds
        self.eta = eta
        self.n_iter = n_iter
 
    def fit(self, X, y):  #"트레이닝 데이터 X와 실제 결과값 y를 인자로 받아 머신러닝 수행"
        self.w_ = np.zeros(1+X.shape[1])  # "X.shape[1]은 트레이닝 데이터의 입력값개수"
        self.errors_ = [] #"퍼셉트론의 예측값과 실제 결과값이 다른 오류 횟수를 저장하기 위한 변수"


        for _ in range(self.n_iter):# "10번 반복수행"
            errors = 0   #"초기오류횟수"
            for xi, target in zip(X, y): #"X와 y는 배열로 이루어져 있으며, x0(=1),x1,x2,x3,,,, "
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi   #퍼셉트론 알고리즘에서 가중치 업데이트하는 식"
                self.w_[0] += update  # x0=1이므로 단순히 update만 더하기(update * 1 = upadate)
                errors  += int(update !=0.0)  # "update가 0이면 리턴값 동일. 아니라면 error++"
            self.errors_.append(errors) # "발생한 오류 횟수 추가(10번 중 몇번)" 
            print(self.w_) #"업데이트된 가중치 출력"








    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0] # "내적 함수 dot - X와 가중치 곱"
        #"이처럼 numpy를 통해 행렬 곱이나 벡터 내적이 쉽게 가능하다 ."


    def predict(self, X):
        return np.where(self.net_input(X) > self.thresholds, 1, -1)