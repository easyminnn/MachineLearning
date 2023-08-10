import numpy as np 

class AdalineGD():
    def __init__(self, eta = 0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):  # 이부분이 다르다! 
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors) # X.T는 X의 전치행렬을 의미.
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2
            self.cost_.append(cost)   # 비용 (2차함수, 하지만 w의 변수가 배열로 되어있는)
            print(self.w_)

        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    

# 퍼셉트론을 구현한 Perceptron 클래스와 아달라인 AdalineGDd 클래스에서의 다른 부분은 fit()내에서 구현한 함수뿐. 


