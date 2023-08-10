import numpy as np
from numpy.random import seed  #난수발생기 초기화를 위해 임포트

class AdalineSGD():
    def __init__ (self, eta=0.01, n_iter = 10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle

        if random_state:
            seed(random_state)     # random_state의 값이 있으면 이 값으로 난수발생기 초기화. 

    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            cost = []   #가중치 업데이트하는 식 구현.
            for xi, target in zip(X, y):
                output = self.net_input(xi)
                error = target - output
                self.w_[1:] += self.eta * xi.dot(error)
                self.w_[0] += self.eta * error
                cost.append(0.5 * error ** 2)

            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))  #0~y까지의 수를 랜덤하게 섞은 것. 
        return X[r], y[r]  #X의 데이터와 y의 데이터는 순서가 섞였다(X,y 쌍은 동일. 순서만 다름)
                
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)