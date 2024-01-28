import numpy as np
import math

class LogisticRegression:
    def __init__(self, alpha = 0.001, num_iters = 1000):
        self.alpha = alpha
        self.num_iters = num_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        g = 1 / (1 + np.exp(-z))

        return g
    
    def compute_cost(self, X, y, m):
        cost = 0

        for i in range(m):
            z_i = np.dot(X[i], self.weights) + self.bias
            f_wb_i = self.sigmoid(z_i)
            cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
        
        cost = cost / m

        return cost

    def compute_gradient(self, X, y, m, n):
        dj_dw = np.zeros(n)
        dj_db = 0

        for i in range(m):
            z_i = np.dot(X[i], self.weights) + self.bias
            f_wb_i = self.sigmoid(z_i)
            err_i = f_wb_i - y[i]

            for j in range(n):
                dj_dw[j] = dj_dw[j] + err_i * X[i, j]

            dj_db = dj_db + err_i

        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db

    def gradient_descent(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        J_hist = []

        for i in range(self.num_iters):
            dj_dw, dj_db = self.compute_gradient(X, y, m, n)

            self.weights = self.weights - self.alpha * dj_dw
            self.bias = self.bias - self.alpha * dj_db

            if i < 100000:
                J_hist.append(self.compute_cost(X, y, m))
            if i % math.ceil(self.num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_hist[-1]:8.2f}")

        return self.weights, self.bias, J_hist

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        f_wb = self.sigmoid(z)

        class_pred = [0 if pred <= 0.5 else 1 for pred in f_wb]

        return class_pred