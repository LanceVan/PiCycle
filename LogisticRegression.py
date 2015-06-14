import numpy as np
import matplotlib.pyplot as plt

from math import isnan

from LinearRegression import *

class LogisticRegression(LinearRegression):
    def __init__(self, x, y):
        LinearRegression.__init__(self, x, y)

    def hypothesis(self, x):
        return 1 / (1 + np.exp(-np.dot(x, self.theta.transpose())))

    def likelyhoodFunction(self):
        return np.dot(self.y, np.log(self.hypothesis(self.x))) + np.dot((1 - self.y), np.log(1 - self.hypothesis(self.x)))

    def gradient(self, i, alpha):
        return (alpha / self.quantity) * np.dot(self.y - self.hypothesis(self.x), self.x)

    def gradientAscent(self, alpha):
        step = np.array(self.size)
        for i in range(self.quantity):
            step = step + self.gradient(i, alpha)
        print(self.theta)
        return self.theta + step

    def run(self, alpha, iters = -1):
        if iters != -1:
            for i in range(iters):
                self.theta = self.gradientAscent(alpha)
        else:
            likelhd0 = 0
            self.theta = self.gradientAscent(alpha)
            likelhd = self.likelyhoodFunction()
            delta = abs((likelhd0 - likelhd) / likelhd)
            while delta > 0.0001 or isnan(delta):
                likelhd0 = likelhd
                self.theta = self.gradientAscent(alpha)
                likelhd = self.likelyhoodFunction()
                delta = abs((likelhd0 - likelhd) / likelhd)
        return 0
