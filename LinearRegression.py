import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Type of Parameter should be numpy.ndarray")
        
        sizex = x.shape
        sizey = y.shape
        if len(sizey) != 1:
            raise ValueError("Size of Dependent Variable Parameter should be vector based on numpy.ndarray")
        if sizex[0] != sizey[0]:
            raise ValueError("Size of Pamameters should be matched")
        
        self.quantity = sizex[0]
        if len(sizex) == 1:
            self.size = 2
        else:
            self.size = sizex[1] + 1

        self.x = np.column_stack((np.ones((self.quantity, 1)), x))
        self.y = y
        self.theta = np.ones(self.size)

    def hypothesis(self, x):
        return np.dot(x, self.theta.transpose())

    def costFunction(self):
        return ((np.dot(self.x, self.theta.transpose()) - self.y) ** 2).sum()

    def gradient(self, i, alpha):
        return (alpha / self.quantity) * (np.dot(self.x[i], self.theta.transpose()) - self.y[i]) * self.x[i]

    def gradientDescent(self, alpha):
        step = np.zeros(self.size)
        for i in range(self.quantity):
            step = step + self.gradient(i, alpha)
#        print(self.theta)
        return self.theta + step

    def run(self, alpha, iters = -1):
        if iters != -1:
            for i in range(iters):
                self.theta = self.gradientDescent(alpha)
        else:
            cost0 = 0
            self.theta = self.gradientDescent(alpha)
            cost = self.costFunction()
            while abs((cost0 - cost) / cost) > 0.0001:
                cost0 = cost
                self.theta = self.gradientDescent(alpha)
                cost = self.costFunction()
        return 0

    def predict(self, x0):
        x = np.array([1, x0])
        return self.hypothesis(x)

    def display(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.x.transpose()[1], np.dot(self.x, self.theta.transpose()), 'b--')
        ax.plot(self.x.transpose()[1], self.y, 'rx')
        plt.show()
        return 0
