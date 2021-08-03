import pandas as pd
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression
# must be global minimum? for least errors
# how about stochastic gradient descent?
# gradient descent time complexity larger than stochastic gradient descent
# normalization or standardization? reason and difference?
# how does Gaussian noise distribution come from? Note that the Gaussian noise assumption implies that the conditional distribution of t given x is unimodal,
# normal equation: X (N*D), X^T*X: O(D^2*N)


# step 1: normalize dataset
# step 2: linear regression: gradient descent. Several iterations (until an acceptable difference between two iterations)
# step 3: find weights
# for SGD, must randomize dataset first






#def stochastic_gradient_descent(X, y, eta=0.01, max_iter=10000, iter_delta=0.001):









class preprocessor():
    def __init__(self, Xtrain, ytrain, Xtest, ytest):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest
        self.mean = np.mean(Xtrain, axis=0)  # mean of each column vector
        self.std = np.std(Xtrain, axis=0)  # std of each column vector
        self.std[self.std <= 1e-5] = 1 # why? # 减去mean， 当成0。就不除标准差

    def normalizer(self, train_or_test="train"):
        if train_or_test == "train":
            self.Xtrain = (self.Xtrain - self.mean) / self.std
            return self.Xtrain
        else:
            self.Xtest = (self.Xtest - self.mean) / self.std
            return self.Xtest


    def randomizer(self, seed=15):
        """

        :param seed:
        :return: only need to randomize train dataset.
        """
        np.random.seed(seed)
        np.random.shuffle(self.Xtrain)
        np.random.seed(seed)
        np.random.shuffle(self.ytrain)
        return self.Xtrain, self.ytrain


def gradient_descent(X, y, eta=0.01, max_iter=10000, iter_delta=0.001):
    """
        Perform gradient descent for linear regression
        :param X: (N, D)   N*D 2D numpy array (N is the number of the training examples; D is the linear regression basis dimension
        :param y: (D,) N element 1D array
        :param eta: learning rate, default 0.01
        :max_iter: max number of iterations
        :iter_delta: assuming no improvement of weights when the absolute value of w of two iterations is less than iter_delta
        :return: w: D element 1D array, weights. starts from a zeros array
    """
    N, D = X.shape
    w = np.zeros(D)
    loss_prev = -float("inf")
    t = 1
    while t < max_iter:
        gradient = - np.dot(y - np.dot(X, w), X)
        w = w - eta / N * gradient
        loss = np.dot(y - np.dot(X, w), y - np.dot(X, w))
        #print('After {} iterations, the loss now becomes {}'.format(t, loss))
        if abs(loss - loss_prev) <= iter_delta:
            print("The weights converge for GD. Performed {} iterations.".format(t))
            break
        loss_prev = loss
        t += 1

    return w


def stochastic_gradient_descent(X, y, eta=0.1, max_iter=1000, iter_delta=0.01, patience=True, wait_rounds = 5): #after randomize dataset already
    N, D = X.shape
    np.random.seed(20)
    w = np.random.normal(scale=100, size=D)
    loss_prev = -float("inf")
    t = 1
    willBreak = False
    patience_rounds = 0
    while t < max_iter:
        for x_n, y_n in zip(X, y):
            gradient = - np.dot(y_n - np.dot(x_n, w), x_n)
            w = w - eta * gradient
            loss = np.dot(y - np.dot(X, w), y - np.dot(X, w))
            if abs(loss - loss_prev) <= iter_delta:
                if patience:
                    patience_rounds += 1
                    if patience_rounds == wait_rounds:
                        willBreak = True
                        break
                else:
                    willBreak = True
            else:
                patience_rounds = 0
            loss_prev = loss

        if willBreak:
            print("The weights converge for SGD.Performed {} iterations.".format(t))
            break
        t += 1
    return w




def PCC(x, y):
    """
        x and y are 1d vectors having the same size.
        return the Pearson's coefficient.
    """
    m = x.shape[0]
    A = x - np.mean(x)
    B = y - np.mean(y)
    return np.dot(A, B) / np.sqrt(np.dot(A, A) * np.dot(B, B))


def RMSE(x, y):
    """
        x and y are 1d vectors having the same size.
    """
    m = x.shape[0]
    return np.sqrt(np.dot(x - y, x - y) / m)


# load data into dataframe

Xtrain = pd.read_csv("airfoil_self_noise_X_train.csv").values
ytrain = pd.read_csv("airfoil_self_noise_y_train.csv").values
Xtest = pd.read_csv("airfoil_self_noise_X_test.csv").values
ytest = pd.read_csv("airfoil_self_noise_y_test.csv").values

print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)
# reshape y
ytrain = ytrain.flatten()  # or np.ravel()
ytest = ytest.flatten()

# rescale Xtrain. Make sure features are on a similar scale.
scaler = preprocessor(Xtrain, ytrain, Xtest, ytest)
# scaler.fit(Xtrain)
normalized_Xtrain = scaler.normalizer()
normalized_Xtest = scaler.normalizer(train_or_test="test")

Xtrain_ran, ytrain_ran = scaler.randomizer(seed=77)


# Add one column of ones to Xtrain
X = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain_ran), axis=1)

# Perform gradient descent
lr = 0.1
num_iters = 200
theta = gradient_descent(X, ytrain_ran)

# Perform SGD
theta_SGD = stochastic_gradient_descent(X, ytrain_ran)

# Add a column of ones to rescaled npXtest
extended_Xtest = np.concatenate((np.ones((Xtest.shape[0], 1)), normalized_Xtest), axis=1)
ypred = np.dot(extended_Xtest, theta)

# Compute pcc
print("The Pearson correlation coefficient is {:.4f}.".format(PCC(ypred, ytest)))
print("The Pearson correlation coefficient calculated by scipy is {:.4f}.".format(stats.pearsonr(ypred, ytest)[0]))

# Compute rmse
print("The root mean square error is {:.4f}".format(RMSE(ypred, ytest)))

# print theta
print("The coefficients given by the gradient descent method are {}".format(theta))
# print theta_SGD
print("The coefficients given by the stochastic gradient descent method are {}".format(theta_SGD))

# (X^T X)^{-1} X^T y
theta_normal_equation = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), ytrain) # np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), ytrain)
print("The coefficients given by the normal equation are {}".format(theta_normal_equation))



reg = LinearRegression().fit(normalized_Xtrain, ytrain)
print("The coefficients given by the sklearn model are {:.4f} and {}".format(reg.intercept_, reg.coef_))
sklearn_ypred = reg.predict(normalized_Xtest)
print(
    "Given the prediction from sklearn, Pearson correlation coefficient is {:.4f}, and the root mean square error is {:.4f}".format(
        PCC(sklearn_ypred, ytest), RMSE(sklearn_ypred, ytest)))


# import matplotlib.pyplot as plt
# x = np.arange(1, num_iters+1)
# plt.plot(x, loss_history[x-1])
# plt.xlabel('iterations')
# plt.ylabel('loss')
# plt.legend()