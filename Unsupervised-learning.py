from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import scipy.io as sio
import numpy.matlib as nm
import numpy as np
import matplotlib.pyplot as plt

partialpressure = np.arange(1, 10)[:, None]
resistance = np.asarray([15.6, 17.5, 36.6, 43.8, 58.2,
                        61.2, 64.2, 70.4, 98.8])[:, None]
plt.figure(1)
plt.plot(partialpressure, resistance, 'k.')
plt.xlim([0, 10])
plt.ylim([0, 120])
plt.xlabel('Ar partial pressure')
plt.ylabel('resistance')
plt.show()
xm = np.arange(1, 10)[:, None]
ym = np.asarray([15.6, 17.5, 36.6, 43.8, 58.2,
                61.2, 64.2, 70.4, 98.8])[:, None]
plt.figure(2)
plt.plot(xm, ym, 'k.')
plt.xlim([0, 10])
plt.ylim([0, 120])
# Fits a linear regression to data

# Ordinary linear regression and there is an intercept on the y-axis
LR = LinearRegression(fit_intercept=True)
LR.fit(xm, ym)

# These two lines create data points for the formation of the red predicted line
xx = np.arange(0, 10, .1)[:, None]
yy = LR.predict(xx)

plt.figure(3)
plt.plot(xm, ym, 'ko')
plt.plot(xx, yy, 'r')
intercept = LR.intercept_
coef = LR.coef_
plt.show()
print(intercept, coef)


# This is a way to visualize the mean squared error loss for linear regression

# Performance metrics
# taken from prediction above
slope_true = 9.4
intercept_true = 4.8

# Creates a grid
slope_, intercept_ = np.meshgrid(np.arange(-10, 20, .1), np.arange(-80, 80, 2))
slope_ = slope_.flatten()
intercept_ = intercept_.flatten()
mse = np.zeros(slope_.shape)  # Creates array for MSE
rsquared = np.zeros(slope_.shape)  # Same for r squared

# Calculates the MSE for each point on said grid
for i in range(slope_.shape[0]):
    y_hat = slope_[i] * xm + intercept_[i]
    mse[i] = mean_squared_error(ym, y_hat)
print(mse)
plt.figure(4)
plt.subplot(1, 2, 2)
# plotting mse^(1/5) just so can see the color range better.
plt.scatter(slope_, intercept_, s=10, c=mse**.2, cmap='viridis')
plt.plot(slope_true, intercept_true, 'ro')
plt.show()
# The yellows at the edges are the most extreme losses and the darker areas are the lowest loss

# Ridge(alpha=1.0, *, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)  #Content of ridge function
# Creates model only for Ridge and LASSO regression


# Ridge Regression
RR = Ridge(alpha=0.0005, fit_intercept=True)
RR.fit(partialpressure, resistance)
print(RR.intercept_, RR.coef_)

# LASSO
L1 = Lasso(alpha=0.0005, fit_intercept=True)
L1.fit(partialpressure, resistance)
print(L1.intercept_, L1.coef_)

# needs additional work, example of use for these models

# More complex data of the form y=0.1x^4+0.5x^3+0.5x^2+x+2
N = 100
x = np.random.uniform(-5, 2, (N, 1))
o = np.ones(x.shape)
X = np.concatenate((x**4, x**3, x**2, x**1, o), axis=1)
c = np.asarray([.1, .5, .5, 1, 2])[:, None]
print(x.shape, X.shape, c.shape)
y = X @ c + np.random.normal(0, .5, x.shape)

plt.figure()
plt.plot(x, y, 'k.')

# Comparison of models on complex data versus simple linear data
RR = Ridge(alpha=0.0005, fit_intercept=True)
RR.fit(X, y)
print(RR.intercept_, RR.coef_)
# LASSO doesn't work as well on complex data
L1 = Lasso(alpha=0.0005, fit_intercept=True)
L1.fit(X, y)
print(L1.intercept_, L1.coef_)

# ? Why are the value differing from above
L1 = Lasso(alpha=1, fit_intercept=True)
L1.fit(X, y)
print(L1.intercept_, L1.coef_)

# let's get a random permutation of the data split
rand_order = np.random.permutation(N)

# use the random permutation to reorder our data.
# our pairs of data (x,y) are still tied together, just new order:
# (x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5)
# becomes:
# (x5,y5), (x2,y2), (x3,y3), (x1,y1), (x4,y4)

X_split = [None]*10
y_split = [None]*10
x_split = [None]*10
labels = np.zeros((N, 1))

x_reordered = x[rand_order]
y_reordered = y[rand_order]

for i in range(10):
    start_num = i*10
    end_num = i*10+10
    idx = rand_order[start_num:end_num]
    X_split[i] = X[idx, :]
    y_split[i] = y[idx]
    labels[start_num:end_num] = i

plt.figure()
plt.scatter(x_reordered, y_reordered, c=labels)
