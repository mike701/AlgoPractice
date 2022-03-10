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
