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

plt.figure(2)
plt.plot(xm, ym, 'ko')
plt.plot(xx, yy, 'r')
intercept = LR.intercept_
coef = LR.coef_
plt.show()
print(intercept, coef)
