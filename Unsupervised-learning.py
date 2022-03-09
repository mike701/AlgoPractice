import scipy.io as sio
import numpy.matlib as nm
import numpy as np
import matplotlib.pyplot as plt
# Import all the data files
plt.figure()
print(np.arange(0, 11, 2))
partialpressure = np.arange(1,10)[:,None]
resistance = np.asarray([15.6, 17.5, 36.6, 43.8, 58.2, 61.2, 64.2, 70.4, 98.8])[:,None]
plt.plot(partialpressure,resistance,'k.')
plt.xlim([0,10])
plt.ylim([0,12])
plt.xlabel('Ar partial pressure')
plt.ylabel('resistance')
plt.show()

