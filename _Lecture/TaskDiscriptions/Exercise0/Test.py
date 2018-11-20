#Test script, no nessecary function here

import numpy as np
import matplotlib.pyplot as plt

a = np.arange(0, 9)

b = a.reshape([3, 3])
print(b)

c = 1/np.sum(b, 0)

b = np.diag(c)
print(b)
print(c)