import numpy as np
import matplotlib.pyplot as plt


reso = 10

v1 = np.arange(0.0, 1.0, 1/reso)
v1 = np.tile(v1, reso)
v1.reshape(v1, (reso, reso))
print(v1)
v2 = np.zeros(reso)
v3 = np.zeros(reso)
v4 = np.zeros(reso)
vr = np.vstack((v1, v2, v3,)).transpose()
vc = np.vstack((v1, v2, v3,)).transpose()
va = np.vstack(([vr], [vc]))
plt.imshow(va)
print(va)
