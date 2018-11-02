#Test script, no nessecary function here

import numpy as np
import matplotlib.pyplot as plt

resolution = 20

vRed    = np.arange(1.0, 0.0, -1/resolution)
vRed    = np.tile(vRed, resolution)
vRed    = np.reshape(vRed, [resolution* resolution, 1])

vBlue   = np.arange(0.0, 1.0, 1/resolution)
vBlue   = np.tile(vBlue, resolution)
vBlue   = np.reshape(vBlue, [resolution* resolution, 1])

vGreen  = np.arange(0.0, 1.0, 1/resolution)
vGreen  = np.arange(0.0, 1.0, 1/resolution)
vGreen  = np.tile(vGreen, resolution)
vGreen.sort()
vGreen  = np.reshape(vGreen, [resolution*resolution, 1])


#vRB = np.hstack((vRed, vBlue))
#vRB = np.reshape(vRB, [resolution, 2])
vRG = np.hstack((vRed, vGreen))
vRGB = np.hstack((vRG, vBlue))
print(vRGB)
vRGB = np.reshape(vRGB, [resolution, resolution, 3])

plt.imshow(vRGB)

a =3
