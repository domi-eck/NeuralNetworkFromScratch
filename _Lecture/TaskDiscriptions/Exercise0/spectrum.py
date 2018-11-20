import numpy as np
import matplotlib.pyplot as plt

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution

    def draw(self):
        #creating red color vector for one row
        vRed = np.arange(1.0, 0.0, -1 / self.resolution)
        #tileing it for (we have row of number of resolution
        vRed = np.tile(vRed, self.resolution)
        #reshaping in [resolution^2, 1]
        vRed = np.reshape(vRed, [self.resolution * self.resolution, 1])

        #same as red only increasing arange (blue depends also on the row)
        vBlue = np.arange(0.0, 1.0, 1 / self.resolution)
        vBlue = np.tile(vBlue, self.resolution)
        vBlue = np.reshape(vBlue, [self.resolution * self.resolution, 1])

        #green depends on the colom -> need for sort
        vGreen = np.arange(0.0, 1.0, 1 / self.resolution)
        vGreen = np.arange(0.0, 1.0, 1 / self.resolution)
        vGreen = np.tile(vGreen, self.resolution)
        vGreen.sort()
        #after sort is resolution times the same color value in the array (or the same color in each row)
        vGreen = np.reshape(vGreen, [self.resolution * self.resolution, 1])

        #for RGB color first stack Green to Red
        vRG         = np.hstack((vRed, vGreen))
        #then Blue to RedGreen
        self.vRGB   = np.hstack((vRG, vBlue))
        #shaping it in form of [resolution, resolution, 3(RGB)] (form requested from imshow)
        self.vRGB   = np.reshape(self.vRGB, [self.resolution, self.resolution, 3])

    def show(self):
        plt.imshow(self.vRGB)


if __name__ == '__main__':
    colorSpec = Spectrum(800)
    colorSpec.draw()
    colorSpec.show()