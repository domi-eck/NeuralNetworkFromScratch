import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
    def draw(self):
        numOfBlackWhiteTiles = round(self.resolution / (2 * self.tile_size))
        vector1     = np.ones(self.tile_size)
        vector0     = np.ones(self.tile_size) * -1
        vector10    = np.append(vector1, vector0)
        vector01    = np.append(vector0, vector1)
        vector10    = np.tile(vector10, numOfBlackWhiteTiles)
        vector01    = np.tile(vector01, numOfBlackWhiteTiles)
        matBoard    = (np.outer(vector10, vector10) + 1) / 2
        self.matBoard    = matBoard * 255
    def show(self):
      #  plt.imshow(self.matBoard, cmap='gray')
      plt.imshow(self.matBoard)


if __name__ == '__main__':
    checker = Checker(800,8)
    checker.draw()
    checker.show()