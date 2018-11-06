import numpy as np
import matplotlib.pyplot as plt

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):

        xx, yy = np.mgrid[:self.resolution, :self.resolution]
        circle = (xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2

        self.circle_shape = np.logical_and(circle < (self.radius)**2, circle >= (0))

    def show(self):
        plt.imshow(self.circle_shape, cmap = "gray")


if __name__ == '__main__':
    circle = Circle(1000, 300, [200,400])
    circle.draw()
    circle.show()

