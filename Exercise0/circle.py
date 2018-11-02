import numpy as np
import matplotlib.pyplot as plt

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):

        # Create a Black Image________________________________
        v_black = np.zeros([1, self.resolution**2]).transpose()
        v_rgb = np.hstack([v_black, v_black, v_black])
        self.m_rgb = np.reshape(v_rgb, [self.resolution, self.resolution, 3])

        #Create Circle_______________________________________'
        for y in np.arange(self.resolution):
            x_min = np.round(-np.sqrt(self.radius**2 - (y - self.position[1])**2) + self.position[0])
            x_max = np.round(np.sqrt(self.radius ** 2 - (y - self.position[1]) ** 2) + self.position[0])

            if not (np.isnan(x_min) or np.isnan(x_max) or x_min == x_max):
                self.m_rgb[y][int(x_min):int(x_max)] = [1.0, 1.0, 1.0]

    def show(self):
        plt.imshow(self.m_rgb)


if __name__ == '__main__':
    circle = Circle(1000, 300, [400,600])
    circle.draw()
    circle.show()

