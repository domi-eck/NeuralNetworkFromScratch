import numpy as np
import matplotlib.pyplot as plt

resolution = 100
radius = 10
z_x = 30
z_y = 30

'Create a Black Image________________________________'
v_black = np.zeros([1,resolution**2]).transpose()
v_rgb = np.hstack([v_black, v_black, v_black])
m_rgb = np.reshape(v_rgb, [resolution, resolution,3])

'Create Circle_______________________________________'
for y in np.arange(resolution):
    a_min = np.round(-np.sqrt(radius**2 - (y - z_y)**2) + z_x)
    a_max = np.round(np.sqrt(radius ** 2 - (y - z_y) ** 2) + z_x)
    print('min: ', a_min, 'max: ', a_max)

    if not (np.isnan(a_min) or np.isnan(a_max) or a_min == a_max):
        m_rgb[y][int(a_min):int(a_max)] = [1.0, 1.0, 1.0]

'Plot Picture'
plt.imshow(m_rgb)
