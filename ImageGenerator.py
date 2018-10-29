import numpy as np
import json
import matplotlib.pyplot as plt

class ImageGenerator:
    def __init__(self):

        self.l_images = np.array([np.load("Exercise0/exercise_data/" + str(k) + ".npy") for k in range(100)])

        with open("Exercise0/Labels.json") as json_data:
            self.d_labels = json.load(json_data)


