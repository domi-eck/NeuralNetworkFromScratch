import numpy as np
import json
import matplotlib.pyplot as plt


class ImageGenerator:
    def __init__(self):
        # read all the images in a list of images
        self.l_images = np.array([np.load("Exercise0/exercise_data/" + str(k) + ".npy") for k in range(100)])

        # read the python dictionary out of the json file
        with open("Exercise0/Labels.json") as json_data:
            self.d_labels = json.load(json_data)

    def next(self, batch_size):
        # TODO rescale the images
        self.batch_size = batch_size
        # choose the images for the batch random
        v_selection = np.array(np.round(np.random.rand(batch_size) * 99), dtype=np.int)
        self.l_batch = self.l_images[v_selection]

        # get the labels for the selected images
        self.d_batch_labels = {key: self.d_labels[str(key)] for key in v_selection}

        #return both, images and labels
        return self.l_batch, self.d_batch_labels

    def show(self):
        # TODO make plot nicer
        fig = plt.figure()
        for i in np.arange(self.batch_size):
            count = np.int(np.round(np.sqrt(self.batch_size) +0.49))
            plt.subplot(count, count, i+1)
            plt.imshow(self.l_batch[i])

if __name__ == '__main__':
    generator = ImageGenerator()
    batch, labels = generator.next(10)
    generator.show()
