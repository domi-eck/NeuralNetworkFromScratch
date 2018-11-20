import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.misc as sim


class Image:
    def __init__(self, s_raw, label, label_name, image_name):
        self.image      = np.load(s_raw)
        self.label      = label
        self.label_name = label_name
        self.name       = image_name


class ImageGenerator:
    def __init__(self):
        with open("Labels.json") as json_data:
            self.d_labels = json.load(json_data)

        self.switcher = {
            1: "car",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "June",
            7: "horse",
            8: "ship",
            9: "truck",
            10: "mysterius",
            0: "airplane"

        }
        # read all the images in a list of images
        # self.l_images = np.array([np.load("Exercise0/exercise_data/" + str(k) + ".npy") for k in range(100)])
        self.l_images = np.array([Image("exercise_data/" + str(k) + ".npy", self.d_labels[str(k)], self.switcher[self.d_labels[str(k)]], k) for k in range(100)])
        self.mirroring()
        self.rotate90()
        # read the python dictionary out of the json file


    def next(self, batch_size):
        self.batch_size = batch_size
        # choose the images for the batch random
        self.shuffling()
        v_selection = np.arange(0, batch_size)
        for i in v_selection:
            self.l_images[i].image = sim.imresize(self.l_images[i].image, (100, 100))
        self.l_batch = self.l_images[v_selection]

        # get the labels for the selected images
        self.d_batch_labels = {key: self.d_labels[str(key)] for key in v_selection}

        #return both, images and labels
        return self.l_batch, self.d_batch_labels

    def show(self):
        count = np.int(np.round(np.sqrt(self.batch_size) + 0.49))
        f, a = plt.subplots(count, count)

        for i in np.arange(count):
            for j in np.arange(count):
                a[i][j].set_axis_off()

        for i in np.arange(self.batch_size):
            j = int(i % count)
            ii = int(np.floor(i/count))
            a[ii][j].set_title(self.l_batch[i].label_name)
            a[ii][j].imshow(self.l_batch[i].image)


    def mirroring(self):
        num = np.random.randint(100)
        v_selection = np.random.randint(100, size = num)
        for k in v_selection:
            self.l_images[k].image = np.flip(self.l_images[k].image, np.random.randint(2))

    def rotate90(self):
        num = np.random.randint(100)
        v_selection = np.random.randint(100, size=num)
        for k in v_selection:
            self.l_images[k].image = np.rot90(self.l_images[k].image, 1)

    def shuffling(self):
        np.random.shuffle(self.l_images)

    def class_name(self):
        out = [""] * self.batch_size
        for i in range(0, self.batch_size):
            out[i] = self.switcher[self.l_images[i].label]
        return out


if __name__ == '__main__':
    generator = ImageGenerator()
    batch, labels = generator.next(10)
    generator.class_name()
    generator.show()

    a = 3
