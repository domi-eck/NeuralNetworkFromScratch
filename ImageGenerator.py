import numpy as np
import json
import matplotlib.pyplot as plt


class Image:
    def __init__(self, s_raw, label, name):
        self.image = np.load(s_raw)
        self.label = label
        self.name = name


class ImageGenerator:
    def __init__(self):
        with open("Exercise0/Labels.json") as json_data:
            self.d_labels = json.load(json_data)
        # read all the images in a list of images
        # self.l_images = np.array([np.load("Exercise0/exercise_data/" + str(k) + ".npy") for k in range(100)])
        self.l_images = np.array([Image("Exercise0/exercise_data/" + str(k) + ".npy", self.d_labels[str(k)], k) for k in range(100)])
        self.mirroring()
        self.rotate90()

        self.switcher = {
            1: "test1",
            2: "test3",
            3: "test2",
            4: "April",
            5: "May",
            6: "June",
            7: "Pferd",
            8: "August",
            9: "September",
            10: "October",
        }

        # read the python dictionary out of the json file


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
            plt.imshow(self.l_batch[i].image)

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
        self.l_images = np.random.shuffle(self.l_images)

    def class_name(self):
        out = np.array(self.batch_size)
        for i in range(0, self.batch_size):
            out[i] = self.switcher[str(self.l_images[i].label)]
        return out


if __name__ == '__main__':
    generator = ImageGenerator()
    batch, labels = generator.next(10)
    generator.class_name()
    generator.show()

    a = 3
