from __future__ import division
from sklearn.decomposition import FastICA
from sklearn import svm
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


train_path = 'C:\\Users\\Yevhen\\PycharmProjects\\untitled\\train\\'
test_path = 'C:\\Users\\Yevhen\\PycharmProjects\\untitled\\test\\'

def transform_images(path):
    images = []
    labels = []
    for filename in os.listdir(path):
        im = cv2.imread(path + filename, -1)
        flat_im = im.flatten()
        images.append(flat_im)
        if 'neg' in filename:
            labels.append(0)
        else:
            labels.append(1)
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels

def benchmark(labels, predictions):
    true = 0
    false = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            true += 1
        else:
            false += 1
    return (true / (false + true)) * 100

if __name__ == '__main__':
    train_images = transform_images(train_path)
    test_images = transform_images(test_path)

    x = []
    y = []

    for i in range(50, 500, 10):
        ica = FastICA(n_components=i, whiten=True)
        train = ica.fit(train_images[0]).transform(train_images[0])
        test = ica.transform(test_images[0])

        clsf = svm.SVC()
        clsf.fit(train, train_images[1])
        predvals = clsf.predict(test)
        
        x.append(i)
        y.append(benchmark(test_images[1], predvals))

    plt.plot(x, y)
    plt.xlabel('Number of independent components')
    plt.ylabel('Performance in %')
    plt.show()

