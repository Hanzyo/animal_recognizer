import utils.kNN as kNN
import utils.reader as reader
import utils.improcess as improccess
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np

k = 10
print("Loading training dataset...")
train_images, train_labels = reader.load_dataset('data/dataset', extra=True)
print("Load completed, please type in the test image name")
image_name = str(input())
img = image.imread(image_name)
img = improccess.downsizer(img)
plt.imshow(img)
plt.show()
img = img.reshape(3072, order='F')
img = img / 255.0
print("classifying...")
hypothesis, score = kNN.classify_single(img, train_images, train_labels, k)
print("Image contains an animal:", bool(hypothesis), " with a confidence of", score[0] * 100 / k, "/100")
