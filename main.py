import utils.kNN as kNN
import utils.reader as reader
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np


print("Loading training dataset...")
train_images, train_labels = reader.load_dataset('data/dataset', extra=True)
print("Load completed, please type in the test image name")
image_name = str(input())
img = image.imread(image_name)
plt.imshow(img)
img = img.reshape(3072, order='F')
img = img / 255.0
print("classifying...")
hypothesis, score = kNN.classify_single(img, train_images, train_labels, 5)
print("Image contains an animal: ", hypothesis, " with a score of", score, "/5")
