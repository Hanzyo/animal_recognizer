'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

def find_distance(test_image, train_image):
    return np.sqrt(np.sum(np.square(np.subtract(train_image, test_image))))
    
    
def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''
    dict = {}
    dim = len(image)
    neighbors = np.empty((k, dim))
    labels = np.empty(k)
    distances = np.empty(train_images.shape[0])
    for i in range(0, train_images.shape[0]):
        distance = find_distance(image, train_images[i])
        distances[i] = distance
        dict[distance] = i
    distances = np.sort(distances)
    for i in range(0, k):
        image_i = dict[distances[i]]
        neighbors[i] = (train_images[image_i])
        labels[i] = (train_labels[image_i])
    return neighbors, labels
  
            
def classify_single(test_image, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    hypotheses = []
    scores = []
    neighbors, labels = k_nearest_neighbors(test_image, train_images, train_labels, k)
    num_true = np.sum(labels)
    num_false = len(labels) - num_true
    if num_true == num_false:
        hypotheses.append(0)
        scores.append(num_false)
    else:
        if num_true > num_false:
            hypotheses.append(1)
            scores.append(num_true)
        else:
            hypotheses.append(0)
            scores.append(num_false)
    return hypotheses, scores


def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    hypotheses = []
    scores = []
    for i in range(0, len(dev_images)):
        neighbors, labels = k_nearest_neighbors(dev_images[i], train_images, train_labels, k)
        num_true = np.sum(labels)
        num_false = len(labels) - num_true
        if num_true == num_false:
            hypotheses.append(0)
            scores.append(num_false)
        else:
            if num_true > num_false:
                hypotheses.append(1)
                scores.append(num_true)
            else:
                hypotheses.append(0)
                scores.append(num_false)
    return hypotheses, scores
        
    


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''

    confusions = np.zeros((2,2))
    accuracy = 0.0
    f1 = 0.0
    
    for i in range(0, len(hypotheses)):
        if hypotheses[i] == 0:
            if references[i] == 0:
                confusions[0][0] += 1
            else:
                confusions[1][0] += 1
        else:
            if references[i] == 0:
                confusions[0][1] += 1
            else:
                confusions[1][1] += 1
    
    precision = (confusions[1][1]) / (confusions[1][1] + confusions[0][1])
    recall = (confusions[1][1]) / (confusions[1][1] + confusions[1][0])
    accuracy = (confusions[1][1] + confusions[0][0]) / np.sum(confusions)
    f1 = 2 / (1 / recall + 1 / precision)
    return confusions, accuracy, f1
