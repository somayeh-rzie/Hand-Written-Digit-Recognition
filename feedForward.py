import math
import random
import numpy as np

DEFINE_NUMBER = 100

def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


def rand():
    return random.uniform(-1, 1)


def random_arr(arr, k, n):
    for i in range(k):
        for j in range(n):
            arr[i][j] = rand()
    return arr


def findMax(arr):
    index = 0
    for i in range(len(arr)):
        if(arr[i] > arr[index]) :
            index = i
    return index



def feedForward(ndarray):
    first_dim = 784
    second_dim = 16
    third_dim = 16
    fourth_dim = 10

    first_weight = np.zeros((second_dim , first_dim))
    first_weight = random_arr(first_weight , second_dim , first_dim)
    first_bias = np.zeros((second_dim , 1))

    second_weight = np.zeros((third_dim, second_dim))
    second_weight = random_arr(second_weight , third_dim , second_dim)
    second_bias = np.zeros((third_dim , 1))

    third_weight = np.zeros((fourth_dim, third_dim))
    third_weight = random_arr(third_weight , fourth_dim , third_dim)
    third_bias = np.zeros((fourth_dim , 1))

    first_temp = np.dot(first_weight, ndarray)
    first_layer = (np.matrix(first_temp)) + (np.matrix(first_bias))
    for i in range(len(first_layer)):
        first_layer[i] = sigmoid(first_layer[i])

    second_temp = np.dot(second_weight, first_layer)
    second_layer = (np.matrix(second_temp)) + (np.matrix(second_bias))
    for i in range(len(second_layer)):
        second_layer[i] = sigmoid(second_layer[i])
    
    third_temp = np.dot(third_weight, second_layer)
    third_layer = (np.matrix(third_temp)) + (np.matrix(third_bias))
    for i in range(len(third_layer)):
        third_layer[i] = sigmoid(third_layer[i])

    index = findMax(third_layer)
    return index
    


def readFile():

    train_images_file = open('train-images-idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)

    train_labels_file = open('train-labels-idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    train_set = []

    counter = 0

    for n in range(DEFINE_NUMBER):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256
        index = feedForward(image)
        
        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        train_set.append((image, label))

        if(index == label_value) :
            counter += 1
    
    print ("Accuracy is : {} %" .format(counter))

readFile()