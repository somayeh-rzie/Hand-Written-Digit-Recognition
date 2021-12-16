import random
import numpy as np
import matplotlib.pyplot as plt
import time


DEFINE_NUMBER = 100
#number_of_epochs
epochs = 20
#batch_size
batch_size = 10
#learning_rate
alpha = 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)


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



def backPropagation():

    first_dim = 784
    second_dim = 16
    third_dim = 16
    fourth_dim = 10

    counter = 0

    first_weight = np.zeros((second_dim , first_dim))
    first_weight = random_arr(first_weight , second_dim , first_dim)
    first_bias = np.zeros((second_dim , 1))

    second_weight = np.zeros((third_dim, second_dim))
    second_weight = random_arr(second_weight , third_dim , second_dim)
    second_bias = np.zeros((third_dim , 1))

    third_weight = np.zeros((fourth_dim, third_dim))
    third_weight = random_arr(third_weight , fourth_dim , third_dim)
    third_bias = np.zeros((fourth_dim , 1))


    first_grad_w = np.zeros((second_dim , first_dim))
    second_grad_w = np.zeros((third_dim, second_dim))
    third_grad_w = np.zeros((fourth_dim, third_dim))
        
    first_grad_b = np.zeros((second_dim , 1))
    second_grad_b = np.zeros((third_dim , 1))
    third_grad_b = np.zeros((fourth_dim , 1))


    train_set = read_train_set()

    start = time.time()

    random.shuffle(train_set)
    chunks = (DEFINE_NUMBER - 1) // batch_size + 1
    outputs = np.zeros((int(batch_size) , 1))
    costs = np.zeros((int(epochs) , 1))

    for e in range(epochs) :

        for c in range(chunks):

            j=0

            batch = train_set[int(c*batch_size) : int((c+1)*batch_size)]

            for (image,label) in batch :

                first_temp = np.dot(first_weight, image)
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

                outputs[j] = findMax(third_layer)

                if (outputs[j] != label[j]) :
                    costs[i] += pow((outputs[j] - label[j]),2)
                # print(costs[i])
                
                if (outputs[j] == label[j]) :
                    counter += 1
                j =+ 1

            for i in range (second_dim):
                for j in range (first_dim):
                    first_grad_w[i][j] +=  2 * (first_layer[i]) * derivative_sigmoid(first_layer[i]) * image[j]
                    first_grad_b[i][0] +=  2 * (first_layer[i]) * derivative_sigmoid(first_layer[i])

            for i in range (second_dim):
                for j in range (first_dim):
                    first_weight[i][j] =  first_weight[i][j] - alpha*(first_grad_w[i][j] / batch_size)
                    first_bias[i][0] = first_bias[i][0] - alpha*(first_grad_b[i][0] / batch_size)
            
            for i in range (third_dim):
                for j in range (second_dim):
                    second_grad_w[i][j] +=  2 * (second_layer[i]) * derivative_sigmoid(second_layer[i]) * first_layer[j]
                    second_grad_b[i][0] +=  2 * (second_layer[i]) * derivative_sigmoid(second_layer[i])

            for i in range (third_dim):
                for j in range (second_dim):
                    second_weight[i][j] =  second_weight[i][j] - alpha*(second_grad_w[i][j] / batch_size)
                    second_bias[i][0] = second_bias[i][0] - alpha*(second_grad_b[i][0] / batch_size)

            for i in range (fourth_dim):
                for j in range (third_dim):
                    third_grad_w[i][j] +=  2 * (third_layer[i]) * derivative_sigmoid(third_layer[i]) * second_layer[j]
                    third_grad_b[i][0] +=  2 * (third_layer[i]) * derivative_sigmoid(third_layer[i])

            for i in range (fourth_dim):
                for j in range (third_dim):
                    third_weight[i][j] =  third_weight[i][j] - alpha*(third_grad_w[i][j] / batch_size)
                    third_bias[i][0] = third_bias[i][0] - alpha*(third_grad_b[i][0] / batch_size)

    print("Accuracy is : {}%" .format(counter/10))

    end=time.time()

    print("Learning time is : {}" .format(end - start))

    for i in range(len(costs)):
        costs[i] = costs[i]/batch_size
    plt.plot(costs)
    plt.show()



def read_train_set():

    train_images_file = open('train-images-idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)

    train_labels_file = open('train-labels-idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    train_set = []


    for n in range(DEFINE_NUMBER):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256
        
        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        train_set.append((image, label))
    
    return train_set

def read_test_set():
    test_images_file = open('t10k-images-idx3-ubyte', 'rb')
    test_images_file.seek(4)

    test_labels_file = open('t10k-labels-idx1-ubyte', 'rb')
    test_labels_file.seek(8)

    num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
    test_images_file.seek(16)

    test_set = []
    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256
        
        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1
        
        test_set.append((image, label))

    return test_set

backPropagation()