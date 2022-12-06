import numpy as np
import matplotlib.pyplot as plt

# First Part of the Code is Exactly the Same as Part1 for Project (## This is Done to Prevent Multilple Runnings in Case of Lack of RAM ##)
# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


# Reading The Train Set
train_images_file = open('train-images.idx3-ubyte', 'rb')
train_images_file.seek(4)
num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
train_images_file.seek(16)

train_labels_file = open('train-labels.idx1-ubyte', 'rb')
train_labels_file.seek(8)

train_set = []
for n in range(num_of_train_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

    label_value = int.from_bytes(train_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1

    train_set.append((image, label))


# Reading The Test Set
test_images_file = open('t10k-images.idx3-ubyte', 'rb')
test_images_file.seek(4)

test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
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


# Second step

# Activator Function is Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Randomizing The Weights with Zero Bias
w1 = np.random.normal(loc=0, scale=1, size=(16, 28*28)) # Refers to firts Layers Weight
w2 = np.random.normal(loc=0, scale=1, size=(16, 16)) # Refers to Second Layers Weight
w3 = np.random.normal(loc=0, scale=1, size=(10, 16)) # Refers to Third Layers Weight

b1 = np.zeros((16,1)) # Refers to First Layers Bias
b2 = np.zeros((16,1)) # Refers to Second Layers Bias
b3 = np.zeros((10,1)) # Refers to Third Layers Bias


accuracy = 0 ## Asked Output ##
#numberOfSamples = 100


# Calculate output of model for first 100 Picture inputs and check its Accuracy
for i in range(100):
    modelInput = train_set[i][0]
    f1 = sigmoid(w1 @ modelInput + b1) ## @ Operator is Used for matrix multiplication
    f2 = sigmoid(w2 @ f1 + b2)
    modelOutput = sigmoid(w3 @ f2 + b3)

    maxValue = np.max(modelOutput) # Finding Models Max Value
    indexMaxValue = np.argmax(modelOutput) # Finding the Indice of the MaxValue
    if train_set[i][1][indexMaxValue] == 1:
        accuracy += 1
        #print(i)


print(f"The Accuracy of the Neural Network for the first 100 images in training set is: {accuracy}%")