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


# Fourth step
# Sigmoid as the Activator Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# A function to calculate the Derivative of the Sigmoid function
def sigmoidDerivation(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Result Function to Create the Chart and Also is Gonna Be Used in the Later Part!
def show_cost(stepNumber, epochCount, costs):
    plt.title("Step " + str(stepNumber) + ": You can View the Chart below")
    x = np.arange(0,epochCount)
    plt.plot(x, costs)
    plt.savefig(f"Chart for Part{stepNumber}.png")

#import time
#start_time = time.time()
# Allocation and Initialize Weight Matrices and Biase Vector Values for Each Layer
w1 = np.random.normal(loc=0, scale=1, size=(16, 28*28))
w2 = np.random.normal(loc=0, scale=1, size=(16, 16))
w3 = np.random.normal(loc=0, scale=1, size=(10, 16))

b1 = np.zeros((16,1))
b2 = np.zeros((16,1))
b3 = np.zeros((10,1))

batchSize = 50
learningRate = 1
epochCount = 5

costs = []
for i in range(epochCount):

    accuracy = 0
    cost = 0

    b_size = int(60000 / batchSize)

    for i in range(b_size):
        gradientW1 = np.zeros((16, 784))
        gradientW2 = np.zeros((16, 16))
        gradientW3 = np.zeros((10, 16))
        gradientB1 = np.zeros((16, 1))
        gradientB2 = np.zeros((16, 1))
        gradientB3 = np.zeros((10, 1))
        gradientA2 = np.zeros((16, 1))
        gradientA1 = np.zeros((16, 1))

        for r in range(batchSize):
            elementNum = i * batchSize + r
            modelInput = np.asarray(train_set[elementNum][0])
            temp1 = w1 @ modelInput + b1
            f1 = sigmoid(temp1)
            temp2 = w2 @ f1 + b2
            f2 = sigmoid(temp2)
            temp3 = w3 @ f2 + b3
            modelOutput = sigmoid(temp3)

            cost += sum(pow((modelOutput - train_set[elementNum][1]), 2))

            y = train_set[elementNum][1]

            gradientW3 += (2 * (modelOutput - y) * sigmoidDerivation(temp3)) @ np.transpose(f2)
            gradientB3 += (2 * (modelOutput - y) * sigmoidDerivation(temp3))
            gradientA2 = np.transpose(w3) @ (2 * (modelOutput - y) * sigmoidDerivation(temp3))
            gradientW2 += (gradientA2 * sigmoidDerivation(temp2)) @ np.transpose(f1)
            gradientB2 += (gradientA2 * sigmoidDerivation(temp2))
            gradientA1 = np.transpose(w2) @ (gradientA2 * sigmoidDerivation(temp2))
            gradientW1 += (gradientA1 * sigmoidDerivation(temp1) @ np.transpose(modelInput))
            gradientB1 += (gradientA1 * sigmoidDerivation(temp1))

            maxValue = np.max(modelOutput)
            indexMaxValue = np.argmax(modelOutput)

            if train_set[elementNum][1][indexMaxValue] == 1:
                accuracy += 1

        w1 = w1 - (learningRate * (gradientW1 / batchSize))
        w2 = w2 - (learningRate * (gradientW2 / batchSize))
        w3 = w3 - (learningRate * (gradientW3 / batchSize))

        b1 = b1 - (learningRate * (gradientB1 / batchSize))
        b2 = b2 - (learningRate * (gradientB2 / batchSize))
        b3 = b3 - (learningRate * (gradientB3 / batchSize))


    cost /= 60000
    costs.append(cost)

print(f"Accuracy is: {(accuracy/60000 * 100)}%")
show_cost(5, epochCount, costs)