import numpy as np
import matplotlib.pyplot as plt
import time


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

# Third Part

# Activator Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Activator Function Derivation
def sigmoidDerivative(x):
    return np.multiply(x, np.subtract(1, x))

# Plot Relsult in a External Png
#def showCost(epochCount, costs):
    plt.title("Part" + str(3) + "Chart")
    x = np.arange(0,epochCount)
    plt.plot(x, costs)
    plt.savefig(f"/Users/saeei/Desktop/Part3Result.png")

# Randomizing The Weights with Zero Bias
w1 = np.random.normal(loc=0, scale=1, size=(16, 28*28)) # Refers to firts Layers Weight
w2 = np.random.normal(loc=0, scale=1, size=(16, 16)) # Refers to Second Layers Weight
w3 = np.random.normal(loc=0, scale=1, size=(10, 16)) # Refers to Third Layers Weight

b1 = np.zeros((16,1)) # Refers to First Layers Bias
b2 = np.zeros((16,1)) # Refers to Second Layers Bias
b3 = np.zeros((10,1)) # Refers to Third Layers Bias

# Given Values in Discription for This Part
batchSize = 10
learningRate = 1
epochCount = 20

costs = []

# Start Training Our Model
for i in range(epochCount):


    startTime = time.time() # Define Start time to Count the Length of Training

    accuracy = 0
    cost = 0

    np.random.shuffle(train_set)
    b_count = int(100 / batchSize)

    for i in range(b_count):
        gw1 = np.zeros((16, 28*28))
        gw2 = np.zeros((16, 16))
        gw3 = np.zeros((10, 16))
        gb1 = np.zeros((16, 1))
        gb2 = np.zeros((16, 1))
        gb3 = np.zeros((10, 1))
        ga2 = np.zeros((16, 1))
        ga1 = np.zeros((16, 1))

        for w in range(batchSize):
            element_num = i * batchSize + w
            main_mtx = np.asarray(train_set[element_num][0])
            z1 = w1 @ main_mtx + b1
            mtx2 = sigmoid(z1)
            z2 = w2 @ mtx2 + b2
            mtx3 = sigmoid(z2)
            z3 = w3 @ mtx3 + b3
            f_mtx = sigmoid(z3)

            cost += sum(pow((f_mtx - train_set[element_num][1]), 2))

            y = train_set[element_num][1]

            for j in range(10):
                for k in range(16):
                    gw3[j, k] += mtx3[k, 0] * sigmoidDerivative(z3[j, 0]) * (2 * f_mtx[j, 0] - 2 * y[j, 0])

            gb3 += (2 * (f_mtx - y) * sigmoidDerivative(z3))

            for k in range(16):
                for j in range(10):
                    ga2[k, 0] += w3[j, k] * sigmoidDerivative(z3[j, 0]) * (2 * f_mtx[j, 0] - 2 * y[j, 0])

            for j in range(10):
                for k in range(16):
                    gw2[j, k] += mtx2[k, 0] * sigmoidDerivative(z2[j, 0]) * (2 * mtx3[j, 0] - 2 * y[j, 0])

            gb2 += (ga2 * sigmoidDerivative(z2))

            for k in range(16):
                for j in range(10):
                    ga1[k, 0] += w2[j, k] * sigmoidDerivative(z2[j, 0]) * (2 * mtx3[j, 0] - 2 * y[j, 0])

            for j in range(10):
                for k in range(16):
                    gw1[j, k] += main_mtx[k, 0] * sigmoidDerivative(z1[j, 0]) * (2 * mtx2[j, 0] - 2 * y[j, 0])

            gb1 += (ga1 * sigmoidDerivative(z2))


            max_value = np.max(f_mtx)
            index_max_value = np.argmax(f_mtx)

            if train_set[element_num][1][index_max_value] == 1:
                accuracy += 1

        w1 = w1 - (learningRate * (gw1 / batchSize))
        w2 = w2 - (learningRate * (gw2 / batchSize))
        w3 = w3 - (learningRate * (gw3 / batchSize))

        b1 = b1 - (learningRate * (gb1 / batchSize))
        b2 = b2 - (learningRate * (gb2 / batchSize))
        b3 = b3 - (learningRate * (gb3 / batchSize))

    costs.append(cost/100)


print('\nTraining is finished...')
print(f'Time taken for training: {time.time() - startTime} seconds')
print(f"Accuracy is: {accuracy}%")
#showCost(epochCount, costs)
