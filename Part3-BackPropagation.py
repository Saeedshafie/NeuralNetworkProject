import numpy as np
import matplotlib.pyplot as plt
import time

INPUT_LAYER_SIZE = 784
HIDDEN_LAYER_SIZE = 16
OUTPUT_LAYER_SIZE = 10
EPOCHS = 20
BATCH_SIZE = 10
LEARNING_RATE = 1

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

# Third step
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_of_sigmoid(x):
    sig_x = sigmoid(x)
    return (1 - sig_x) * sig_x


def random_matrix_generator(rows, cols):
    return np.random.randn(rows, cols)


def zero_matrix_generator(rows, cols):
    return np.zeros((rows, cols))


def feed_forward(input_data, weights, biases):
    input_data = input_data.reshape(-1, 1)
    w0, w1, w2 = weights
    b0, b1, b2 = biases

    z1 = w0 @ input_data + b0
    a1 = sigmoid(z1).reshape(-1, 1)

    z2 = w1 @ a1 + b1
    a2 = sigmoid(z2).reshape(-1, 1)

    z3 = w2 @ a2 + b2
    a3 = sigmoid(z3).reshape(-1, 1)

    return (a1, a2, a3), (z1, z2, z3)

def plot(x, y, x_label, y_label):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def cost_function(actual_data, desired_data):
    return np.sum(np.square(actual_data - desired_data))


def create_gradient_matrices():
    grad_w0 = zero_matrix_generator(HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE)
    grad_w1 = zero_matrix_generator(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
    grad_w2 = zero_matrix_generator(OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE)

    grad_a1 = zero_matrix_generator(HIDDEN_LAYER_SIZE, 1)
    grad_a2 = zero_matrix_generator(HIDDEN_LAYER_SIZE, 1)

    grad_b0 = zero_matrix_generator(HIDDEN_LAYER_SIZE, 1)
    grad_b1 = zero_matrix_generator(HIDDEN_LAYER_SIZE, 1)
    grad_b2 = zero_matrix_generator(OUTPUT_LAYER_SIZE, 1)

    grad_w = np.array([grad_w0, grad_w1, grad_w2])
    grad_b = np.array([grad_b0, grad_b1, grad_b2])
    grad_a = np.array([grad_a1, grad_a2])

    return grad_w, grad_b, grad_a


def backpropagation(data, data_label, w, z, a):
    data = data.reshape(-1, 1)
    data_label = data_label.reshape(-1, 1)

    grad_w, grad_b, grad_a = create_gradient_matrices()

    d_sig_z0 = derivative_of_sigmoid(z[0]).reshape(-1, 1)
    d_sig_z1 = derivative_of_sigmoid(z[1]).reshape(-1, 1)
    d_sig_z2 = derivative_of_sigmoid(z[2]).reshape(-1, 1)

    # Calculating gradients of parameters in the last layer
    for i in range(OUTPUT_LAYER_SIZE):
        for j in range(HIDDEN_LAYER_SIZE):
            grad_w[2][i][j] = 2 * (a[2][i] - data_label[i]) * d_sig_z2[i] * a[1][j]

    for i in range(OUTPUT_LAYER_SIZE):
        grad_b[2][i] = 2 * (a[2][i] - data_label[i]) * d_sig_z2[i]

    for i in range(HIDDEN_LAYER_SIZE):
        for j in range(OUTPUT_LAYER_SIZE):
            grad_a[1][i] += 2 * (a[2][j] - data_label[j]) * d_sig_z2[j] * w[2][j][i]

    # Calculating gradients of parameters in the second hidden layer
    for i in range(HIDDEN_LAYER_SIZE):
        for j in range(HIDDEN_LAYER_SIZE):
            grad_w[1][i][j] = grad_a[1][i] * d_sig_z1[i] * a[0][j]

    for i in range(HIDDEN_LAYER_SIZE):
        grad_b[1][i] = grad_a[1][i] * d_sig_z1[i]

    for i in range(HIDDEN_LAYER_SIZE):
        for j in range(OUTPUT_LAYER_SIZE):
            grad_a[0][i] = w[1][j][i] * d_sig_z1[j] * grad_a[1][j]

    # Calculating gradients of parameters in the first hidden layer
    for i in range(HIDDEN_LAYER_SIZE):
        for j in range(INPUT_LAYER_SIZE):
            grad_w[0][i][j] = grad_a[0][i] * d_sig_z0[i] * data[j]

    for i in range(HIDDEN_LAYER_SIZE):
        grad_b[0][i] = grad_a[0][i] * d_sig_z0[i]

    return grad_w, grad_b, grad_a


def train_network(input_data, input_data_labels, weights, biases, backpropagation_method=backpropagation):
    w0, w1, w2 = weights
    b0, b1, b2 = biases

    import time
    start_time = time.time()

    avg_cost = []
    for epoch in range(EPOCHS):
        batches, batch_labels = input_data, input_data_labels
        batches = [batches[i:i + BATCH_SIZE] for i in range(0, len(batches), BATCH_SIZE)]
        batch_labels = [batch_labels[i:i + BATCH_SIZE] for i in range(0, len(batch_labels), BATCH_SIZE)]

        cost = 0
        correct_guesses = 0
        for batch, batch_label in zip(batches, batch_labels):
            grad_w, grad_b, grad_a = create_gradient_matrices()

            for image, image_label in zip(batch, batch_label):
                a, z = feed_forward(image, (w0, w1, w2), (b0, b1, b2))
                w = [w0, w1, w2]

                actual_data = np.argmax(a[-1])
                desired_data = np.argmax(image_label)
                cost += cost_function(a[-1], image_label.reshape(-1, 1))

                # Checking if the network's output is valid or not
                if actual_data == desired_data:
                    correct_guesses += 1

                grad_w_temp, grad_b_temp, grad_a_temp = backpropagation_method(image, image_label, w, z, a)

                grad_w += grad_w_temp
                grad_b += grad_b_temp
                grad_a += grad_a_temp

            # Updating the network's weights based on the average gradients
            w0 -= LEARNING_RATE * grad_w[0] / BATCH_SIZE
            w1 -= LEARNING_RATE * grad_w[1] / BATCH_SIZE
            w2 -= LEARNING_RATE * grad_w[2] / BATCH_SIZE

            # Updating the network's biases based on the average gradients
            b0 -= LEARNING_RATE * grad_b[0] / BATCH_SIZE
            b1 -= LEARNING_RATE * grad_b[1] / BATCH_SIZE
            b2 -= LEARNING_RATE * grad_b[2] / BATCH_SIZE

        avg_cost.append(cost / len(input_data))
        print(f'Epoch{epoch + 1}: {correct_guesses}/{len(input_data)} = {correct_guesses/len(input_data) * 100}%')

    print('\nTraining is finished...')
    print(f'Time taken for training: {time.time() - start_time} seconds')

    # Returning the trained parameters and also the average cost
    return (w0, w1, w2), (b0, b1, b2), avg_cost

EPOCHS = 10
BATCH_SIZE = 16

weights = (random_matrix_generator(HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE),
               random_matrix_generator(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
               random_matrix_generator(OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE))
biases = (zero_matrix_generator(HIDDEN_LAYER_SIZE, 1),
          zero_matrix_generator(HIDDEN_LAYER_SIZE, 1),
          zero_matrix_generator(OUTPUT_LAYER_SIZE, 1))

weights, biases, avg_costs = train_network(train_set[0][:200],train_set[1][:200],
                                            weights, biases, backpropagation_method=backpropagation)

plot(range(len(avg_costs)), avg_costs, 'epoch', 'loss')