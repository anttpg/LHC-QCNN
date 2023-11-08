import numpy as np
import matplotlib.pyplot as plt
from constants import *
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier


def get_initial_points():
    initial_point = []
    first_line = True

    with open("trained_weights.txt", "r") as file:
        for line in file:

            # Check if the weights are for the correct number of qubits
            if (first_line):

                # Raise error if wrong num qubit file given 
                if (USE_TRAINED and int(line) != NUM_QUBITS):
                    raise ValueError(
                        "Cannot load trained weights for " + line + " qubits, into a circuit with " + str(NUM_QUBITS) + " qubits\n. " +
                        "Replace trained_weights with the appropriate size, or run without trained weights."
                        )
                first_line = False

            # Otherwise read out the weights
            else:
                weight = float(line.strip())
                initial_point.append(weight)

    return initial_point


def create_nn(qnn, callback, initial_point):
    return NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=MAX_ITER),
        initial_point=initial_point,
        callback=callback,
    )


def train_nn(train_images, train_labels, classifier):
    images = []
    for image in train_images:
        images.append(image[0])

    # Train the neural network on our training data
    x = np.asarray(images)
    y = np.asarray(train_labels)

    plt.rcParams["figure.figsize"] = (12, 6)

    classifier.fit(x, y)

    print(f"Accuracy from the training data : {np.round(100 * classifier.score(x, y), 2)}%")


def get_results(test_images, test_labels, classifier):
    tests = []
    for im in test_images:
        tests.append(im[0])

    y_predict = classifier.predict(tests)
    x = np.asarray(tests)
    y = np.asarray(test_labels)
    print(f"Accuracy from the test data : {np.round(100 * classifier.score(x, y), 2)}%")


    return y_predict