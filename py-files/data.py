import numpy as np
import cv2
import os
from qiskit.utils import algorithm_globals
from constants import *

algorithm_globals.random_seed = 12345



def generate_dataset(num_images):
    images = []
    labels = []
    hor_array = np.zeros((6, 8))
    ver_array = np.zeros((4, 8))

    # Creates arrays that will represent horizontal and vertical lines
    j = 0
    for i in range(0, 7):
        if i != 3:
            hor_array[j][i] = np.pi / 2
            hor_array[j][i + 1] = np.pi / 2
            j += 1

    j = 0
    for i in range(0, 4):
        ver_array[j][i] = np.pi / 2
        ver_array[j][i + 4] = np.pi / 2
        j += 1

    for n in range(num_images):
        rng = algorithm_globals.random.integers(0, 2)
        # If rng == 0, create a graph where the image is a horizontal line
        if rng == 0:
            labels.append(-1)
            random_image = algorithm_globals.random.integers(0, 6)
            images.append(np.array(hor_array[random_image]))
        elif rng == 1:
            labels.append(1)
            random_image = algorithm_globals.random.integers(0, 4)
            images.append(np.array(ver_array[random_image]))

        # Create noise
        for i in range(8):
            if images[-1][i] == 0:
                images[-1][i] = algorithm_globals.random.uniform(0, np.pi / 4)
    return images, labels



def convert_dataset():
    images = []
    labels = []

    path = DATA_PATH

    # Loop through each image in the dataset
    for file in os.listdir(path):
        # Generate file path
        file_path = os.path.join(path, file)

        # Open the image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Check if image is read successfully
        if image is None:
            print(f"Error: Unable to read {file_path}!")
            continue

        # Resize the image to (SHAPE_X, SHAPE_Y)
        image = cv2.resize(image, (SHAPE_X, SHAPE_Y))
        image = image.reshape(1, NUM_QUBITS)  # RESHAPE BACK TO 128x1

        # Normalize the image to be values between 0 and pi
        image = (image / 255.0) * np.pi

        # Add image to list
        images.append(image)

        # Add label to list
        if LABEL in file:
            labels.append(1)  # Dogs are labeled 1
        else:
            labels.append(0)  # Cats are labeled 0

    return np.array(images), np.array(labels)
