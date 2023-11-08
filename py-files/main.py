from circuit import *
from nn import *
from constants import *
from data import *
from observe import *
import os

from sklearn.model_selection import train_test_split


def main():
    # Change directory to our folder
    os.chdir(DATA_PATH)

    images, labels = convert_dataset()
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3
    )

    plot_images(train_images)

    qnn = create_qnn(NUM_QUBITS)
    classifier = create_nn(qnn, callback_graph, get_initial_points() if USE_INITIAL else None)

    train_nn(train_images, train_labels, classifier)
    get_predictions(test_images, test_labels, classifier)


if __name__ == '__main__':
    main()