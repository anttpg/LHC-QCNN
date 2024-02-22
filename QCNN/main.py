from circuit import *
from nn import *
from constants import *
from data import *
from graphs import *
import os
from sklearn.model_selection import train_test_split


def main():
    # Change directory to our folder
    # os.chdir(DATA_PATH)

    images, labels = convert_dataset()
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3
    )


    #Show some example training data
    update_plots(train_images, train_labels, test_images, test_labels, None)

    qnn = create_qnn(NUM_QUBITS)
    classifier = create_nn(qnn, callback_graph, get_initial_points() if USE_TRAINED else None)

    train_nn(train_images, train_labels, classifier)
    results = get_results(test_images, test_labels, classifier)

    # Display the resulting results 
    update_plots(train_images, train_labels, test_images, test_labels, results)
    display_results()

    


if __name__ == '__main__':
    main()