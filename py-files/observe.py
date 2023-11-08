from constants import *
from matplotlib import pyplot as plt
from IPython.display import clear_output


objective_func_vals = []

def plot_images(train_images):
    # Makes a 2x2 subplot image, and passes as args to each subplot that they should not have any ticks (Marks on the image)
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={
                        "xticks": [], "yticks": []})
    for i in range(4):  # For each subplot
        ax[i // 2, i % 2].imshow(  # Show 2 by 2 plots.
            train_images[i].reshape(SHAPE_Y, SHAPE_X),  # Change back to NxN matrix
            aspect="equal",
        )
    plt.subplots_adjust(wspace=0.1, hspace=0.025)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    
    # Save weights to a text file
    with open("weights.txt", "w") as file:

        #Write the number of qubits to the start
        file.write(str(NUM_QUBITS))

        #Write each weight to the file
        for weight in weights:
            file.write(str(weight) + "\n")

    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()