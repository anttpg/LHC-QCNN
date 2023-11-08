from constants import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np



# Initialize figure grid
fig = plt.figure(figsize=(20, 10))  # Consider maintaining a ratio where left side is ~30% of total width
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])  # Split into 1:2 ratio

# Left subplot for the callback graph that takes about 30% of the width
objective_ax = plt.subplot(gs[0])

# Right subplot area for additional subplots with a nested GridSpec
right_subplot_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1])

# Upper right subplot area for example images using another nested GridSpec
example_images_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=right_subplot_gs[0])
example_image_axs = np.array([[plt.subplot(example_images_gs[i, j]) for j in range(2)] for i in range(2)])

# Lower right subplot area for prediction results using the same approach
results_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=right_subplot_gs[1])
results_axs = np.array([[plt.subplot(results_gs[i, j]) for j in range(2)] for i in range(2)])

# Turn on interactive mode
plt.ion()


objective_func_vals = []

# Define the callback function for updating the objective function plot
def callback_graph(weights, obj_func_eval):
    global objective_func_vals
    
    # Append the new objective function evaluation to our list and update the plot
    objective_func_vals.append(obj_func_eval)
    objective_ax.clear()
    objective_ax.plot(objective_func_vals, 'r-')  # Plot the new data
    objective_ax.set_title("Objective function value against iteration")
    objective_ax.set_xlabel("Iteration")
    objective_ax.set_ylabel("Objective function value")
    plt.draw()
    plt.pause(0.1)

    # Save weights to a file after updating the graph
    with open("weights.txt", "w") as file:
        file.write(str(NUM_QUBITS) + "\n")
        for weight in weights:
            file.write(f"{weight}\n")





# Define the update_plots function
def update_plots(train_images, train_labels, test_images, test_labels, y_predict=None):
    # Clear previous images and predictions
    for ax_array in [example_image_axs, results_axs]:
        for sub_ax in ax_array.flatten():
            sub_ax.clear()
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])

    # Plot example training images and their labels on the upper right subplots
    for i, ax_row in enumerate(example_image_axs):
        for j, ax in enumerate(ax_row):
            img_index = i * 2 + j  # Calculate the index of the image to display
            if img_index < len(train_images):  # Check if there is an image to display
                ax.imshow(train_images[img_index].reshape(SHAPE_Y, SHAPE_X), aspect="equal")
                label = "Ant-Particle" if train_labels[img_index] == 0 else "Ben-Particle"
                ax.set_title(f"Label: {label}")

    # Update only if predictions are provided and test_images is not None
    if y_predict is not None and test_images is not None:
        # Plot prediction results on the lower right subplots
        for i, ax_row in enumerate(results_axs):
            for j, ax in enumerate(ax_row):
                img_index = i * 2 + j  # Calculate the index of the image to display
                if img_index < len(test_images):  # Check if there is an image to display
                    ax.imshow(test_images[img_index].reshape(SHAPE_Y, SHAPE_X), aspect="equal")
                    label = "Ant-Particle" if y_predict[img_index] == 0 else "Ben-Particle"
                    ax.set_title(f"Predicted: {label}")

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.draw()  # Redraw the updated figure
    plt.pause(0.1)  # Brief pause to update the UI



# Function to display the results at the end, BLOCKS THE WINDOW FROM AUTO CLOSING.
def display_results():
    plt.ioff()  # Turn off interactive mode to prevent closing
    plt.show()  # Show the final plot in a blocking way
