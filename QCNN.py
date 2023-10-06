


# Throughout this tutorial, we discuss a Quantum Convolutional Neural Network (QCNN), first proposed by Cong et. al. [1]. We implement such a QCNN on Qiskit by modeling both the convolutional layers and pooling layers using a quantum circuit. After building such a network, we train it to differentiate horizontal and vertical lines from a pixelated image. The following tutorial is thus divided accordingly;
# 
# 1. Differences between a QCNN and CCNN
# 2. Components of a QCNN
# 3. Data Generation
# 4. Building a QCNN
# 5. Training our QCNN
# 6. Testing our QCNN
# 7. References
# 
# We first begin by importing the libraries and packages we will need for this tutorial.


import json
import matplotlib.pyplot as plt
import numpy as np
import os
from constants import *
from PIL import Image
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split

algorithm_globals.random_seed = 12345


# We now define a two qubit unitary as defined in [3]
def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


# Let's draw this circuit and see what it looks like
params = ParameterVector("θ", length=3)
circuit = conv_circuit(params)
circuit.draw("mpl")




def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

#This is a test example circuit
circuit = conv_layer(4, "θ")
circuit.decompose().draw("mpl")


# ### 2.2 Pooling Layer
def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target


params = ParameterVector("θ", length=3)
circuit = pool_circuit(params)
circuit.draw("mpl")



def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


#This is to show a test example
sources = [0, 1]
sinks = [2, 3]
circuit = pool_layer(sources, sinks, "θ")
circuit.decompose().draw("mpl")


# In this particular example, we reduce the dimensionality of our four qubit circuit to the last two qubits, i.e. the last two qubits in this particular example. These qubits are then used in the next layer, while the first two are neglected for the remainder of the QCNN.


# ## 3. Data Generation
import numpy as np
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



# In addition, we want the ability to convert a regular image into a quantum dataset. 
# Here, we can use a preset image, compress it, and convert it onto a 0-pi scale
import cv2

def convert_dataset(SHAPE_X, SHAPE_Y):
    images = []
    labels = []

    path = 'cats_dogs_light/test'
    
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
        
        # Normalize the image to be values between 0 and pi
        image = (image / 255.0) * math.pi
        
        # Add image to list
        images.append(image)
        
        # Add label to list
        if "dog" in file:
            labels.append(1) # Dogs are labeled 1
        else:
            labels.append(0) # Cats are labeled 0
    
    return np.array(images), np.array(labels)

# Example usage:
images, labels = convert_dataset(SHAPE_X, SHAPE_Y)



# Let's now create our dataset below and split it into our test and training datasets.


#images, labels = generate_dataset(50)
images, labels = convert_dataset(SHAPE_X, SHAPE_Y)
# Uncomment this line if you want to use predefined images

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.3
)



# Let's see some examples in our dataset


from constants import SHAPE_X, SHAPE_Y

# Makes a 2x2 subplot image, and passes as args to each subplot that they should not have any ticks (Marks on the image)
fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"xticks": [], "yticks": []})
for i in range(4): # For each subplot
    ax[i // 2, i % 2].imshow( # Show 2 by 2 plots. 
        train_images[i].reshape(SHAPE_Y, SHAPE_X),  # Change back to NxN matrix
        aspect="equal",
    )
plt.subplots_adjust(wspace=0.1, hspace=0.025)


# As we can see each image contains either a vertical or horizontal line, that the QCNN will learn how to differentiate. Now that we have built our dataset, it is time to discuss the components of the QCNN and build our model.


# ## 4. Modeling our QCNN
#This feature map is an example of how the ZFeatureMap
feature_map = ZFeatureMap(NUM_QUBITS)
#feature_map.decompose().draw("mpl")


# ## 5. Training our QCNN
def create_ansatz(N):
    # Initialize Quantum Circuit
    ansatz = QuantumCircuit(N, name="Ansatz")
    
    curr_qubits = list(range(N))
    layer = 1
    
    while len(curr_qubits) > 1:  # Repeat until only 1 qubit remains
        num_qubits = len(curr_qubits)

        # Generating arrays for layers
        conv_qubits = curr_qubits
        pool_sources = curr_qubits[0:num_qubits//2]
        pool_sinks = curr_qubits[num_qubits//2:num_qubits]
        
        # Naming layers
        conv_name = f"c{layer}"
        pool_name = f"p{layer}"
        
        # Convolutional Layer
        ansatz.compose(conv_layer(num_qubits, conv_name), conv_qubits, inplace=True)

        # Pooling Layer
        ansatz.compose(pool_layer(pool_sources, pool_sinks, pool_name), curr_qubits, inplace=True)
        
        # Updating curr_qubits for the next iteration
        curr_qubits = pool_sources  # Here assuming pool keeps the source qubits
        layer += 1  # Move to the next layer

    
    #display(ansatz.draw("mpl"))
    return ansatz




def create_qnn(N):
    feature_map = ZFeatureMap(N)
    ansatz = create_ansatz(N)

    # Combining the feature map and ansatz
    circuit = QuantumCircuit(N)
    circuit.compose(feature_map, range(N), inplace=True)
    circuit.compose(ansatz, range(N), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (N - 1), 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

    return qnn

# Usage
qnn = create_qnn(NUM_QUBITS)



# We will also define a callback function to use when training our model. This allows us to view and plot the loss function per each iteration in our training process.


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()


# In this example, we will use the COBYLA optimizer to train our classifier, which is a numerical optimization method commonly used for classification machine learning algorithms.
# 
# We then place the the callback function, optimizer and operator of our QCNN created above into Qiskit's built in Neural Network Classifier, which we can then use to train our model.
# 
# Since model training may take a long time we have already pre-trained the model for some iterations and saved the pre-trained weights. We'll continue training from that point by setting `initial_point` to a vector of pre-trained weights.


initial_point = [1.930057091422052, 0.2829424508139703, 0.35555636265939633, 0.1750006532903061, 0.3002103666790018, 0.6641911912373437, 1.3310981300850042, 0.5022717197547227, 0.44912874128880675, 0.40236963192983266, 0.3459537084665159, 0.9786311288435154, 0.48716712269991697, -0.007081389738930712, 0.21570815199311827, 0.07334182375267477, 0.6907887498355103, 0.21771166428570735, 1.087665977608006, 1.2571463700739218, 1.0866597360102666, 2.126145551821481, 0.8914518096731741, 1.5053260036617715, 0.44798876926441555, 0.9498701675467225, 0.15490304396579338, 0.1338674031994701, -0.6938374500039391, 0.029396385425104116, -0.09785818314088227, -0.31198441382224246, 0.20004568516690807, 1.848494069662786, -0.028371899054628447, -0.15229494459622284, 0.7653870524298326, 0.6881492316484289, 0.6759011152318357, 1.6028387103546868, 0.47711915171800057, -0.26162053028790294, -0.12898443497061718, 0.5281303751714184, 0.4957555866394333, 1.6095784010055925, 0.5685823964468215, 1.2812276175594062, 0.3032325725579015, 1.4291081956286258, 0.7081163438891277, 1.8291375321912147, -0.11047287562207528, 0.2751308409529747, 0.2834764252747557, 0.29668607404725605, 0.008300790063532154, 0.6707732056265118, 0.5325267632509095, 0.7240676576317691, 0.08123934531343553, -0.0038536767244725153, -0.1001165849018211]

classifier = NeuralNetworkClassifier(
    qnn,
    optimizer=COBYLA(maxiter=200),  # Set max iterations here
    callback=callback_graph,
)


# After creating this classifier, we can train our QCNN using our training dataset and each image's corresponding label. Because we previously defined the callback function, we plot the overall loss of our system per iteration.
# 
# It may take some time to train the QCNN so be patient!


x = np.asarray(train_images)
y = np.asarray(train_labels)

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)
np.reshape(x, )
classifier.fit(x, y)

# score classifier
print(f"Accuracy from the train data : {np.round(100 * classifier.score(x, y), 2)}%")


# As we can see from above, the QCNN converges slowly, hence our `initial_point` was already close to an optimal solution. The next step is to determine whether our QCNN can classify data seen in our test image data set.


# ## 6. Testing our QCNN


# After building and training our dataset we now test whether our QCNN can predict images that are not from our test data set.


y_predict = classifier.predict(test_images)
x = np.asarray(test_images)
y = np.asarray(test_labels)
print(f"Accuracy from the test data : {np.round(100 * classifier.score(x, y), 2)}%")

# Let's see some examples in our dataset
fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"xticks": [], "yticks": []})
for i in range(0, 4):
    ax[i // 2, i % 2].imshow(test_images[i].reshape(2, 4), aspect="equal")
    if y_predict[i] == -1:
        ax[i // 2, i % 2].set_title("The QCNN predicts this is a Horizontal Line")
    if y_predict[i] == +1:
        ax[i // 2, i % 2].set_title("The QCNN predicts this is a Vertical Line")
plt.subplots_adjust(wspace=0.1, hspace=0.5)


