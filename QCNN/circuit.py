from constants import *
from convolution import *
from pooling import *
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap, RealAmplitudes



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
        ansatz.compose(conv_layer(num_qubits, conv_name),
                       conv_qubits, inplace=True)

        # Pooling Layer
        ansatz.compose(pool_layer(pool_sources, pool_sinks,
                       pool_name), curr_qubits, inplace=True)

        # Updating curr_qubits for the next iteration
        curr_qubits = pool_sources  # Here assuming pool keeps the source qubits
        layer += 1  # Move to the next layer

    # display(ansatz.draw("mpl"))
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
