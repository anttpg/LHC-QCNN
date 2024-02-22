import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2

num_features = 7
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = EfficientSU2(num_qubits=num_features, reps=3)

# Combine the circuits
full_circuit = feature_map.compose(ansatz)

# Visualize the combined circuit
full_circuit.draw("mpl")
plt.title("Full VQC Circuit")
plt.show()
