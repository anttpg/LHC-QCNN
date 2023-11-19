import numpy as np
from SimpleGates import *

class Circuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        # Initialize the state to |0...0>
        self.state = np.zeros((2**num_qubits, 1))
        self.state[0, 0] = 1
        self.gate_sequence = []

    def apply_gate(self, gate, qubits):
        # Create a full-size gate matrix for the specified qubits
        full_gate = 1
        for qubit in range(self.num_qubits):
            if qubit in qubits:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, Identity())

        self.gate_sequence.append(full_gate)

    def run(self):
        for gate in self.gate_sequence:
            self.state = gate @ self.state

        return self.state
