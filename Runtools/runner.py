import os
from model import run

class Runner:
    
    class Results:
        def __init__(self):
            self.info1 = 1
            self.info2 = 'time'
            print('initialised')

    def __init__(self, params, data):
        self.results = None
        self.params = params
        self.data = data
       
        return None # Controller will read ? do what ? interface how ?


    def start(self, train_data, test_data):
        circuit = create_circuit()
        run(self.params, circuit) # Updates self.results inside the model...'
        return None

    # Custom function to build the circuit, eventually will be modular 
    def create_circuit(self):
        n_qubits = self.params.n_qubits  
        num_layers = self.params.num_layers  

        # BUILD CIRCUIT
        qc_template = QuantumCircuit(n_qubits)

        par_inputs = ParameterVector("input", n_qubits)
        par_weights = ParameterVector("weights", num_layers * n_qubits)

        for i in range(n_qubits):
            qc_template.rx(par_inputs[i], i)

        for i in range(n_qubits):
            qc_template.ry(par_weights[i], i)

        for i in range(n_qubits):
            qc_template.cx(i, (i+1)%n_qubits)

        for i in range(n_qubits):
            qc_template.rz(par_weights[i+3], i)

        for i in range(n_qubits):
            qc_template.cx(i, (i+1)%n_qubits)

        for i in range(n_qubits):
            qc_template.ry(par_weights[i+6], i)

        for i in range(n_qubits):
            qc_template.cx(i, (i+1)%n_qubits)

        for i in range(n_qubits):
            qc_template.rz(par_weights[i+9], i)

        for i in range(n_qubits):
            qc_template.cx(i, (i+1)%n_qubits)

        for i in range(n_qubits):
            qc_template.rz(par_weights[i+12], i)

        for i in range(n_qubits):
            qc_template.cx(i, (i+1)%n_qubits)


        # qc_template.measure_all()
        obs = SparsePauliOp("XXI")

        return qc_template