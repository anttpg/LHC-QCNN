from model import run
from qiskit import QuantumCircuit

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


    def start(self):
        circuit = self.create_circuit()
        run(self.params, self.data, circuit) # Updates self.results inside the model...'
        return None # Return results when nessicary

    # Custom function to build the circuit, eventually will be modular 
    def create_circuit(self):
        n_qubits = self.params.n_qubits  
        par_inputs = self.params.par_inputs
        par_weights = self.params.par_weights

        # BUILD CIRCUIT
        qc_template = QuantumCircuit(n_qubits)


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

        return qc_template