from model import run
from qiskit import QuantumCircuit


class Runner:
    def __init__(self, params, data):
        self.results = None
        self.params = params
        self.data = data
       
        return None # Controller will read ? do what ? interface how ?




    class Results:
        # This class will be used to store the results of the run
        # set_run_data will be called by the run function to update the results
        def __init__(self):
            self.test_labels = None
            self.test_prob = None
            self.test_pred = None

            self.valid_loss = None

            self.cost = None
            self.test_accuracy = None

            self.weights = None

            self.log = None

            self.run_time = None

        def set_run_data(self, labels, prob, pred, valid_loss, cost, test_accuracy, weights, run_time, log):
            self.test_labels = labels
            self.test_prob = prob
            self.test_pred = pred
            self.valid_loss = valid_loss
            self.cost = cost
            self.test_accuracy = test_accuracy
            self.weights = weights
            self.run_time = run_time
            self.log = log





    def start(self):
        circuit = self.create_circuit2()
        # Returns the Results object that was passed to it
        # The object will be updated with the results of the run
        return run(self.params, self.data, circuit, self.Results())





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



    def create_circuit2(self):
        n_qubits = self.params.n_qubits  
        par_inputs = self.params.par_inputs
        par_weights = self.params.par_weights

        # BUILD CIRCUIT
        qc_template = QuantumCircuit(n_qubits)

        for i in range(n_qubits):
            qc_template.rx(par_inputs[i], i)

        for i in range(self.params.num_layers):
            for j in range(n_qubits):
                qc_template.ry(par_weights[self.params.n_par_per_layer*i + self.params.n_rots*j + 0], j)
                qc_template.rz(par_weights[self.params.n_par_per_layer*i + self.params.n_rots*j + 1], j)
                # qc.ry(par_weights[n_par_per_layer*i + n_rots*j + 2], j)
            for j in range(n_qubits-1, 0, -1):
                qc_template.cx(j-1, j)

        if self.params.final_rotation_layer:
            for j in range(n_qubits):
                qc_template.ry(par_weights[self.params.n_par_per_layer*self.params.num_layers + self.params.n_rots*j + 0], j)
                qc_template.rz(par_weights[self.params.n_par_per_layer*self.params.num_layers + self.params.n_rots*j + 1], j)


        return qc_template