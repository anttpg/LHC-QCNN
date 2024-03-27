import os
import platform
from sklearn.model_selection import train_test_split
from collections import deque
# from Runtools.database import *
from database import *
from runner import *
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector



class Controller:
    def __init__(self, database):
        self.runner_queue = deque()
        self.database = database




    def create_runner(self, raw_params):
        # Initialize parameters
        params = self.Parameters(raw_params)
        # Create data object with parameters and run_id (run_id necessary for graphs)
        data = Train_Test_Data(params) ## EVENTUALLY, WE WILL REUSE DATA/ PARAMS FROM ELSEWHERE.
        # Plot datapoints
        data.tts_preprocess(None, None) # For now, recreate for each runner

        circuit_id = self.database.create_circuit_entry(params)

        self.runner_queue.append((Runner(params, data), circuit_id))

        # Return the circuit_id for this specific run
        # Easiest to do this here instead of passing it in since the database will keep track of the unique ids
        # So initialize new row in database for output, then return the id
        return circuit_id




    def run_one(self):
        runner, circuit_id = self.runner_queue.popleft() 
        #When results recieved, send to database as well
        print("\n-------------------------------------------------\n")
        print(f"Running {circuit_id}\n\n")
        rval = runner.start()
        self.database.update_callback(circuit_id, rval)
        return rval





    def run_all(self):
        results = []
        while bool(self.runner_queue): # While not empty
            results.append(self.run_one())
        
        return results 





    class Parameters:
        # Params can just be initialized from the dictionary from the json file
        def __init__(self, raw_params):
            self.seed = raw_params["seed"]
            # Features to train on
            self.training_feature_keys = raw_params["training_feature_keys"]

            self.num_features = raw_params["num_features"]

            self.batch_size = raw_params["batch_size"]
            self.n_epochs = raw_params["n_epochs"]

            self.use_pca = raw_params["use_pca"]

            self.train_data_size = raw_params["train_data_size"]
            self.test_data_size = raw_params["test_data_size"]
            self.valid_data_size = raw_params["valid_data_size"]
            self.total_datasize = self.train_data_size + self.test_data_size + self.valid_data_size
            self.half_datasize = self.total_datasize // 2 # 80 signal and 80 backgrounds

            self.is_local_simulator = raw_params["is_local_simulator"]

            self.n_qubits = raw_params["n_qubits"]
            self.num_layers = raw_params["n_layers"]
            try:
                self.n_rots = raw_params["n_rots"]
                self.final_rotation_layer = raw_params["final_rotation_layer"]
                self.n_par_per_layer = self.n_qubits * self.n_rots
                self.n_params = self.n_rots * self.n_qubits * (self.num_layers + 1) if self.final_rotation_layer else self.n_qubits*self.num_layers*self.n_rots
            except KeyError:
                self.n_params = self.n_qubits * self.num_layers
            # For database
            self.obs_text = raw_params["obs"]
            self.obs = SparsePauliOp(raw_params["obs"])
            self.par_inputs = ParameterVector("input", self.n_qubits)
            self.par_weights = ParameterVector("weights", self.n_params)

            self.spsa_alpha = raw_params["spsa_alpha"] 
            self.spsa_gamma = raw_params["spsa_gamma"] 
            self.spsa_c     = raw_params["spsa_c"] 
            self.spsa_A     = raw_params["spsa_A"]
            self.spsa_a1    = raw_params["spsa_a1"]
            self.spsa_a     = self.spsa_a1 * (self.spsa_A + 1) ** self.spsa_alpha


            if platform.system() == "Windows":
                self.signals_folder = "LHC_data\\actual_data\\histos4mu\\signal"
                self.backgrounds_folder = "LHC_data\\actual_data\\histos4mu\\background"
            elif platform.system() == "Darwin":
                self.signals_folder = "./../LHC_data/actual_data/histos4mu/signal"
                self.backgrounds_folder = "./../LHC_data/actual_data/histos4mu/background"