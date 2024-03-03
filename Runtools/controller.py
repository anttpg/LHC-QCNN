import os
import platform
from sklearn.model_selection import train_test_split
from collections import deque
# from Runtools.database import *
from database import *
from runner import *
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector

from compileoutputs import *



class Controller:
    def __init__(self):
        self.runner_queue = deque()

    def create_runner(self, data, params, runner_id):
        params = self.Parameters()
        data = Train_Test_Data(params) ## EVENTUALLY, WE WILL REUSE DATA/ PARAMS FROM ELSEWHERE.
        data.plot_datapoints(runner_id)
        data.tts_preprocess(None, None) # FOr now, recreate for each runner
        self.runner_queue.append( (Runner(params, data), runner_id) )

    def run_one(self):
        runner, runner_id = self.runner_queue.popleft() 
        #When results recieved, send to database as well
        rval = runner.start()
        compile_run_plots(runner_id)            
        # Once parameters are modular, the self.Parameters call can be changed to fit whatever we do, but for now, need a way to get the params here
        get_output_text(runner_id, self.Parameters())
        return rval


    def run_all(self):
        results = []
        while bool(self.runner_queue): # While not empty
            results.append(self.run_one())
        
        return results 



    class Parameters:
        ## EVENTUALLY MAKE THIS MODULAR 

        def __init__(self):

            self.seed = 123
            # Features to train on
            self.training_feature_keys = [
                "f_mass4l",
                # "f_eta4l",
                "f_Z2mass",
                "f_Z1mass",
            ]

            self.num_features = len(self.training_feature_keys)

            self.save_folder = os.path.join("saved", "model1g3-qiskit-estimator-corrected")

            self.batch_size = 2
            self.n_epochs = 1

            self.use_pca = False

            self.train_data_size = 80
            self.test_data_size = 80
            self.valid_data_size = 40
            self.total_datasize = self.train_data_size + self.test_data_size + self.valid_data_size
            self.half_datasize = self.total_datasize // 2 # 80 signal and 80 backgrounds

            self.is_local_simulator = True

            self.n_qubits = 3
            self.num_layers = 5
            self.obs = SparsePauliOp("XXI")
            self.par_inputs = ParameterVector("input", self.n_qubits)
            self.par_weights = ParameterVector("weights", self.num_layers * self.n_qubits)


            self.spsa_alpha = 0.5
            self.spsa_gamma = 0.101
            self.spsa_c     = 0.2
            self.spsa_A     = 2.
            self.spsa_a1    = 0.2
            self.spsa_a     = self.spsa_a1 * (self.spsa_A + 1) ** self.spsa_alpha

            if platform.system() == "Windows":
                self.signals_folder = "LHC_data\\actual_data\\histos4mu\\signal"
                self.backgrounds_folder = "LHC_data\\actual_data\\histos4mu\\background"
            elif platform.system() == "Darwin":
                self.signals_folder = "./../LHC_data/actual_data/histos4mu/signal"
                self.backgrounds_folder = "./../LHC_data/actual_data/histos4mu/background"

    