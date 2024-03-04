import json
import random
import os
from controller import Controller
#from multiprocessing import Pool TODO Implement multiprocessing 

# This is a temporary function for creating the parameter configurations
# Once the interface is implemented, we can do this a different way
# I think json is probably the best way to do this generally
def create_param_configs():
    param_dict = {}

    for i in range(5):
        param_dict[i] = {
            "n_qubits": 3,
            "n_layers": 5,
            "batch_size": 2,
            "n_epochs": 1,
            "train_data_size": 80,
            "valid_data_size": 40,
            "test_data_size": 80,
            "is_local_simulator": True,
            "spsa_alpha": 0.5 + random.uniform(-0.015, 0.015),
            "spsa_gamma": 0.101 + random.uniform(-0.01, 0.01),
            "spsa_c": 0.2 + random.uniform(-0.01, 0.015),
            "spsa_A": 2 + random.uniform(-0.025, 0.025),
            "spsa_a1": 0.2 + random.uniform(-0.01, 0.015),
            "use_pca": False,
            "seed": 123,
            "training_feature_keys": [
                "f_pt4l",
                "f_eta4l",
                "f_mass4l"
            ],
            "num_features": 3,
            "obs": "XXI"
        }


    for i in range(5, 10):
        param_dict[i] = {
            "n_qubits": 3,
            "n_layers": 5,
            "batch_size": 2,
            "n_epochs": 1,
            "train_data_size": 80,
            "valid_data_size": 40,
            "test_data_size": 80,
            "is_local_simulator": True,
            "spsa_alpha": 0.5 + random.uniform(-0.015, 0.015),
            "spsa_gamma": 0.101 + random.uniform(-0.01, 0.01),
            "spsa_c": 0.2 + random.uniform(-0.01, 0.015),
            "spsa_A": 2 + random.uniform(-0.025, 0.025),
            "spsa_a1": 0.2 + random.uniform(-0.01, 0.015),
            "use_pca": False,
            "seed": 123,
            "training_feature_keys": [
                "f_lept3_pt",
                "f_lept4_pt",
                "f_angle_costheta2"
            ],
            "num_features": 3,
            "obs": "XXI"
        }

    with open("params.json", "w") as f:
        json.dump({run_id: params for run_id, params in param_dict.items()}, f, indent=4)





def main():
    create_param_configs()
    # database = Database() # Feed reference to controller 1
    c = Controller()

    with open("params.json", "r") as f:
        param_dict = json.load(f)

        # Create a runner for each set of parameters in the json file
        for run_id, params in param_dict.items():
            c.create_runner(params, run_id)

    c.run_all()

    # It is fine to do this because the params are saved in the output files
    os.remove("params.json")
    
    # test_c.create_runner(None, "1") # Give specific hyperparameters, otherwise use some defaults 2
    # test_c.create_runner(None, "2")
    # test_c.create_runner(None, "3")
    # test_c.run_all()


if __name__ == "__main__":
    main()