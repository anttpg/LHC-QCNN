import json
import random
import os
import itertools
import time
from controller import Controller
from database import Database

DATABASE_PATH = "./database/database.db"

#from multiprocessing import Pool TODO Implement multiprocessing 

# This is a temporary function for creating the parameter configurations
# Once the interface is implemented, we can do this a different way
# I think json is probably the best way to do this generally
def create_param_configs():
    param_dict = {}
    # USE N FEATURE KEYS GIVEN N QUBITS
    training_feature_keys = ['f_lept3_pt', 'f_lept4_pt', 'f_Z1mass', 'f_angle_costheta2', 'f_pt4l', 'f_eta4l', 'f_jet1_pt', 'f_jet1_e']
    runs_per_permutation = 5
    combinations = list(itertools.combinations(training_feature_keys, 3))

    """
    maxiter (int) – the maximum number of iterations expected to be performed. Used to determine A, if A is not supplied, otherwise ignored.

    alpha (float) – A hyperparameter to calculate ak = a/(A + k + 1)α for each iteration. Its asymptotically optimal value is 1.0.

    gamma (float) – An hyperparameter to calculate ck = c/(k + 1)γ for each iteration. Its asymptotically optimal value is 1/6.

    c (float) – A hyperparameter related to the expected noise. It should be approximately the standard deviation of the expected noise of the cost function.

    A (float) – The stability constant; if not provided, set to be 10% of the maximum number of expected iterations.

    a (float) – A hyperparameter expected to be small in noisy situations, its value could be picked using A, α and ^g0(^θ0). For more details, see Spall (1998b).
    """

    # for i in range(len(combinations)):
        # for j in range(runs_per_permutation):
            # param_dict[j + (i*runs_per_permutation)] = {
    for i in range(3):
        param_dict[i] = {
        "n_qubits": 3,
        "n_layers": 5,
        "batch_size": 2,
        "n_epochs": 1,
        "train_data_size": 80,
        "valid_data_size": 40,
        "test_data_size": 80,
        "is_local_simulator": True,
        "spsa_alpha": 0.5 + random.uniform(-0.015, 0.015) if i == 1 else 0.5,
        "spsa_gamma": 0.101 + random.uniform(-0.01, 0.01) if i == 1 else 0.101,
        "spsa_c": 0.2 + random.uniform(-0.01, 0.015) if i == 1 else 0.2,
        "spsa_A": 2 + random.uniform(-0.025, 0.025) if i == 1 else 2,
        "spsa_a1": 0.2 + random.uniform(-0.01, 0.015) if i == 1 else 0.2,
        "use_pca": False,
        "seed": 123,
        "training_feature_keys": combinations[i if i != 1 else 0],
        "num_features": 3,
        "obs": "XXI"
    }

    with open("params.json", "w") as f:
        json.dump({run_id: params for run_id, params in param_dict.items()}, f, indent=4)


# Only call when you want to optimize the hyperparameters of a network
def optimize_hyperparams(param_dict ):

    return 0


def main():
    create_param_configs()
    database = Database(DATABASE_PATH)
    c = Controller(database)

    start = time.time()

    with open("params.json", "r") as f:
        param_dict = json.load(f)

        # Create a runner for each set of parameters in the json file
        for run_id, params in param_dict.items():
            circuit_id = c.create_runner(params)

    c.run_all()

    print("Empty params:", database.get_conditional_data())
    print("Reg spsas: ", database.get_conditional_data(spsas={"spsa_alpha": 0.5, "spsa_gamma": 0.101, "spsa_c": 0.2, "spsa_A": 2, "spsa_a1": 0.2}))
    print("Bad spsas: ", database.get_conditional_data(spsas={"spsa_alpha": 0.9, "spsa_gamma": 0.7, "spsa_c": 0.3, "spsa_A": 2, "spsa_a1": 0.2}))
    print("Partial spsas: ", database.get_conditional_data(spsas={"spsa_alpha": None, "spsa_gamma": 0.101, "spsa_c": None, "spsa_A": 2, "spsa_a1": None}))
    print("Partial spsas and all else: ", database.get_conditional_data(spsas={"spsa_alpha": None, "spsa_gamma": 0.101, "spsa_c": None, "spsa_A": 2, "spsa_a1": None}, 
                                        feature_keys=['f_lept3_pt', 'f_lept4_pt', 'f_Z1mass'], test_accuracy_gt=0.5, test_accuracy_lt=0.95, 
                                        data_sizes=(80, 40, 80), misc_params={"is_local_simulator": True, "use_pca": False, "seed": 123, "batch_size": None, "n_epochs": None,  "num_layers": None, "obs": "XXI"}))
    # This is important, otherwise the database will not be saved
    database.close()

    # It is fine to do this because the params are saved in the output files
    os.remove("params.json")

    print(f"Total time: {time.time() - start}")


if __name__ == "__main__":
    main()