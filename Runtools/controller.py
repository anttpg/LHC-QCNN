import numpy as np
import os

import LHC_QML_module as lqm

class Controller:
    class Parameters:
        seed = 123
        # Features to train on
        training_feature_keys = [
            "f_mass4l",
            # "f_eta4l",
            "f_Z2mass",
            "f_Z1mass",
        ]

        num_features = len(training_feature_keys)

        save_folder = os.path.join("saved", "model1g3-qiskit-estimator-corrected")

        batch_size = 2
        n_epochs = 1

        use_pca = False

        train_data_size = 80
        test_data_size = 80
        valid_data_size = 40
        total_datasize = train_data_size + test_data_size + valid_data_size
        half_datasize = total_datasize // 2 # 80 signal and 80 backgrounds

        is_local_simulator = True

        n_qubits = 3
        num_layers = 5

        spsa_alpha = 0.5
        spsa_gamma = 0.101
        spsa_c     = 0.2
        spsa_A     = 2.
        spsa_a1    = 0.2
        spsa_a     = spsa_a1 * (spsa_A + 1) ** spsa_alpha

        signals_folder = "LHC_data\\actual_data\\histos4mu\\histos4mu\\signal"
        backgrounds_folder = "LHC_data\\actual_data\\histos4mu\\histos4mu\\background"



    def __init__(self, ):
        pass

    def tts_preprocess(self, signal, background):
        # load data from files
        signal_dict, background_dict, files_used = lqm.load_data(
            signals_folder, backgrounds_folder, training_feature_keys
        )

        # formats data for input into vqc
        features, labels = lqm.format_data(signal_dict, background_dict)

        n_signal_events = (labels == 1).sum()
        n_background_events = (labels == 0).sum()

        features_signal = features[(labels==1)]
        features_background = features[(labels==0)]

        np.random.shuffle(features_signal)
        np.random.shuffle(features_background)

        features = np.concatenate((features_signal[:half_datasize], features_background[:half_datasize]))
        # labels = np.array([1]*half_datasize + [0]*half_datasize, requires_grad=False)
        labels = np.array([1]*half_datasize + [0]*half_datasize)

        train_features, rest_features, train_labels, rest_labels = train_test_split(
            features,
            labels,
            train_size=train_data_size,
            test_size=test_data_size + valid_data_size,
            random_state=seed,
            stratify=labels
        )

        # preprocess data (rescaling)
        train_features, rest_features = lqm.preprocess_data(
            train_features, rest_features, use_pca, num_features, seed
        )


        valid_features, test_features, valid_labels, test_labels = train_test_split(
            rest_features,
            rest_labels,
            train_size=valid_data_size,
            test_size = test_data_size,
            random_state=seed,
            stratify=rest_labels
        )