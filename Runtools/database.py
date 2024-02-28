from sklearn.model_selection import train_test_split
import Modules.LHC_QML_module as lqm
import numpy as np



# Make system that aceepts list of args and plots whichever on a graph against each other


class Train_Test_Data:

        def __init__(self):
            self.n_signal_events = None
            self.n_background_events = None
            
            self.train_features = None
            self.train_labels = None

            self.valid_features = None
            self.valid_labels = None

            self.rest_features = None
            self.rest_labels = None

            self.test_features = None
            self.test_labels = None

    
        # Todo update to use signal / background processes
        def tts_preprocess(self, signal, background, params):
            # load data from files
            signal_dict, background_dict, files_used = lqm.load_data(
                params.signals_folder, params.backgrounds_folder, params.training_feature_keys
            )

            # formats data for input into vqc
            features, labels = lqm.format_data(signal_dict, background_dict)

            n_signal_events = (labels == 1).sum()
            n_background_events = (labels == 0).sum()

            features_signal = features[(labels==1)]
            features_background = features[(labels==0)]

            np.random.shuffle(features_signal)
            np.random.shuffle(features_background)

            features = np.concatenate((features_signal[:params.half_datasize], features_background[:params.half_datasize]))
            # labels = np.array([1]*half_datasize + [0]*half_datasize, requires_grad=False)
            labels = np.array([1]*params.half_datasize + [0]*params.half_datasize)

            train_features, rest_features, train_labels, rest_labels = train_test_split(
                features,
                labels,
                train_size=params.train_data_size,
                test_size=params.test_data_size + params.valid_data_size,
                random_state=params.seed,
                stratify=labels
            )

            # preprocess data (rescaling)
            train_features, rest_features = lqm.preprocess_data(
                train_features, rest_features, params.use_pca, params.num_features, params.seed
            )


            valid_features, test_features, valid_labels, test_labels = train_test_split(
                rest_features,
                rest_labels,
                train_size=params.valid_data_size,
                test_size = params.test_data_size,
                random_state=params.seed,
                stratify=rest_labels
            )


            self.n_signal_events = n_signal_events 
            self.n_background_events = n_background_events 
            
            self.train_features = train_features 
            self.train_labels = train_labels 

            self.valid_features = valid_features 
            self.valid_labels = valid_labels 

            self.rest_features = rest_features 
            self.rest_labels = rest_labels 

            self.test_features = test_features 
            self.test_labels = test_labels 
            
