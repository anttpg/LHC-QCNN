import LHC_QML_module as lqm
from sklearn.model_selection import train_test_split 
from qiskit_machine_learning.algorithms.classifiers import VQC
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# signals_folder = "./data/signal/4e"
signals_folder = "./../LHC_data/actual_data/histos4mu/signal"
# backgrounds_folder = "./data/background/4e"
backgrounds_folder = "./../LHC_data/actual_data/histos4mu/background"

load_path = "./models/trained_vqc8"
#you can either give path to folder containing data files to be used as above or give paths to files individually in array
#i.e. signals_paths = ['./file1', './file2', './file3']

choice_feature_keys = [
 'f_lept1_pt', 'f_lept1_eta', 'f_lept1_phi', 'f_lept1_pfx', 'f_lept2_pt',
 'f_lept2_eta', 'f_lept2_phi', 'f_lept2_pfx', 'f_lept3_pt', 'f_lept3_eta',
 'f_lept3_phi', 'f_lept4_pt', 'f_lept4_eta', 'f_lept4_phi', 'f_Z1mass',
 'f_Z2mass', 'f_angle_costhetastar', 'f_angle_costheta1', 'f_angle_costheta2', 'f_angle_phi',
 'f_angle_phistar1', 'f_pt4l', 'f_eta4l', 'f_mass4l', 'f_deltajj',
 'f_massjj', 'f_jet1_pt', 'f_jet1_eta', 'f_jet1_phi', 'f_jet1_e',
 'f_jet2_pt', 'f_jet2_eta', 'f_jet2_phi', 'f_jet2_e']

use_pca = False
seed = 123

if use_pca:
    # Any three of these variables for training this week (02/21)
    training_feature_keys = ['f_lept3_pt', 'f_lept4_pt', 'f_Z1mass', 'f_angle_costheta2', 'f_pt4l', 'f_eta4l', 'f_jet1_pt', 'f_jet1_e']
    num_features = 4
else:
    training_feature_keys = ['f_lept1_pt', 'f_lept3_pt', 'f_lept4_pt', 'f_Z1mass']
    num_features = len(training_feature_keys)

#loads data from files
signal_dict, background_dict, files_used = lqm.load_data(signals_folder, backgrounds_folder,training_feature_keys)

#formats data for input into vqc
features, labels = lqm.format_data(signal_dict, background_dict)

#this is clunky, might want to make this its own function or something
#makes sure we use an equal amount of signal and background even if we have more signal than background
n_signal_events = (labels == 1).sum()
n_background_events = (labels ==0).sum()
if n_signal_events <= n_background_events:
    start = 0
    stop = 2*n_signal_events
else:
    start = -2*n_background_events
    stop = None

#splits data into testing and training sets
#data is first cut to inlcude equal number of signal and background events
#TODO: maybe split signal and backgrounds seperately to ensure equal number of signal/background in each test/training set and then combine and randomize order
train_features, test_features, train_labels, test_labels = train_test_split(
    features[start:stop,:], labels[start:stop], train_size=0.8, random_state=seed)

train_features, test_features = lqm.preprocess_data(train_features, test_features, use_pca, num_features, seed)


#lqm.plot_pairwise_dicts(signal, background)
#lqm.plot_pairwise(train_features, labels)

vqc = VQC.load(load_path)
lqm.score_model(vqc, train_features, test_features, train_labels, test_labels)

#prediction = vqc.predict(test_features)
#print('prediction finished')
prob = vqc._neural_network.forward(test_features, vqc._fit_result.x)

#lqm.plot_class_hist(prediction, test_labels)
#lqm.plot_roc(prediction, test_labels)

lqm.plot_class_hist(prob[:,1], test_labels)
lqm.plot_roc(prob[:,1], test_labels)
plt.show()