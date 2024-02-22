import LHC_QML_module as lqm
from sklearn.model_selection import train_test_split 
import time
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA
from qiskit.utils.algorithm_globals import algorithm_globals
from matplotlib import pyplot as plt
from qiskit.primitives import BackendSampler
from qiskit_ionq import IonQProvider

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

signals_folder = "./data/signal/4e"
backgrounds_folder = "./data/background/4e"
save_folder = "./models"
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
seed = 42
algorithm_globals._random_seed = seed
n_training_points = 400

if use_pca:
    # training_feature_keys = choice_feature_keys
    training_feature_keys = ['f_lept3_pt', 'f_lept4_pt', 'f_pt4l', 'f_angle_costheta2', 'f_jet1_pt', 'f_jet2_pt', 'f_Z1mass', 'f_eta4l']
    # training_feature_keys = ['f_lept3_pt', 'f_lept4_pt', 'f_pt4l', 'f_angle_costheta2', 'f_jet1_pt', 'f_jet2_pt', 'f_Z1mass', 'f_eta4l']
    num_features = 4 #number of pca components to use
else:
    training_feature_keys = ['f_lept1_pt', 'f_lept3_pt', 'f_Z1mass', 'f_lept4_pt']
    # training_feature_keys = ['f_jet1_pt', 'f_pt4l', 'f_lept3_pt', 'f_massjj']
    #training_feature_keys = ['f_Z2mass','f_pt4l', 'f_eta4l', 'f_mass4l']
    num_features = len(training_feature_keys)

#flattening these circuits makes the training faster, but to do so you need to modify their
#implementation (in the qiskit code) to inherit the flatten input parameter from its base class
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)#, flatten=True)
ansatz = EfficientSU2(num_qubits=num_features, reps=3)#, flatten=True)
optimizer = SLSQP(maxiter=35)

#NOTE: do not make code publicly available with this key in it
provider = IonQProvider("placeholder")
backend_ionq_sim = provider.get_backend("ionq_simulator")
backend_ionq_qpu = provider.get_backend("ionq_qpu")

backend = backend_ionq_sim

sampler = Sampler()
# sampler = BackendSampler(backend, options={"shots":1024})


#loads data from files
signal_dict, background_dict, files_used = lqm.load_data(signals_folder, backgrounds_folder, training_feature_keys)

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

losses = []
times = []

def callback(weights, step_loss):
#function called at each step of vqc training
#expected to only have these two inputs and return None
    losses.append(step_loss)
    times.append(time.time())
    print(f"Iteration {len(losses)} Training time: {round(times[-1]-times[-2],2)} seconds")
    print(f"Loss: {step_loss:.4f}")

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback
)
# vqc = VQC.load('./models/trained_vqc')
# vqc.warm_start = True
# vqc.optimizer = SLSQP(maxiter=15)

start = time.time()
times.append(start)
vqc.fit(train_features[:n_training_points,:], train_labels[:n_training_points]) #training the vqc
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds\n")

scores = lqm.score_model(vqc, train_features, test_features, train_labels, test_labels)
fit_filepath = lqm.save_model(vqc, save_folder, seed=seed, n_training_points=n_training_points, 
               training_feature_keys=training_feature_keys, files_used=files_used, scores=scores, use_pca=use_pca)

lqm.plot_loss(losses)
plt.savefig(fit_filepath + '_loss.png')

prob = vqc._neural_network.forward(test_features, vqc._fit_result.x)
lqm.plot_class_hist(prob[:,1], test_labels)
plt.savefig(fit_filepath + '_dis.png')