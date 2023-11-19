import time
import torch

N_QUBITS = 4                # Number of qubits
ALPHA = 0.0002               # Learning rate
BATCH_SIZE = 4              # Number of samples for each training step
NUM_EPOCHS = 200              # Number of training epochs
Q_DEPTH = 6                 # Depth of the quantum circuit (number of variational layers)
GAMMA_LR_SCHEDULER = 0.1    # Learning rate reduction applied every 10 epochs.
Q_DELTA = 0.01              # Initial spread of random quantum weights
START_TIME = time.time()    # Start of the computation timer
DATA_DIR = '../_data/Dark-QCD-Data/'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")