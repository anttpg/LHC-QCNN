import math
from argparse import ArgumentParser

# Define the number of qubit weights per image in the dataset
# Must be a multiple of 2 

#For cmd arguments
parser = ArgumentParser()
parser.add_argument("--trained", help="with pre-trained weights", action="store_true", default= False)
parser.add_argument("--iterations",  help="set the num iterations to run", type=int, default=30)
parser.add_argument("--qubits",  help="override the default number of qubits to run (16)", type=int, default=4)
ARGS = parser.parse_args()


MAX_PIXEL = 0xFFFFFF
DATA_PATH = "QCNN"
LABEL = "ant"

NUM_QUBITS = ARGS.qubits
MAX_ITER = ARGS.iterations
USE_TRAINED = ARGS.trained

#Used to define the shape of the image we will use
def find_factors_close_to_sqrt(n):
    for i in range(int(math.sqrt(n)), 0, -1):
        if n % i == 0:
            return i, n // i

SIZE = (NUM_QUBITS, 1)
SHAPE_X, SHAPE_Y = find_factors_close_to_sqrt(NUM_QUBITS)

# Ensure SHAPE_X is always >= SHAPE_Y
if SHAPE_Y > SHAPE_X:
    SHAPE_X, SHAPE_Y = SHAPE_Y, SHAPE_X

print("Shape X Pixels:", SHAPE_X)
print("Shape Y Pixels:", SHAPE_Y)


