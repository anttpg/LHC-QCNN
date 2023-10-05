import math

#Define the number of qubit weights per image in the dataset
NUM_QUBITS = 8 # Must be a multiple of 2

MAX_PIXEL = 0xFFFFFF
SIZE = (NUM_QUBITS, 1)

# Original was 2, 4 for vertical line test
SHAPE_X = 4
SHAPE_Y = 2
# SHAPE_X = int(math.sqrt(NUM_QUBITS))
# SHAPE_Y = int(math.sqrt(NUM_QUBITS))
