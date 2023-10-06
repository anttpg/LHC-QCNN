import math

# Define the number of qubit weights per image in the dataset
# Must be a multiple of 2 

# Would be a 5x4 image
NUM_QUBITS = 16

# Would be a 16x8 image
#NUM_QUBITS = 128

#  Would be a 32x32 image
#NUM_QUBITS = 1024

#  Would be a 64x64 image
#NUM_QUBITS = 4096 

USE_INITIAL = True

MAX_PIXEL = 0xFFFFFF
SIZE = (NUM_QUBITS, 1)

DATA_PATH = "LHC_data/test_data"
LABEL = "ant"

# DATA_PATH = "cats_dogs_light/test"
# LABEL = "dog"

#Used to define the shape of the image we will use
def find_factors_close_to_sqrt(n):
    for i in range(int(math.sqrt(n)), 0, -1):
        if n % i == 0:
            return i, n // i


SHAPE_X, SHAPE_Y = find_factors_close_to_sqrt(NUM_QUBITS)

# Ensure SHAPE_X is always >= SHAPE_Y
if SHAPE_Y > SHAPE_X:
    SHAPE_X, SHAPE_Y = SHAPE_Y, SHAPE_X
    
print("Shape X Pixels:", SHAPE_X)
print("Shape Y Pixels:", SHAPE_Y)
