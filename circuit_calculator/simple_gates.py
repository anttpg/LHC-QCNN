import numpy as np

def Rz(b):
    # Rotation about the Z-axis
    return np.array([
        [np.exp(-1j * (b / 2)), 0],
        [0, np.exp(1j * (b / 2))]
    ])

def Ry(b):
    # Rotation about the Y-axis
    return np.array([
        [np.cos(b / 2), -np.sin(b / 2)],
        [np.sin(b / 2), np.cos(b / 2)]
    ])

def Rx(b):
    # Rotation about the X-axis
    return np.array([
        [np.cos(b / 2), -1j * np.sin(b / 2)],
        [-1j * np.sin(b / 2), np.cos(b / 2)]
    ])

def Hadamard():
    # Hadamard gate
    return np.array([
        [1/np.sqrt(2), 1/np.sqrt(2)],
        [1/np.sqrt(2), -1/np.sqrt(2)]
    ])

def PauliX():
    # Pauli X gate (NOT gate)
    return np.array([
        [0, 1],
        [1, 0]
    ])

def PauliY():
    # Pauli Y gate
    return np.array([
        [0, -1j],
        [1j, 0]
    ])

def PauliZ():
    # Pauli Z gate
    return np.array([
        [1, 0],
        [0, -1]
    ])

def Identity():
    # Identity gate
    return np.array([
        [1, 0],
        [0, 1]
    ])

def CnotAB():
    # control on A target on B
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

def CnotBA():
    # control on B target on A
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ])
