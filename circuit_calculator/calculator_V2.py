import numpy as np

def U2_Circuit_Calculator():
    CnotAB = np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 0, 1], 
        [0, 0, 1, 0]
    ])

    CnotBA = np.array([
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [0, 0, 1, 0], 
        [0, 1, 0, 0]
    ])

    theta = np.pi
    phi = np.pi
    lambda_ = 0
    input_ = np.array([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]).reshape(-1, 1)

    Q2_Rz = np.kron(np.eye(2), Rz(-np.pi/2))
    Rz_Ry = np.kron(Rz(theta), Ry(phi))

    Q1_Rz = np.kron(Rz(np.pi/2), np.eye(2))
    Q2_Ry = np.kron(np.eye(2), Ry(lambda_))

    currentVals = Q2_Rz @ input_
    currentVals = CnotBA @ currentVals
    currentVals = Rz_Ry @ currentVals
    currentVals = CnotAB @ currentVals
    currentVals = Q2_Ry @ currentVals
    currentVals = CnotBA @ currentVals
    currentVals = Q1_Rz @ currentVals

    return currentVals


def Rz(b):
    element1 = np.exp(-1j * (b / 2))
    element2 = 0
    element3 = 0
    element4 = np.exp(1j * (b / 2))
    Rz = np.array([[element1, element2], [element3, element4]])
    return Rz

def Ry(b):
    element1 = np.cos(b / 2)
    element2 = -np.sin(b / 2)
    element3 = np.sin(b / 2)
    element4 = np.cos(b / 2)
    Ry = np.array([[element1, element2], [element3, element4]])
    return Ry



def approximate_part(value):
    # Near zero check
    if abs(value) < np.exp(-5):
        return "NEAR ZERO", value
    
    # Adjusted for small magnitudes
    sign = -1 if value < 0 else 1  
    magnitude = abs(value)
    reciprocal_sqrt_magnitude = 1 / np.sqrt(magnitude)
    N = np.round(1 / (reciprocal_sqrt_magnitude ** 2)).astype(int)


    if N == 0:
        return "1/sqrt(>0)", value 
    
    approx = f"{sign}/sqrt({N})"
    return approx, value

def approximate_as_reciprocal_sqrt(x):
    real_part = np.real(x)
    imag_part = np.imag(x)

    approx_real, true_real = approximate_part(real_part)
    approx_imag, true_imag = approximate_part(imag_part)

    return (approx_real, approx_imag), (true_real, true_imag)

if __name__ == "__main__":
    output = U2_Circuit_Calculator()
    print("Output:")
    for elem in output:
        (approx_real, approx_imag), (true_real, true_imag) = approximate_as_reciprocal_sqrt(elem)
        print(f"Real: {approx_real} (True: {true_real}), Imaginary: {approx_imag} (True: {true_imag})")
