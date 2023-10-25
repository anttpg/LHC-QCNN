

% Input will a 4d vector
function U2 = U2_Circuit_Calculator()
    format rat


    CnotAB = [
    1 0 0 0 ; 
    0 1 0 0 ; 
    0 0 0 1 ; 
    0 0 1 0
    ];

    CnotBA = [
    1 0 0 0 ; 
    0 0 0 1 ; 
    0 0 1 0 ; 
    0 1 0 0
    ];

    theta = pi
    phi = pi
    lambda = 0
    input = [0 ; 0 ; 1/sqrt(2) ; 1/sqrt(2)]

    Q2_Rz = kron(eye(2), Rz(-pi/2))
    Rz_Ry = kron(Rz(theta), Ry(phi))

    Q1_Rz = kron(Rz(pi/2), eye(2))
    Q2_Ry = kron(eye(2), Ry(lambda))


    currentVals = Q2_Rz * input
    currentVals = CnotBA * currentVals
    currentVals = Rz_Ry * currentVals
    currentVals = CnotAB * currentVals
    currentVals = Q2_Ry * currentVals
    currentVals = CnotBA * currentVals
    currentVals = Q1_Rz * currentVals

    return 
end







function Rz = Rz(b)
    % Calculate the elements of the matrix based on the input 'b'
    element1 = exp(-1i * b / 2);
    element2 = 0;
    element3 = 0;
    element4 = exp(1i * b / 2);

    % Create the 2x2 matrix
    Rz = [element1, element2; element3, element4];
end

function Ry = Ry(b)
    % Calculate the elements of the matrix based on the input 'b'
    element1 = cos(b/2);
    element2 = -sin(b/2);
    element3 = sin(b/2);
    element4 = cos(b/2);

    % Create the 2x2 matrix
    Ry = [element1, element2; element3, element4];
end

