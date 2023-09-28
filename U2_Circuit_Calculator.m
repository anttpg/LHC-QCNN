
function U2 = U2_Circuit_Calculator()
    U2 = [];

    CnotAB = [
    1 0 0 0 ; 
    0 1 0 0 ; 
    0 0 0 1 ; 
    0 0 1 0
    ];

    CnotBA = [
    0 1 0 0 ; 
    1 0 0 0 ; 
    0 0 1 0 ; 
    0 0 0 1
    ];
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

