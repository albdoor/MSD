% Define the matrix A
A = [-2 -4  2;
     -2  1  2;
      4  2  5];

% Compute the condition number (2-norm)
kappa = cond(A);

% Display result
fprintf('Condition number of A (2-norm) = %.4f\n', kappa);
