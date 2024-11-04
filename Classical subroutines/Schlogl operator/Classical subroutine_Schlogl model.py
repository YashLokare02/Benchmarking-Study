## Authors: Tilas Kabengele

## Importing relevant libraries
import numpy as np

## Function to generate the Schlogl operator matrix
def TridiagA(lambda_func, mu_func, N):
    """
    Creates tridiagonal stochastic matrix for the Schlogl model
    Inputs lambda and mu model functions, and N desired size of matrix
    Returns stochastic matrix (non-hermitian)
    """
    # initialize diagonals
    d1 = np.zeros(N - 1)
    d2 = np.zeros(N)
    d3 = np.zeros(N - 1)

    # element 1,1
    d2[0] = -lambda_func(0)

    # element N,N
    d2[N - 1] = -mu_func(N - 1)

    # bottom diagonal elements
    for i in range(N - 1):
        d1[i] = lambda_func(i)

    # top diagonal elements
    for i in range(1, N):
        d3[i - 1] = mu_func(i)

    # main diagonal elements
    for i in range(1, N - 1):
        d2[i] = -lambda_func(i) - mu_func(i)

    # putting the diagonals together
    Q = np.diag(d1, k = -1) + np.diag(d2, k = 0) + np.diag(d3, k = 1)

    return Q

def get_volume_array(start_V, stop_V, n_operator_qubits):
    # Function to generate the initial volume array (to carry out computations)

    num_elements = 2 ** n_operator_qubits
    step_size = (stop_V - start_V)/(num_elements - 1)

    # Generate the volume array, given that the step size has been determined
    volume_array = np.arange(start_V, stop_V, step_size)

    return volume_array

# Computing the block diagonal representation of the Schlogl operator matrix
Q0 = np.zeros((len(Q), len(Q)))
hermitian_matrix = np.block([[Q0, Q], [Q.T, Q0]])
