## Author: Yash Lokare

## Importing relevant libraries

import numpy as np
import scipy.linalg as la
import scipy.spatial as spat
from scipy.stats import unitary_group
from scipy.stats import moment
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
from scipy.linalg import norm
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass

# Libraries for implementing the VQD algorithm
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import Sampler, Estimator
from qiskit_aer import AerSimulator
# from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Sampler, Session, Options
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes, TwoLocal, EfficientSU2
from qiskit_algorithms.optimizers import *
from qiskit_algorithms.state_fidelities import ComputeUncompute

from qiskit_algorithms.eigensolvers import EigensolverResult, VQD
from qiskit_algorithms import NumPyMinimumEigensolver, VQE

# Import classical optimizers
from qiskit_algorithms.optimizers import SPSA, P_BFGS, COBYLA, IMFIL, SNOBFIT, NELDER_MEAD, SLSQP, NFT, ADAM, POWELL, GradientDescent, BOBYQA

# Import Statevector and SparsePauliOp
from qiskit.quantum_info import SparsePauliOp, Statevector

# Import noise models
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

# Import a fake backend and Qiskit simulators and/or noise libraries
from qiskit_aer import AerSimulator
# from qiskit_aer.primitives import Estimator as AerEstimator
# from qiskit_aer.noise import NoiseModel

## Helper functions (remain the same for zeromode and eigenvalue computations; no changes)

def get_unitary(matrix, add_half = False):
    """
    Purpose: given a matrix, returns the unitary, hermitian matrix to be diagonalized
    Input: matrix -> the matrix to be diagonalized
    Output: U -> the unitary matrix
            nqubits -> the number of qubis needed to represent the basis of U
            dimension -> the dimension of the original matrix
    """
    assert matrix.ndim == 2, "Error: Only a matrix maybe processed"
    assert matrix.shape[0] == matrix.shape[1], "Error: Only a square matrix maybe processed"

    if np.any(np.transpose(matrix) != matrix):
        matrix_T = np.transpose(matrix)
        matrix = np.dot(matrix_T, matrix)

    ## Finding the dimension of the matrix
    dimension_hermitian = matrix.shape[0]

    ## Finding the number of qubits required to represent the matrix
    nqubits = int(np.ceil(np.log2(dimension_hermitian)))

    ## Construct the relevant matrix
    op_dim = 2 ** nqubits
    op = np.eye(op_dim)
    op[0:dimension_hermitian, 0:dimension_hermitian] = np.copy(matrix)

    if add_half:
        op = op + np.pi * np.eye(op.shape[0])

    U = la.expm(1j*op)

    # Get the dimensions of the unitary matrix
    dimension = U.shape[0]

    return U, nqubits, dimension

def find_probability(eigenvector_raw):
    """
    Purpose: Find the probability associated with each basis of an eigenvector
    Input: eigenvector_raw -> Numpy array documenting the number of times each basis is detected within the eigenvector
    Output: eigenvector_prob -> Numpy array documenting the probability of detecting each basis
    """
    count_total = np.sum(eigenvector_raw)
    eigenvector_prob = eigenvector_raw / count_total

    return eigenvector_prob

def find_amplitude(eigenvector_prob):
    """
    Purpose: Finding the probability amplitude of each basis using quantum mechanics
    Input: eigenvector_prob -> Numpy array documenting the probability that each basis is measured
    Output: eigenvector -> Numpy array representing the eigenvector
    """
    eigenvector = np.sqrt(eigenvector_prob)
    return eigenvector

def normalize_eigenvector(vector):
    """
    Purpose: Normalizes a vector such that its norm is 1
    Input: vector -> The vector to be normalized
    Output: vector -> The normalized vector
    """
    L2 = np.sum(np.square(vector))
    vector = vector / np.sqrt(L2)

    return vector

def get_similarity(a, b):
    # Function to compute the similarity between 2 zeromodes

    # Get absolute values
    b = np.abs(b)

    numerator = np.abs(np.dot(a.conj().T, b))**2
    denominator = np.linalg.norm(a)**2 * np.linalg.norm(b)**2

    return numerator / denominator

def get_expectation(matrix, zeromode):
    # Compute the expectation value of the Schlogl operator in the steady-state

    # Convert the zeromode into a matrix (column vector)
    zeromode = np.array(zeromode).reshape(len(zeromode), 1)
    zeromode = np.abs(zeromode) # get rid of all (extraneous) negative values (since this is a PDF)

    # Compute the steady-state expectation value
    value = np.dot(matrix, zeromode)
    expectation_value = np.dot(zeromode.T, value)

    return expectation_value

def compute_rmsd(list1, list2):
    # Ensure the lists have the same length

    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length.")

    # Compute the RMSD
    rmsd = np.sqrt(np.mean((np.array(list1) - np.array(list2)) ** 2))
    return rmsd

def get_expectation(matrix, zeromode):
    # Compute the expectation value of the Schlogl operator in the steady-state

    # Convert the zeromode into a matrix (column vector)
    zeromode = np.array(zeromode).reshape(len(zeromode), 1)
    zeromode = np.abs(zeromode) # get rid of all (extraneous) negative values (since this is a PDF)

    # Compute the steady-state expectation value
    value = np.dot(matrix, zeromode)
    expectation_value = np.dot(zeromode.T, value)

    return expectation_value

def compute_eigenvalues(matrix):
    """
    Computes the two eigenvalues with the highest real parts of a general matrix.

    Parameters:
        matrix (np.ndarray): The matrix for which to compute the eigenvalues.

    Returns:
        list: A list containing the two eigenvalues with the highest real parts [eigenvalue1, eigenvalue2].
    """
    # Compute all eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Sort by real part in descending order and select the two with the lowest real parts
    lowest_two_eigenvalues = sorted(eigenvalues, key=lambda x: x.real, reverse=False)[:2]

    # Print
    print('The lowest two eigenvalues are:')
    print(lowest_two_eigenvalues)
    print()

    return lowest_two_eigenvalues

def compute_relative_errors(exact_eigenvalues, vqd_eigenvalues):
    """
    Computes the relative errors for the first and second eigenvalues for each optimizer.

    Parameters:
        exact_eigenvalues (list): List of the two lowest exact eigenvalues [lambda_1, lambda_2].
        vqd_eigenvalues (dict): Dictionary where keys are optimizer names and values are lists of the two
                                lowest VQD eigenvalues computed by each optimizer.

    Returns:
        dict: Dictionary where each key is an optimizer name and the value is a tuple of the relative errors
              (error_first, error_second).
    """
    # Extract the exact eigenvalues for easy reference
    exact_first, exact_second = exact_eigenvalues

    # Initialize a dictionary to store relative errors for each optimizer
    relative_errors = {}

    for optimizer, eigenvalues in vqd_eigenvalues.items():
        vqd_first, vqd_second = eigenvalues  # VQD-computed first and second eigenvalues for this optimizer

        # Calculate relative errors for the first and second eigenvalues
        error_first = abs((vqd_first - exact_first) / exact_first) if exact_first != 0 else np.nan
        error_second = abs((vqd_second - exact_second) / exact_second) if exact_second != 0 else np.nan

        # Store the errors in the dictionary
        relative_errors[optimizer] = (error_first, error_second)

    return relative_errors

## VQD implementation (remains the same for zeromode and eigenvalue computations)

def run_vqd(matrix, ansatz, optimizer, seed, zeromode_exact):
    # Function to compute the execution time for different optimizers and ansatz depths (averaged over 10 independent VQD runs in each case)

    dimension = matrix.shape[0]
    num_qubits = int(np.log2(dimension))

    # Define the qubit Hamiltonian
    qub_hamiltonian = SparsePauliOp.from_operator(matrix)

    # Compute using NumPyMinimumEigensolver for reference
    sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qub_hamiltonian)

    # Initial point for the classical optimizer
    seed_value = seed
    np.random.seed(seed_value)
    initial_point = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    # Initializing the Estimator, Sampler, and fidelity
    estimator = Estimator()
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler)

    # Run the VQE algorithm
    @dataclass
    class VQDLog:
        values: list = None
        parameters: list = None

        def __post_init__(self):
            self.values = []
            self.parameters = []

        # Update function to match the expected arguments
        def update(self, count, parameters, mean, _metadata, _extra):
            self.values.append(mean)
            self.parameters.append(parameters)

    log = VQDLog()

    vqd = VQD(estimator,
              fidelity, ansatz, optimizer, k=2, initial_point=initial_point, callback=log.update)
    result = vqd.compute_eigenvalues(qub_hamiltonian)

    # Estimate the zeromode
    optimal_params = result.optimal_points
    zeromode_points = optimal_params[0]
    final_circuit = ansatz.assign_parameters(zeromode_points)
    zeromode_vqd = Statevector.from_instruction(final_circuit)
    zeromode = zeromode_vqd.data.tolist()

    # Get the eigenvalues
    eigenvalues_vqd = result.eigenvalues

    # Compute the fidelity between the classical and quantum zeromodes
    zeromode_exact = np.array(zeromode_exact).reshape(len(zeromode_exact), 1)
    zeromode = np.array(zeromode).reshape(len(zeromode_exact), 1)
    fidelity = get_similarity(zeromode_exact, zeromode)

    # Get the number of function calls
    num_func_calls = result.cost_function_evals

    return zeromode, fidelity, eigenvalues_vqd, num_func_calls

## Evaluating performance metrics at multiple ansatz depths

def analyze_ansatz_performance_mutiple_depths_schlogl_model(matrix, zeromode_classic, optimizer, num_qubits, max_depth):

    # Initialize dictionaries to store results
    zeromodes = {}
    metrics = {}  # To store fidelity and function call counts
    eigenvalues = {} # to store the eigenvalues

    optimizer_name = optimizer.__class__.__name__

    # Loop through each ansatz type
    for AnsatzClass in [RealAmplitudes, TwoLocal, EfficientSU2]:
        # Initialize ansatz with appropriate configurations
        if AnsatzClass == RealAmplitudes:
            ansatz = AnsatzClass(num_qubits=num_qubits, entanglement='full', reps=1)
        elif AnsatzClass == TwoLocal:
            ansatz = AnsatzClass(num_qubits=num_qubits, rotation_blocks=['ry'], entanglement_blocks='cx', reps=1)
        elif AnsatzClass == EfficientSU2:
            ansatz = AnsatzClass(num_qubits=num_qubits, su2_gates=['ry'], entanglement='sca', reps=1)

        ansatz_name = AnsatzClass.__name__
        pair_name = f"{optimizer_name}-{ansatz_name}"
        print(f"\nRunning VQD for optimizer-ansatz pair: {pair_name}")

        # Initialize dictionaries to store depth-specific results
        zeromodes[pair_name] = {}
        metrics[pair_name] = {}
        eigenvalues[pair_name] = {}

        for current_depth in range(1, max_depth + 1):
            # Set `reps` (depth) for the current ansatz
            ansatz.reps = current_depth

            # Temporary storage for the fidelity results
            fidelities = []
            function_calls = []  # To track the number of function calls for each run
            all_zeromodes = []  # To store all zeromodes for the depth
            all_eigenvalues = [] # to store all the eigenvalues

            # Perform multiple independent VQD runs to calculate fidelity and resource usage
            for run in range(10):  # Number of independent runs
                seed = run + 1  # Set a unique seed for each run

                # Run VQD
                zeromode, fidelity_value, eigenvalues_vqd, function_call_count = run_vqd(
                    matrix=matrix, ansatz=ansatz, optimizer=optimizer, seed=seed,
                    zeromode_exact=zeromode_classic)

                # Append results
                fidelities.append(fidelity_value)
                function_calls.append(function_call_count)
                all_zeromodes.append(zeromode)
                all_eigenvalues.append(eigenvalues_vqd)

            # Identify the run with the highest fidelity at this depth
            best_run_index = np.argmax(fidelities)
            best_fidelity = fidelities[best_run_index]
            best_zeromode = all_zeromodes[best_run_index]
            best_function_calls = function_calls[best_run_index]
            best_eigenvalues = all_eigenvalues[best_run_index]

            # Store results for the current depth
            zeromodes[pair_name][current_depth] = best_zeromode
            metrics[pair_name][current_depth] = {
                'fidelity': best_fidelity,
                'function_calls': best_function_calls
            }
            eigenvalues[pair_name][current_depth] = {
                'lowest eigenvalue': best_eigenvalues[0],
                'first excited state eigenvalue': best_eigenvalues[1]
            }

            print(f"{pair_name} - Depth {current_depth}: Best fidelity = {best_fidelity}, Function calls = {best_function_calls}, \
            Best zeromode: {best_zeromode}, Best lowest eigenvalue: {best_eigenvalues[0]}, First excited state eigenvalue: {best_eigenvalues[1]}")

    return zeromodes, metrics, eigenvalues
