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

## Helper functions

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

## VQD implementation for eigenvalue calculations (state vector simulations)

def run_vqd_for_eigenvalues(matrix, ansatz, optimizer, seed):
    """
    Function to compute the zeromode and eigenvalues for different optimizers and ansatz depths,
    averaged over multiple independent VQD runs.

    Args:
        matrix (np.ndarray): The Hamiltonian matrix.
        ansatz (QuantumCircuit): The ansatz circuit.
        optimizer (Optimizer): Classical optimizer for VQD.
        seed (int): Seed value for reproducibility.
        eigenvalues_exact (list): List of exact eigenvalues for reference (classical values).

    Returns:
        zeromode (list): The zeromode statevector obtained from VQD.
        eigenvalues_vqd (list): List of eigenvalues obtained from VQD.
        num_func_calls (int): Number of function calls made by the optimizer.
    """

    dimension = matrix.shape[0]
    num_qubits = int(np.log2(dimension))

    # Define the qubit Hamiltonian
    qub_hamiltonian = SparsePauliOp.from_operator(matrix)

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

    # Get the optimizer runtime (not used in the current context, but can be logged)
    time = result.optimizer_times

    # Extract the optimal parameters
    optimal_params = result.optimal_points
    zeromode_points = optimal_params[0]
    final_circuit = ansatz.assign_parameters(zeromode_points)
    zeromode_vqd = Statevector.from_instruction(final_circuit)
    zeromode = zeromode_vqd.data.tolist()

    # Extract the first and second eigenvalues from VQD results
    eigenvalues_vqd = result.eigenvalues

    # Get the number of function calls
    num_func_calls = result.cost_function_evals

    # Get the gate count and circuit depth of the variational ansatz
    decomposed_ansatz = ansatz.decompose()
    gate_count_dict = decomposed_ansatz.count_ops()
    gate_count = sum(gate_count_dict.values())
    circuit_depth = decomposed_ansatz.depth()  # Depth of the ansatz circuit

    return zeromode, eigenvalues_vqd, gate_count, circuit_depth, num_func_calls

## Estimate quantum resources (accuracy threshold pertains to \lambda_1) --- discussed in the main text

def estimate_resources_first_eigenvalue(matrix, optimizers, num_qubits, target_relative_error_threshold, exact_second_eigenvalue):
    """
    Computes the minimum ansatz depth required to achieve a target relative error threshold for the second eigenvalue
    for a set of optimizers and ansatz types. For each optimizer-ansatz pair, at the minimum depth that achieves
    the relative error threshold, the quantum-computed lowest two eigenvalues of the original matrix H are returned.

    Parameters:
        matrix (np.ndarray): The original matrix H.
        optimizers (list): A list of optimizers to use.
        num_qubits (int): The number of qubits.
        target_relative_error_threshold (float): The relative error threshold for the second eigenvalue.
        exact_second_eigenvalue (float): The exact second eigenvalue of the original matrix H.

    Returns:
        min_depths (dict): Minimum depths for each optimizer-ansatz pair to achieve the relative error threshold.
        eigenvalues (dict): Quantum-computed first and second eigenvalues for each optimizer-ansatz pair at the minimum depth.
        metrics (dict): Number of function calls, gate count, circuit depth, and relative error at the threshold meeting depth.
    """

    # Initialize dictionaries to store results
    min_depths = {}
    eigenvalues = {}
    metrics = {}

    # Define ansatz types
    ansatz_classes = [RealAmplitudes, TwoLocal, EfficientSU2]

    # Loop through each optimizer
    for optimizer in optimizers:
        optimizer_name = optimizer.__class__.__name__

        # Loop through each ansatz type
        for AnsatzClass in ansatz_classes:
            ansatz_name = AnsatzClass.__name__
            pair_name = f"{optimizer_name}-{ansatz_name}"
            print(f"\nRunning VQD for optimizer-ansatz pair: {pair_name}")

            current_depth = 1
            max_depth = 10
            converged = False  # Flag to check if convergence occurs

            while current_depth <= max_depth:  # Loop up to max_depth
                best_relative_error = np.inf  # Initialize to a high value
                best_run_result = None  # Track best eigenvalues for the run with the best relative error
                best_run_metrics = {}  # Track metrics for the run that meets the threshold
                best_relative_error_value = None  # Store the relative error at the best run

                # Initialize ansatz with appropriate configurations
                if AnsatzClass == RealAmplitudes:
                    ansatz = AnsatzClass(num_qubits=num_qubits, entanglement='full', reps=current_depth)
                elif AnsatzClass == TwoLocal:
                    ansatz = AnsatzClass(num_qubits=num_qubits, rotation_blocks=['ry'], entanglement_blocks='cx', reps=current_depth)
                elif AnsatzClass == EfficientSU2:
                    ansatz = AnsatzClass(num_qubits=num_qubits, su2_gates=['ry'], entanglement='sca', reps=current_depth)

                # Perform multiple independent VQD runs
                for run in range(10):  # 10 independent VQD runs
                    seed = run + 1

                    # Run VQD and capture additional metrics
                    zeromode, eigenvalues_quantum, gate_count, circuit_depth, func_calls = run_vqd_for_eigenvalues(
                        matrix=matrix,
                        ansatz=ansatz,
                        optimizer=optimizer,
                        seed=seed,
                    )

                    # Calculate the relative error in the second eigenvalue
                    second_eigenvalue = eigenvalues_quantum[1]
                    relative_error_in_second_eigenvalue = abs(second_eigenvalue - exact_second_eigenvalue) / abs(exact_second_eigenvalue)

                    # Update if this run provides a lower relative error
                    if relative_error_in_second_eigenvalue < best_relative_error:
                        best_relative_error = relative_error_in_second_eigenvalue
                        best_relative_error_value = relative_error_in_second_eigenvalue
                        best_run_result = eigenvalues_quantum  # Store the two eigenvalues
                        best_run_metrics = {
                            "run_number": run + 1,
                            "function_calls": func_calls,
                            "gate_count": gate_count,
                            "circuit_depth": circuit_depth,
                            "relative_error": best_relative_error_value
                        }

                # Check if the best relative error in the second eigenvalue meets the threshold
                if best_relative_error <= target_relative_error_threshold:
                    min_depths[pair_name] = current_depth
                    eigenvalues[pair_name] = best_run_result
                    metrics[pair_name] = best_run_metrics
                    print(f"Minimum ansatz depth for {pair_name} to achieve target relative error threshold: {current_depth}")
                    print(f"Best estimate for lowest two eigenvalues at this depth: {best_run_result}")
                    print(f"Metrics at threshold: Function calls = {best_run_metrics['function_calls']}, "
                          f"Gate count = {best_run_metrics['gate_count']}, Circuit depth = {best_run_metrics['circuit_depth']}, "
                          f"Relative Error = {best_run_metrics['relative_error']}")
                    print(f"Relative error threshold met at run number: {best_run_metrics['run_number']}")

                    converged = True
                    break  # Exit if threshold is met

                current_depth += 1  # Increase depth if threshold not met

            # If no convergence occurs within max_depth, mark as "did not converge"
            if not converged:
                min_depths[pair_name] = "Did not converge"
                eigenvalues[pair_name] = "Did not converge"
                metrics[pair_name] = "Did not converge"
                print(f"{pair_name} did not converge within {max_depth} depths.")

    return min_depths, eigenvalues, metrics
