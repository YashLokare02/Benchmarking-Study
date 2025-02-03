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

## VQD implementation for zeromode calculations (state vector simulations)

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
              fidelity, ansatz, optimizer, k=1, initial_point=initial_point, callback=log.update)
    result = vqd.compute_eigenvalues(qub_hamiltonian)

    # Get the optimizer runtime
    time = result.optimizer_times

    # Estimate the zeromode
    optimal_params = result.optimal_points
    zeromode_points = optimal_params[0]
    final_circuit = ansatz.assign_parameters(zeromode_points)
    zeromode_vqd = Statevector.from_instruction(final_circuit)
    zeromode = zeromode_vqd.data.tolist()

    # Compute the fidelity between the classical and quantum zeromodes
    zeromode_exact = np.array(zeromode_exact).reshape(len(zeromode_exact), 1)
    zeromode = np.array(zeromode).reshape(len(zeromode_exact), 1)
    fidelity = get_similarity(zeromode_exact, zeromode)

    # Get the number of function calls
    num_func_calls = len(log.values)

    return zeromode, fidelity, num_func_calls

## VQD implementation for hardware experiments

def run_vqd_on_hardware(matrix, ansatz, optimizer, seed, shots, backend):
    # Function to run VQD on IBM hardware

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

    # Initialize the Estimator primitive with backend options
    noise_model = NoiseModel.from_backend(backend)

    # Get the device characteristics
    basis_gates = noise_model.basis_gates
    coupling_map = backend.coupling_map
    backend = AerSimulator(noise_model = noise_model, coupling_map = coupling_map, \
                           basis_gates = basis_gates)

    # Set up Options()
    options = Options()
    options.execution.shots = shots
    options.resilience_level = 2
    options.seed_simulator = 1
    options.optimization_level = 3

    # Initializing the Estimator, Sampler, and fidelity
    estimator = Estimator(backend = backend, options = options)
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
              fidelity, ansatz, optimizer, k=1, initial_point=initial_point, callback=log.update)
    result = vqd.compute_eigenvalues(qub_hamiltonian)

    # Get the minimum eigenvalue
    eigenvalue = result.eigenvalues

    # Estimate the zeromode
    optimal_params = result.optimal_points
    zeromode_points = optimal_params[0]
    final_circuit = ansatz.assign_parameters(zeromode_points)
    zeromode_vqd = Statevector.from_instruction(final_circuit)
    zeromode = np.real(zeromode_vqd.data.tolist())

    # Get the number of function calls
    num_func_calls = result.cost_function_evals

    return eigenvalue, zeromode, num_func_calls

## Get the matrix and the zeromode

def get_zeromode(num_operator_qubits, V):
    # Function to get the matrix and the zeromode

    ## Computing the block diagonal representation of the Schlogl operator matrix
    # Defining parameters
    a = 1
    b = 1
    k1 = 3
    k2 = 0.6
    k3 = 0.25
    k4 = 2.95

    # Number of qubits
    num_operator_qubits = num_operator_qubits

    # Matrix dimensions
    N = 2 ** num_operator_qubits

    # Generating the basis size array
    x_axis = []

    for i in range(N):
        x_axis.append(i)

    # # Constructing the Schlogl operator for V = 1.1
    # # Get the volume array
    # start_V = 0.1
    # stop_V = 1.6
    # volume_array = get_volume_array(start_V, stop_V, num_operator_qubits)

    # # For system volume V = 1.1
    # volume_array = np.arange(0.1, 10.6, 0.1)

    # # Construct the matrix representation of the operator
    # for i, V in enumerate(volume_array):

    #     # Birth and death rates
    #     lambda_fn = lambda n: ((a*k1*n*(n-1))/V + b*k3*V)
    #     mu_fn = lambda n: ((k2*n*(n-1)*(n-2))/V**2 + n*k4)

    #     # stochastic matrix Q of dimension N x N
    #     Q = TridiagA(lambda_fn, mu_fn, N)

    #     i += 1
    # ######################################################################
    # Construct the matrix
    # Birth and death rates
    lambda_fn = lambda n: ((a*k1*n*(n-1))/V + b*k3*V)
    mu_fn = lambda n: ((k2*n*(n-1)*(n-2))/V**2 + n*k4)

    # stochastic matrix Q of dimension N x N
    Q = TridiagA(lambda_fn, mu_fn, N)

    # Print the original Schlogl operator matrix
    print('The Schlogl operator matrix is:')
    print(Q)
    print()

    # Compute the Hermitian form of the matrix
    hermitian_matrix = np.dot(Q.T, Q)

    # # Print the volume array
    # print('The volume array is:')
    # print(volume_array)
    # print()

    # Print the volume parameter for which the simulations are being run
    print('The volume parameter is:')
    print(V)
    print()

    # Print the Hermitian matrix (block diagonal form)
    print('The Hermitian form of the Schlogl matrix is:')
    print(hermitian_matrix)
    print()

   ## Get the classical zeromode
    A, P = la.eig(Q)

    A_real = np.real(A)
    index = np.where(A_real == np.amax(A_real))[0][0]

    eigenvalue = A[index]
    zeromode_classic = P[:, index]

    zeromode_classic = np.real(normalize_eigenvector(zeromode_classic))
    zeromode_classic = np.reshape(zeromode_classic, (zeromode_classic.size, 1))

    print("the available eigenvalues are: \n" + str(A))
    print()

    print("The minimum eigenvalue located is: \n" + str(eigenvalue))
    print()

    print("The minimum zeromode located is: \n" + str(np.real(zeromode_classic)))

    return Q, hermitian_matrix, zeromode_classic

## Estimate quantum resources (state vector simulations)

def estimate_resources(matrix, zeromode_classic, optimizers, num_qubits, target_fidelity_threshold):

    # Initial depth and maximum depth
    initial_depth = 1
    max_depth = 10  # Set the maximum number of depths to check

    # Initialize dictionaries to store minimum depths, zeromodes, and resources for each optimizer-ansatz pair
    min_depths = {}
    zeromodes = {}
    resource_info = {}  # To store gate count, circuit depth, and function calls

    # Loop through each optimizer
    for optimizer in optimizers:
        optimizer_name = optimizer.__class__.__name__

        # Loop through each ansatz type and configure dynamically
        for AnsatzClass in [RealAmplitudes, TwoLocal, EfficientSU2]:
            # Initialize ansatz with appropriate configurations
            if AnsatzClass == RealAmplitudes:
                ansatz = AnsatzClass(num_qubits=num_qubits, entanglement='full', reps=initial_depth)
            elif AnsatzClass == TwoLocal:
                ansatz = AnsatzClass(num_qubits=num_qubits, rotation_blocks=['ry'], entanglement_blocks='cx', reps=initial_depth)
            elif AnsatzClass == EfficientSU2:
                ansatz = AnsatzClass(num_qubits=num_qubits, su2_gates=['ry'], entanglement='sca', reps=initial_depth)

            ansatz_name = AnsatzClass.__name__
            pair_name = f"{optimizer_name}-{ansatz_name}"
            print(f"\nRunning VQD for optimizer-ansatz pair: {pair_name}")

            current_depth = initial_depth
            converged = False  # Flag to check if convergence occurs

            while current_depth <= max_depth:  # Loop for up to max_depth
                # Set `reps` (depth) for the current ansatz
                ansatz.reps = current_depth

                # Temporary storage for the fidelity results
                fidelities = []
                function_calls = []  # To track the number of function calls for each run
                all_zeromodes = []  # To store all zeromodes for the depth

                # Perform multiple independent VQE runs to calculate average fidelity
                for run in range(10):  # Number of independent runs
                    # Set a seed for a specific VQE run
                    seed = run + 1

                    # Run VQE
                    zeromode, fidelity_value, function_call_count = run_vqd(
                        matrix=matrix, ansatz=ansatz, optimizer=optimizer, seed=seed,
                        zeromode_exact=zeromode_classic)

                    # Append the fidelity result and function call count
                    fidelities.append(fidelity_value)
                    function_calls.append(function_call_count)
                    all_zeromodes.append(zeromode)

                # Calculate the average fidelity over the runs
                average_fidelity = np.mean(fidelities)
                print(f"{pair_name} - Depth {current_depth}: Average fidelity = {average_fidelity}")

                # Check if the average fidelity meets the threshold
                if average_fidelity >= target_fidelity_threshold:
                    min_depths[pair_name] = current_depth
                    converged = True

                    # Identify the run with the highest fidelity at this depth
                    best_run_index = np.argmax(fidelities)
                    best_fidelity = fidelities[best_run_index]
                    best_zeromode = all_zeromodes[best_run_index]
                    best_function_calls = function_calls[best_run_index]

                    # Print the run number and highest fidelity at optimal depth
                    print(f"Optimal depth {current_depth} for {pair_name} achieved highest fidelity = {best_fidelity}")
                    print(f"Run number with highest fidelity: {best_run_index + 1}")

                    # Calculate gate count and circuit depth for the ansatz at this depth
                    decomposed_ansatz = ansatz.decompose()  # Decompose to get actual gate operations
                    gate_count_dict = decomposed_ansatz.count_ops()
                    total_gates = sum(gate_count_dict.values())
                    circuit_depth = decomposed_ansatz.depth()

                    # Store zeromode and resource information
                    zeromodes[pair_name] = best_zeromode
                    resource_info[pair_name] = {
                        'gate_count': total_gates,
                        'circuit_depth': circuit_depth,
                        'function_calls': best_function_calls
                    }

                    print(f"Zeromode at optimal fidelity for {pair_name}: {best_zeromode}")
                    print(f"Resource estimates for {pair_name}: Gate count = {total_gates}, Circuit depth = {circuit_depth}, Function calls = {best_function_calls}")

                    break  # Exit the loop if the threshold is met

                current_depth += 1  # Increase depth and try again

            # If the loop finishes and no convergence occurs, mark as "did not converge"
            if not converged:
                min_depths[pair_name] = "Did not converge"
                zeromodes[pair_name] = "Did not converge"
                resource_info[pair_name] = {
                    'gate_count': "N/A",
                    'circuit_depth': "N/A",
                    'function_calls': "N/A"
                }
                print(f"{pair_name} did not converge within {max_depth} depths.")

    return min_depths, zeromodes, resource_info

## Run VQD on hardware for different optimizer-ansatz pairs

def schlogl_hardware_experiments(matrix, optimizers, fixed_depth, fixed_shots, backend):
    # Infer the number of qubits from the matrix dimensions
    num_qubits = int(np.log2(matrix.shape[0]))

    # Initialize dictionaries to store eigenvalues and zeromodes
    eigenvalues = {}
    zeromodes = {}
    num_function_calls = {}

    # List of ansatz classes to evaluate
    ansatz_classes = [RealAmplitudes, TwoLocal, EfficientSU2]

    # Loop through each optimizer
    for optimizer in optimizers:
        optimizer_name = optimizer.__class__.__name__

        # Loop through each ansatz class
        for AnsatzClass in ansatz_classes:
            # Configure the ansatz with the fixed depth
            if AnsatzClass == RealAmplitudes:
                ansatz = AnsatzClass(num_qubits=num_qubits, entanglement='full', reps=fixed_depth)
            elif AnsatzClass == EfficientSU2:
                ansatz = AnsatzClass(num_qubits=num_qubits, su2_gates=['ry'], entanglement='sca', reps=fixed_depth)
            if AnsatzClass == TwoLocal:
                ansatz = AnsatzClass(num_qubits=num_qubits, rotation_blocks=['ry'], entanglement_blocks='cx', reps=fixed_depth)

            ansatz_name = AnsatzClass.__name__
            pair_name = f"{optimizer_name}-{ansatz_name}"
            print(f"\nRunning VQD for optimizer-ansatz pair: {pair_name}")

            best_eigenvalue = None
            best_zeromode = None
            best_function_calls = None

            # Perform 4 independent VQD runs
            for run in range(4):
                # Set a unique seed for reproducibility
                seed = run + 1

                # Run the VQE algorithm for the specific seed
                eigenvalue, zeromode, func_calls = run_vqd_on_hardware(matrix, ansatz, optimizer, seed, fixed_shots, backend)

                # Log the current result
                print(f"Run: {run + 1} | Eigenvalue: {eigenvalue} | Zeromode: {zeromode}")

                # Update the best results if the current eigenvalue is better
                if best_eigenvalue is None or np.abs(eigenvalue) < np.abs(best_eigenvalue):
                    best_eigenvalue = eigenvalue
                    best_zeromode = zeromode
                    best_function_calls = func_calls

            # Store the best results for this optimizer-ansatz pair
            eigenvalues[pair_name] = best_eigenvalue
            zeromodes[pair_name] = best_zeromode
            num_function_calls[pair_name] = best_function_calls

            print(f"Best eigenvalue for {pair_name}: {best_eigenvalue}")
            print(f"Best zeromode for {pair_name}: {best_zeromode}")
            print(f"Function calls for best result: {best_function_calls}")

    return eigenvalues, zeromodes, num_function_calls