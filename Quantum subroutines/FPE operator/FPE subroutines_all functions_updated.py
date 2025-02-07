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
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector, Gate
from qiskit.primitives import Sampler, Estimator
from qiskit_aer import AerSimulator
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
from qiskit_ibm_runtime import Estimator as EstimatorNew
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

def make_operator_even(op):
    op_new = np.zeros((op.shape[0]//2, op.shape[1]//2))

    for row in range(op_new.shape[0]):
        for col in range(op_new.shape[1]):
            op_new[row, col] = op[row*2, col * 2]

    return op_new

def get_pdf(n, x, dx, L, shift, zeromode_qpe, normalize = True, make_even = False):
    # Function to construct the ground state PDF using the VQSVD zeromode

    if not make_even:
        eigenvector = zeromode_qpe
    else:
        eigenvector_old = zeromode_qpe
        eigenvector = np.zeros(n + 1)
        for i in range(len(eigenvector_old)):
            eigenvector[2*i] = eigenvector_old[i]

    x0 = x - shift

    # Computing the PDF
    y = np.zeros(len(x0))

    for i in range(len(x0)):
        states = state_n(nmax, x0[i], L)
        y[i] += (np.dot(states, eigenvector))

    if normalize:
        y = normalize_probability(y, dx)

    return x0, y

def compute_expectation_x_squared_simpson(x, y, n):
    """
    Computes the expectation value of x^2 using Simpson's rule for numerical integration.

    Parameters:
    x (array-like): Discrete values of x.
    y (array-like): Corresponding values of the probability density function (PDF) at x.

    Returns:
    float: The expectation value of x^2.
    """
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Compute x^2
    x_squared = x**n

    # Check if the number of intervals is even, if not make it even by truncating the last point
    if len(x) % 2 == 0:
        x = x[:-1]
        y = y[:-1]
        x_squared = x_squared[:-1]

    # Compute the integral using Simpson's rule
    h = (x[-1] - x[0]) / (len(x) - 1)
    integral = y[0] * x_squared[0] + y[-1] * x_squared[-1] + \
               4 * np.sum(y[1:-1:2] * x_squared[1:-1:2]) + \
               2 * np.sum(y[2:-2:2] * x_squared[2:-2:2])
    integral *= h / 3

    return integral

def get_pdf(n, x, dx, L, shift, zeromode_qpe, normalize = True, make_even = False):
    # Function to construct the ground state PDF using the VQSVD zeromode

    if not make_even:
        eigenvector = zeromode_qpe
    else:
        eigenvector_old = zeromode_qpe
        eigenvector = np.zeros(n + 1)
        for i in range(len(eigenvector_old)):
            eigenvector[2*i] = eigenvector_old[i]

    x0 = x - shift

    # Computing the PDF
    y = np.zeros(len(x0))

    for i in range(len(x0)):
        states = state_n(nmax, x0[i], L)
        y[i] += (np.dot(states, eigenvector))

    if normalize:
        y = normalize_probability(y, dx)

    return x0, y

# Fidelity measure 1
def get_fidelity(zeromode_classic, zeromode_vqe):
    # Function to compute the infidelity

    overlap = np.dot(np.transpose(zeromode_vqe), zeromode_classic)
    fidelity = 1 - overlap ** 2
    return fidelity

# Fidelity measure 2
def get_similarity(a, b):
    # Function to compute the similarity between 2 zeromodes

    numerator = np.abs(np.dot(a.conj().T, b))**2
    denominator = np.linalg.norm(a)**2 * np.linalg.norm(b)**2

    return numerator / denominator

def compute_errors(expect_classical, expect_quantum):

    error = np.abs(expect_classical - expect_quantum) / expect_classical
    return error

## VQE implementation (state vector simulations)

def run_vqe_ansatz_analysis(matrix, ansatz, optimizer, seed, exact_ground_state):

    # Get the Pauli-decomposed form of the operator
    qub_hamiltonian = SparsePauliOp.from_operator(matrix)
    dimension = matrix.shape[0]
    num_qubits = int(np.log2(dimension))

    # Set up the random initial point
    np.random.seed(seed)
    initial_point = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    # Initialize the Estimator primitive
    estimator = Estimator()

    # Logging class for VQE callback
    @dataclass
    class VQELog:
        parameters: list
        values: list
        def update(self, count, parameters, mean, _metadata):
            self.values.append(mean)
            self.parameters.append(parameters)

    log = VQELog([], [])

    # Run VQE with the given ansatz and optimizer
    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point, callback=log.update)

    # Get the VQE results
    result = vqe.compute_minimum_eigenvalue(qub_hamiltonian)

    # Get the number of optimizer function calls
    num_calls = len(log.values)

    # Extract the optimal parameters and construct the state vector
    optimal_params = result.optimal_point
    final_circuit = ansatz.assign_parameters(optimal_params)
    vqe_statevector = Statevector.from_instruction(final_circuit)

     # Convert the quantum and classical zeromodes into 4 x 1 arrays
    exact_ground_state = np.array(exact_ground_state).reshape((len(exact_ground_state), 1))
    vqe_statevector = vqe_statevector.data.tolist()

    if len(exact_ground_state) == 6:
        vqe_statevector = vqe_statevector[:6]
        zeromode = np.array(vqe_statevector).reshape((len(exact_ground_state), 1))
        zeromode = np.real(zeromode)

        # Compute the fidelity measure
        fidelity_value = get_similarity(exact_ground_state, zeromode)

    else:
        zeromode = np.array(vqe_statevector).reshape((len(exact_ground_state), 1))
        zeromode = np.real(zeromode)

        # Compute the fidelity measure
        fidelity_value = get_similarity(exact_ground_state, zeromode)

    return zeromode, fidelity_value, num_calls

## VQE implementation (hardware experiments)

def run_vqe_on_hardware(matrix, ansatz, optimizer, seed, shots, backend):

    # Get the Pauli-decomposed form of the operator
    qub_hamiltonian = SparsePauliOp.from_operator(matrix)
    dimension = matrix.shape[0]
    num_qubits = int(np.log2(dimension))

    # Set up the random initial point
    np.random.seed(seed)
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

    # Initialize the Estimator primitive
    estimator = Estimator(backend = backend, options = options)

    # Logging class for VQE callback
    @dataclass
    class VQELog:
        parameters: list
        values: list
        def update(self, count, parameters, mean, _metadata):
            self.values.append(mean)
            self.parameters.append(parameters)

    log = VQELog([], [])

    # Run VQE with the given ansatz and optimizer
    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point, callback=log.update)

    # Get the VQE results (including the eigenvalue)
    result = vqe.compute_minimum_eigenvalue(qub_hamiltonian)
    eigenvalue = result.eigenvalue.real

    # Extract the optimal parameters and construct the state vector
    optimal_params = result.optimal_point
    final_circuit = ansatz.assign_parameters(optimal_params)
    vqe_statevector = Statevector.from_instruction(final_circuit)
    vqe_statevector = np.real(vqe_statevector.data.tolist())

    # Get the number of function calls
    function_calls = result.cost_function_evals

    return eigenvalue, vqe_statevector, function_calls

## Get the matrix and the zeromode

def get_zeromode(nmax, a, c, L, gamma):
    # Function to construct the matrix and get the zeromode

    #3, 5, 1, 1
    dx = 0.01
    x = np.linspace(-4, 4, int((8)/dx))

    ## Finding the zeromode through diagonalization
    op_nonhermitian = create_operator_perturbed(nmax, L, a, c, gamma)
    only_even = True

    # Matrix
    cache_diagonalization = find_zeromode(op_nonhermitian, nmax, x, dx, L, which = "single", only_even = only_even)
    matrix = cache_diagonalization['operator']

    # Get the classical zeromode
    A, P = la.eig(matrix)

    # Get the zeromode
    A_real = np.real(A)
    index = np.where(A_real == np.amin(A_real))[0][0]

    eigenvalue = A[index]
    zeromode_classic = P[:, index]

    zeromode_classic = np.real(normalize_eigenvector(zeromode_classic))
    zeromode_classic = np.reshape(zeromode_classic, (zeromode_classic.size, 1))

    return matrix, zeromode_classic

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
            print(f"\nRunning VQE for optimizer-ansatz pair: {pair_name}")

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
                    zeromode, fidelity_value, function_call_count = run_vqe_ansatz_analysis(
                        matrix=matrix, ansatz=ansatz, optimizer=optimizer, seed=seed,
                        exact_ground_state=zeromode_classic)

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

## Running for hardware experiments

def recover_eigenvalues_and_zeromodes(matrix, optimizers, fixed_depth, fixed_shots, backend):
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
            print(f"\nRunning VQE for optimizer-ansatz pair: {pair_name}")

            best_eigenvalue = None
            best_zeromode = None
            best_function_calls = None

            # Perform 10 independent VQE runs
            for run in range(10):
                # Set a unique seed for reproducibility
                seed = run + 1

                # Run the VQE algorithm for the specific seed
                eigenvalue, zeromode, func_calls = run_vqe_on_hardware(matrix, ansatz, optimizer, seed, fixed_shots, backend)

                # Log the current result
                print(f"Run {run + 1} | Eigenvalue: {eigenvalue} | Zeromode: {zeromode}")

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

## Running VQE for multiple ansatz depths (no benchmarks set) --- state vector simulations

def analyze_ansatz_performance(matrix, zeromode_classic, optimizer, num_qubits, max_depth):

    # Initialize dictionaries to store results
    zeromodes = {}
    metrics = {}  # To store fidelity and function call counts

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
        print(f"\nRunning VQE for optimizer-ansatz pair: {pair_name}")

        # Initialize dictionaries to store depth-specific results
        zeromodes[pair_name] = {}
        metrics[pair_name] = {}

        for current_depth in range(1, max_depth + 1):
            # Set `reps` (depth) for the current ansatz
            ansatz.reps = current_depth

            # Temporary storage for the fidelity results
            fidelities = []
            function_calls = []  # To track the number of function calls for each run
            all_zeromodes = []  # To store all zeromodes for the depth

            # Perform multiple independent VQE runs to calculate fidelity and resource usage
            for run in range(6):  # Number of independent runs
                seed = run + 1  # Set a unique seed for each run

                # Run VQE
                zeromode, fidelity_value, function_call_count = run_vqe_ansatz_analysis(
                    matrix=matrix, ansatz=ansatz, optimizer=optimizer, seed=seed,
                    exact_ground_state=zeromode_classic)

                # Append results
                fidelities.append(fidelity_value)
                function_calls.append(function_call_count)
                all_zeromodes.append(zeromode)

            # Identify the run with the highest fidelity at this depth
            best_run_index = np.argmax(fidelities)
            best_fidelity = fidelities[best_run_index]
            best_zeromode = all_zeromodes[best_run_index]
            best_function_calls = function_calls[best_run_index]

            # Store results for the current depth
            zeromodes[pair_name][current_depth] = best_zeromode
            metrics[pair_name][current_depth] = {
                'fidelity': best_fidelity,
                'function_calls': best_function_calls
            }

            print(f"{pair_name} - Depth {current_depth}: Best fidelity = {best_fidelity}, Function calls = {best_function_calls}, \
            Best zeromode: {best_zeromode}")

    return zeromodes, metrics

## Theoretical estimation of the state fidelities
### Extracting the number of single- and two-qubit gates in the variational ansatz

def compute_gate_counts(ansatz_type, num_qubits, depth):
    """
    Compute the number of single and two-qubit gates for a given ansatz.

    Args:
        ansatz_type (str): Type of ansatz ('RealAmplitudes', 'TwoLocal', 'EfficientSU2').
        num_qubits (int): Number of qubits.
        depth (int): Depth of the ansatz.

    Returns:
        dict: Dictionary containing the number of single and two-qubit gates.
    """
    # Define the ansatz based on the type
    if ansatz_type == "RealAmplitudes":
        ansatz = RealAmplitudes(num_qubits, entanglement='full', reps=depth)
    elif ansatz_type == "TwoLocal":
        ansatz = TwoLocal(num_qubits, rotation_blocks=['ry'], entanglement_blocks='cx', reps=depth)
    elif ansatz_type == "EfficientSU2":
        ansatz = EfficientSU2(num_qubits, su2_gates = ['ry'], entanglement = 'sca', reps=depth)
    else:
        raise ValueError("Invalid ansatz type. Choose from 'RealAmplitudes', 'TwoLocal', or 'EfficientSU2'.")

    # Decompose the ansatz
    decomposed_ansatz = ansatz.decompose()

    # Count single and two-qubit gates
    single_qubit_gates = sum(1 for gate in decomposed_ansatz.data if isinstance(gate[0], Gate) and gate[0].num_qubits == 1)
    two_qubit_gates = sum(1 for gate in decomposed_ansatz.data if isinstance(gate[0], Gate) and gate[0].num_qubits == 2)

    return {
        "single_qubit_gates": single_qubit_gates,
        "two_qubit_gates": two_qubit_gates,
    }

### Computing the theoretical state fidelities (post-processing)

def compute_fidelity(single_qubit_gates, two_qubit_gates, n_qubits):
    e_q1 = 1e-3  # Single-qubit gate error
    e_q2 = 1e-2  # Two-qubit gate error
    e_q = 1e-2   # SPAR error
    F = (1 - e_q1)**single_qubit_gates * (1 - e_q2)**two_qubit_gates * (1 - e_q)**n_qubits

    return F
