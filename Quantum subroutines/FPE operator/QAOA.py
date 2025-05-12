## Importing relevant libraries
import numpy as np
import scipy.linalg as la
import scipy.spatial as spat
from scipy.stats import unitary_group
from scipy.stats import moment
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.linalg import norm
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
import time
from datetime import datetime

# Libraries for implementing the VQD algorithm
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector, Gate
from qiskit.primitives import Sampler, Estimator
from qiskit_aer import AerSimulator
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
from qiskit_ibm_runtime import Estimator as EstimatorNew
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes, TwoLocal, EfficientSU2, QAOAAnsatz
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

## QAOA implementation
def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    # Function to implement the cost function (with the SLSQP optimizer)

    # Transform the observable defined on virtual qubits to physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    # Run the estimator
    job = estimator.run([ansatz], [isa_hamiltonian], [params])

    # Extract the results
    result = job.result()
    cost = result.values[0]

    # Cost function values
    objective_func_vals.append(cost)

    return cost

def run_qaoa_analysis(matrix, seed, depth, exact_ground_state, classical_expectation, nmax, L):

    # Get the Pauli-decomposed form of the operator and initialize a random seed
    qub_hamiltonian = SparsePauliOp.from_operator(matrix)

    # Initialize the Estimator primitive
    estimator = Estimator()

    # Initialize the parameters \beta and \gamma
    random_generator = np.random.default_rng(seed)
    betas_initial_guess = 2 * np.pi * random_generator.random(depth)
    gammas_initial_guess = 2 * np.pi * random_generator.random(depth)
    parameter_values_initial_guess = [*betas_initial_guess, *gammas_initial_guess]

    # Construct the QAOA circuit
    ansatz = QAOAAnsatz(cost_operator = qub_hamiltonian, reps = depth)

    # Run QAOA
    result = minimize(
            cost_func_estimator,
            parameter_values_initial_guess,
            args=(ansatz, qub_hamiltonian, estimator),
            method="SLSQP",
            tol=1e-8,
        )

    # Number of optimizer function calls
    func_calls = result.nfev

    # Extract the statevector
    optimal_params = result.x
    final_circuit = ansatz.assign_parameters(optimal_params)
    zeromode_quantum = Statevector.from_instruction(final_circuit)

    # Convert the quantum and classical zeromodes into arrays of an appropriate size
    exact_ground_state = np.array(exact_ground_state).reshape((len(exact_ground_state), 1))
    qaoa_statevector = zeromode_quantum.data.tolist()

    # Get the real and imaginary components of the statevector
    qaoa_statevector_real = np.array(np.real(qaoa_statevector)).reshape((len(qaoa_statevector), 1))
    qaoa_statevector_imag = np.array(np.imag(qaoa_statevector)).reshape((len(qaoa_statevector), 1))

    # Compute the fidelity measures for both
    fidelity_real = get_similarity(exact_ground_state, qaoa_statevector_real)
    fidelity_imag = get_similarity(exact_ground_state, qaoa_statevector_imag)

    # Decide which one to use
    if fidelity_real > fidelity_imag:
        quantum_statevector = qaoa_statevector_real
        fidelity_value = fidelity_real

    elif fidelity_imag > fidelity_real:
        quantum_statevector = qaoa_statevector_imag
        fidelity_value = fidelity_imag

    # Compute <x^2>
    x_quantum, y_quantum = get_pdf(nmax, x, dx, L, shift = 0, zeromode_qpe = quantum_statevector, normalize = True, make_even = True)
    quantum_expectation = compute_expectation_x_squared_simpson(x_quantum, y_quantum, 2)
    error = compute_errors(classical_expectation, quantum_expectation)

    return ansatz, quantum_statevector, fidelity_value, error, func_calls

def find_minimum_qaoa_layers(matrix, exact_ground_state, error_threshold, classical_expectation, nmax, L):
    """
    Dynamically finds the minimum number of QAOA layers needed to reach a target error.

    Parameters:
        matrix (Operator): The Hamiltonian operator.
        exact_ground_state (array): Classical ground state array.
        error_threshold (float): Target error threshold to reach.
        classical_expectation (float): <x^2> obtained via exact diagonalization.
        nmax (int): Basis size of Hermite polynomials. 
        L (float): Characteristic length scale. 

    Returns:
        dict: {
            'depth': optimal number of layers,
            'zeromode': best zeromode at that depth, 
            'relative error': lowest relative error at that depth, 
            'fidelity': best fidelity at that depth,
            'gate_count': total gate count of best circuit,
            'circuit_depth': depth of best circuit,
            'func_calls': number of optimizer evaluations,
        }
    """
    # Initialize
    depth = 1  # start with 1 layer
    max_depth = 10

    while depth <= max_depth:
        fidelities = []
        relative_errors = []
        results_per_seed = []

        print(f"Trying depth = {depth}...")

        for run in range(10):  # 10 independent runs
            try:
                seed = run + 1
                ansatz, quantum_statevector, fidelity, error, func_calls = run_qaoa_analysis(
                    matrix, seed, depth, exact_ground_state, classical_expectation, nmax, L
                )
                fidelities.append(fidelity)
                relative_errors.append(error)
                results_per_seed.append((quantum_statevector, fidelity, error, func_calls, ansatz, seed))
            except Exception as e:
                print(f"Depth {depth}, Seed {seed}: Error - {e}")
                continue

        if not relative_errors:
            print(f"No valid runs at depth = {depth}")
            depth += 1
            continue

        avg_relative_error = np.mean(relative_errors)
        print(f"Average relative error at depth {depth}: {avg_relative_error:.5f}")

        if avg_relative_error <= error_threshold:
            # Select best run
            best_run = min(results_per_seed, key=lambda x: x[2])
            best_zeromode, best_fidelity, lowest_error, best_func_calls, best_ansatz, best_seed = best_run

            # Print the run number and lowest relative error at optimal depth
            print(f"Optimal depth {depth} achieved lowest relative error = {lowest_error}")
            print(f"Run number with lowest relative error: {best_seed}")
            
            # Get resource estimates
            decomposed_ansatz = best_ansatz.decompose()
            gate_count_dict = decomposed_ansatz.count_ops()
            total_gates = sum(gate_count_dict.values())
            circuit_depth = decomposed_ansatz.depth()

            print(f"Zeromode at optimal relative error: {best_zeromode}")
            print(f"Resource estimates: Gate count = {total_gates}, Circuit depth = {circuit_depth}, Function calls = {best_func_calls}")

            return {
                "depth": depth,
                "zeromode": best_zeromode,
                "relative error": lowest_error, 
                "fidelity": best_fidelity,
                "gate_count": total_gates,
                "circuit_depth": circuit_depth,
                "func_calls": best_func_calls
            }

        depth += 1  # Try next depth

    # If target relative error not met within max depth
    return {
        "depth": "not converged",
        "zeromode": "not converged",
        "relative error": "not converged", 
        "fidelity": "not converged",
        "gate_count": "not converged",
        "circuit_depth": "not converged",
        "func_calls": "not converged"
    }

def run_qaoa_multiple_depths(matrix, exact_ground_state, max_depth, classical_expectation, nmax, L):
    """
    Runs QAOA for depths from 1 to max_depth.
    For each depth, performs 10 independent runs and returns data from the run
    that yields the lowest relative error compared to the classical ground state.

    Parameters:
        matrix (Operator): The Hamiltonian operator.
        exact_ground_state (array): Classical ground state array.
        max_depth (int): Maximum number of QAOA layers to try.
        classical_expectation (float): <x^2> obtained via exact diagonalization. 
        nmax (int): Basis size of Hermite polynomials. 
        L (float): Characteristic length scale. 

    Returns:
        dict: A dictionary with depth as key and result dictionary as value.
              Each result dictionary contains:
              {
                'zeromode': best zeromode at that depth,
                'lowest error': lowest relative error at that depth,  
                'gate_count': total gate count of best circuit,
                'circuit_depth': depth of best circuit,
                'func_calls': number of optimizer evaluations,
                'seed': random seed that gave the best result
              }
    """
    # Initialize 
    results_by_depth = {}

    for depth in range(1, max_depth + 1):
        print(f"\nRunning QAOA at depth = {depth}")
        fidelities = []
        relative_errors = []
        results_per_seed_fidelities = []
        results_per_seed_errors = []
        
        for run in range(10):  # 10 independent runs
            seed = run + 1
            try:
                ansatz, quantum_statevector, fidelity, error, func_calls = run_qaoa_analysis(
                    matrix, seed, depth, exact_ground_state, classical_expectation, nmax, L
                )
                fidelities.append(fidelity)
                relative_errors.append(error)
                
                results_per_seed_fidelities.append((quantum_statevector, fidelity, func_calls, ansatz, seed))
                results_per_seed_errors.append((quantum_statevector, error, func_calls, ansatz, seed))
            except Exception as e:
                print(f"Depth {depth}, Seed {seed}: Error - {e}")
                continue

        if not relative_errors:
            print(f"No valid runs at depth {depth}")
            results_by_depth[depth] = {
                "zeromode_qaoa": "error",
                "relative_error": "error", 
                "gate_count": "error",
                "circuit_depth": "error",
                "func_calls": "error",
                "seed": "error"
            }
            continue

        # Get best run
        best_run = min(results_per_seed_errors, key=lambda x: x[1]) # compare the relative errors
        best_zeromode, best_error, best_func_calls, best_ansatz, best_seed = best_run

        # Extract resource estimates
        decomposed_ansatz = best_ansatz.decompose()
        gate_count_dict = decomposed_ansatz.count_ops()
        total_gates = sum(gate_count_dict.values())
        circuit_depth = decomposed_ansatz.depth()

        # Print the highest fidelity at the corresponding depth
        print(f"Lowest relative error at depth {depth}: {float(best_error):.5f} (Seed {best_seed})")
        print(f"The best zeromode at depth {depth} is: {best_zeromode}")

        results_by_depth[depth] = {
            "zeromode_qaoa": best_zeromode,
            "lowest error": best_error,
            "gate_count": total_gates,
            "circuit_depth": circuit_depth,
            "func_calls": best_func_calls,
            "seed": best_seed
        }

    return results_by_depth
