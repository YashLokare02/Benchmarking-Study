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

# Import the FakeManila backend
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeMontrealV2, FakeGuadalupeV2, FakeManila

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

def run_qaoa_analysis(matrix, seed, depth, exact_ground_state):

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

    return ansatz, quantum_statevector, fidelity_value, func_calls