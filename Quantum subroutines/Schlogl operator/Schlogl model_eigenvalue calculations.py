## Author: Yash Lokare

## VQD
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

def estimate_resources_first_eigenvalue_vqd(matrix, optimizers, num_qubits, target_relative_error_threshold, exact_second_eigenvalue):
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

###########################################################################################################################################################
## SSVQE 
## Importing relevant libraries
from __future__ import annotations

import logging
import warnings
from time import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from qiskit_algorithms.gradients import BaseEstimatorGradient
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, TwoLocal, EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector
from qiskit_algorithms.utils import algorithm_globals

from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit_algorithms.optimizers import Optimizer, Minimizer, OptimizerResult
from qiskit_algorithms.variational_algorithm import (
    VariationalAlgorithm,
    VariationalResult,
)
from qiskit_algorithms.eigensolvers import Eigensolver, EigensolverResult
from qiskit_algorithms.observables_evaluator import estimate_observables

## Instantiating the SSVQE class (Ref.: https://github.com/JoelHBierman/SSVQE)
logger = logging.getLogger(__name__)

class SSVQE(VariationalAlgorithm, Eigensolver):
    r"""The Subspace Search Variational Quantum Eigensolver algorithm.
    `SSVQE <https://arxiv.org/abs/1810.09434>`__ is a hybrid quantum-classical
    algorithm that uses a variational technique to find the low-lying eigenvalues
    of the Hamiltonian :math:`H` of a given system. SSVQE can be seen as
    a natural generalization of VQE. Whereas VQE minimizes the expectation
    value of :math:`H` with respect to the ansatz state, SSVQE takes a set
    of mutually orthogonal input states :math:`\{| \psi_{i}} \rangle\}_{i=0}^{k-1}`,
    applies the same parameterized ansatz circuit :math:`U(\vec\theta)` to all of them,
    then minimizes a weighted sum of the expectation values of :math:`H` with respect
    to these states.

    An instance of SSVQE requires defining four algorithmic sub-components:

    An :attr:`estimator` to compute the expectation values of operators, an integer ``k`` denoting
    the number of eigenstates that the algorithm will attempt to find, an ansatz which is a
    :class:`QuantumCircuit`, and one of the classical :mod:`~qiskit.algorithms.optimizers`.

    The ansatz is varied, via its set of parameters, by the optimizer, such that it works towards
    a set of mutually orthogonal states, as determined by the parameters applied to the ansatz,
    that will result in the minimum weighted sum of expectation values being measured
    of the input operator (Hamiltonian) with respect to these states. The weights given
    to this list of expectation values is given by ``weight_vector``. An optional array of
    parameter values, via the ``initial_point``, may be provided as the starting point for
    the search of the low-lying eigenvalues. This feature is particularly useful such as
    when there are reasons to believe that the solution point is close to a particular
    point. The length of the ``initial_point`` list value must match the number of the
    parameters expected by the ansatz being used. If the ``initial_point`` is left at the
    default of ``None``, then SSVQE will look to the ansatz for a preferred value, based
    on its given initial state. If the ansatz returns ``None``, then a random point will
    be generated within the parameter bounds set, as per above. If the ansatz provides
    ``None`` as the lower bound, then SSVQE will default it to :math:`-2\pi`; similarly,
    if the ansatz returns ``None`` as the upper bound, the default value will be :math:`2\pi`.

    An optional list of initial states, via the ``initial_states``, may also be provided.
    Choosing these states appropriately is a critical part of the algorithm. They must
    be mutually orthogonal because this is how the algorithm enforces the mutual
    orthogonality of the solution states. If the ``initial_states`` is left as ``None``,
    then SSVQE will automatically generate a list of computational basis states and use
    these as the initial states. For many physically-motivated problems, it is advised
    to not rely on these default values as doing so can easily result in an unphysical
    solution being returned. For example, if one wishes to find the low-lying excited
    states of a molecular Hamiltonian, then we expect the output states to belong to a
    particular particle-number subspace. If an ansatz that preserves particle number
    such as :class:`UCCSD` is used, then states belonging to the incorrect particle
    number subspace will be returned if the ``initial_states`` are not in the correct
    particle number subspace. A similar statement can often be made for the
    spin-magnetization quantum number.

    A minimal example of how one may initialize an instance of ``SSVQE`` and use it
    to compute the low-lying eigenvalues of an operator:

    .. code-block:: python

      from qiskit import QuantumCircuit
      from qiskit.quantum_info import Pauli
      from qiskit.primitives import Estimator
      from qiskit.circuit.library import RealAmplitudes
      from qiskit.algorithms.optimizers import SPSA
      from qiskit.algorithms.eigensolvers import SSVQE

      operator = Pauli("ZZ")
      input_states = [QuantumCircuit(2), QuantumCircuit(2)]
      input_states[0].x(0)
      input_states[1].x(1)

      ssvqe_instance = SSVQE(k=2,
                            estimator=Estimator(),
                            optimizer=SPSA(),
                            ansatz=RealAmplitudes(2),
                            initial_states=input_states)

      result = ssvqe_instance.compute_eigenvalues(operator)

    The following attributes can be set via the initializer but can also be read and
    updated once the SSVQE object has been constructed.

    Attributes:
            estimator (BaseEstimator): The primitive instance used to perform the expectation
                estimation of observables.
            k (int): The number of eigenstates that SSVQE will attempt to find.
            ansatz (QuantumCircuit): A parameterized circuit used as an ansatz for the
                wave function.
            optimizer (Optimizer): A classical optimizer, which can be either a Qiskit optimizer
                or a callable that takes an array as input and returns a Qiskit or SciPy
                optimization result.
            gradient (BaseEstimatorGradient | None): An optional estimator gradient to be used
                with the optimizer.
            initial_states (Sequence[QuantumCircuit]): An optional list of mutually orthogonal
                initial states. If ``None``, then SSVQE will set these to be a list of mutually
                orthogonal computational basis states.
            weight_vector (Sequence[float]): A 1D array of real positive-valued numbers to assign
                as weights to each of the expectation values. If ``None``, then SSVQE will default
                to [k, k-1, ..., 1].
            callback (Callable[[int, np.ndarray, Sequence[float], dict[str, Any]], None] | None): A
                function that can access the intermediate data at each optimization step.
                These data are the evaluation count, the optimizer parameters for
                the ansatz, the evaluated mean energies, and the metadata dictionary.
            check_input_states_orthogonality: A boolean that sets whether or not to check
                that the value of initial_states passed consists of a mutually orthogonal
                set of states. If ``True``, then SSVQE will check that these states are mutually
                orthogonal and return an :class:`AlgorithmError` if they are not.
                This is set to ``True`` by default, but setting this to ``False`` may be desirable
                for larger numbers of qubits to avoid exponentially large computational overhead.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        k: int | None = 2,
        ansatz: QuantumCircuit | None = None,
        optimizer: Optimizer | Minimizer | None = None,
        initial_point: Sequence[float] | None = None,
        initial_states: list[QuantumCircuit] | None = None,
        weight_vector: Sequence[float] | Sequence[int] | None = None,
        gradient: BaseEstimatorGradient | None = None,
        callback: Callable[[int, np.ndarray, Sequence[float], float], None]
        | None = None,
        check_input_states_orthogonality: bool = True,
    ) -> None:
        """
        Args:
            estimator: The estimator primitive.
            k: The number of eigenstates that the algorithm will attempt to find.
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            optimizer: A classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            initial_states: An optional list of mutually orthogonal initial states.
                If ``None``, then SSVQE will set these to be a list of mutually orthogonal
                computational basis states.
            weight_vector: An optional list or array of real positive numbers with length
                equal to the value of ``num_states`` to be used in the weighted energy summation
                objective function. This fixes the ordering of the returned eigenstate/eigenvalue
                pairs. If ``None``, then SSVQE will default to [n, n-1, ..., 1] for `k` = n.
            gradient: An optional gradient function or operator for optimizer.
            callback: A callback that can access the intermediate data at each optimization step.
                These data are: the evaluation count, the optimizer ansatz parameters,
                the evaluated mean energies, and the metadata dictionary.
            check_input_states_orthogonality: A boolean that sets whether or not to check
                that the value of ``initial_states`` passed consists of a mutually orthogonal
                set of states. If ``True``, then SSVQE will check that these states are mutually
                orthogonal and return an error if they are not. This is set to ``True`` by default,
                but setting this to ``False`` may be desirable for larger numbers of qubits to avoid
                exponentially large computational overhead before the simulation even starts.
        """

        super().__init__()

        self.k = k
        self.initial_states = initial_states
        self.weight_vector = weight_vector
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.initial_point = initial_point
        self.gradient = gradient
        self.callback = callback
        self.estimator = estimator
        self.check_initial_states_orthogonal = check_input_states_orthogonality

    @property
    def initial_point(self) -> Sequence[float] | None:
        """Returns initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Sequence[float] | None):
        """Sets initial point"""
        self._initial_point = initial_point

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def compute_eigenvalues(
        self,
        operator: BaseOperator | SparsePauliOp,
        aux_operators: ListOrDict[BaseOperator | SparsePauliOp] | None = None,
    ) -> EigensolverResult:

        ansatz = self._check_operator_ansatz(operator)

        initial_point = _validate_initial_point(self.initial_point, ansatz)

        initial_states = self._check_operator_initial_states(
            self.initial_states, operator
        )

        bounds = _validate_bounds(ansatz)

        initialized_ansatz_list = [
            initial_states[n].compose(ansatz) for n in range(self.k)
        ]

        self.weight_vector = self._check_weight_vector(self.weight_vector)

        evaluate_weighted_energy_sum = self._get_evaluate_weighted_energy_sum(
            initialized_ansatz_list, operator
        )

        if self.gradient is not None:  # need to implement _get_evaluate_gradient
            evaluate_gradient = self._get_evalute_gradient(
                initialized_ansatz_list, operator
            )
        else:
            evaluate_gradient = None

        if aux_operators:
            zero_op = SparsePauliOp.from_list([("I" * self.ansatz.num_qubits, 0)])

            # Convert the None and zero values when aux_operators is a list.
            # Drop None and convert zero values when aux_operators is a dict.
            if isinstance(aux_operators, list):
                key_op_iterator = enumerate(aux_operators)
                converted = [zero_op] * len(aux_operators)
            else:
                key_op_iterator = aux_operators.items()
                converted = {}
            for key, op in key_op_iterator:
                if op is not None:
                    converted[key] = zero_op if op == 0 else op

            aux_operators = converted

        else:
            aux_operators = None

        start_time = time()

        if callable(self.optimizer):
            optimizer_result = self.optimizer(
                fun=evaluate_weighted_energy_sum,
                x0=initial_point,
                jac=evaluate_gradient,
                bounds=bounds,
            )
        else:
            optimizer_result = self.optimizer.minimize(
                fun=evaluate_weighted_energy_sum,
                x0=initial_point,
                jac=evaluate_gradient,
                bounds=bounds,
            )

        optimizer_time = time() - start_time

        logger.info(
            "Optimization complete in %s seconds.\nFound opt_params %s",
            optimizer_time,
            optimizer_result.x,
        )

        if aux_operators is not None:
            bound_ansatz_list = [
                initialized_ansatz_list[n].bind_parameters(optimizer_result.x)
                for n in range(self.k)
            ]

            aux_values_list = [
                estimate_observables(
                    self.estimator,
                    bound_ansatz_list[n],
                    aux_operators,
                )
                for n in range(self.k)
            ]
        else:
            aux_values_list = None

        return self._build_ssvqe_result(
            optimizer_result,
            aux_values_list,
            optimizer_time,
            operator,
            initialized_ansatz_list,
        )

    def _get_evaluate_weighted_energy_sum(
        self,
        initialized_ansatz_list: list[QuantumCircuit],
        operator: BaseOperator | SparsePauliOp,
    ) -> tuple[Callable[[np.ndarray], float | list[float]], dict]:
        """Returns a function handle to evaluate the weighted energy sum at given parameters
        for the ansatz. This is the objective function to be passed to the optimizer
        that is used for evaluation.
        Args:
            initialized_anastz_list: A list consisting of copies of the ansatz initialized
                in the initial states.
            operator: The operator whose expectation value with respect to each of the
                states in ``initialzed_ansatz_list`` is being measured.
        Returns:
            Weighted expectation value sum of the operator for each parameter.
        Raises:
            AlgorithmError: If the primitive job to evaluate the weighted energy
                sum fails.
        """
        num_parameters = initialized_ansatz_list[0].num_parameters

        eval_count = 0

        def evaluate_weighted_energy_sum(parameters):
            nonlocal eval_count
            parameters = np.reshape(parameters, (-1, num_parameters)).tolist()
            batchsize = len(parameters)

            try:
                job = self.estimator.run(
                    [
                        initialized_ansatz_list[m]
                        for n in range(batchsize)
                        for m in range(self.k)
                    ],
                    [operator] * self.k * batchsize,
                    [parameters[n] for n in range(batchsize) for m in range(self.k)],
                )
                result = job.result()
                values = result.values

                energies = np.reshape(values, (batchsize, self.k))
                weighted_energy_sums = np.dot(energies, self.weight_vector).tolist()
                energies = energies.tolist()

            except Exception as exc:
                raise AlgorithmError(
                    "The primitive job to evaluate the energy failed!"
                ) from exc

            if self.callback is not None:
                metadata = result.metadata
                for params, energies_value, metadata in zip(
                    parameters, energies, metadata
                ):
                    eval_count += 1
                    self.callback(eval_count, params, energies_value, metadata)

            return (
                weighted_energy_sums[0]
                if len(weighted_energy_sums) == 1
                else weighted_energy_sums
            )

        return evaluate_weighted_energy_sum

    def _get_evalute_gradient(
        self,
        initialized_ansatz_list: list[QuantumCircuit],
        operator: BaseOperator | SparsePauliOp,
    ) -> tuple[Callable[[np.ndarray], np.ndarray]]:
        """Get a function handle to evaluate the gradient at given parameters for the ansatz.
        Args:
            initialized_ansatz_list: The list of initialized ansatz preparing the quantum states.
            operator: The operator whose energy to evaluate.
        Returns:
            A function handle to evaluate the gradient at given parameters for the initialized
            ansatz list.
        Raises:
            AlgorithmError: If the primitive job to evaluate the gradient fails.
        """

        def evaluate_gradient(parameters):
            # broadcasting not required for the estimator gradients
            try:
                job = self.gradient.run(
                    initialized_ansatz_list,
                    [operator] * self.k,
                    [parameters for n in range(self.k)],
                )
                energy_gradients = job.result().gradients
                weighted_energy_sum_gradient = sum(
                    [self.weight_vector[n] * energy_gradients[n] for n in range(self.k)]
                )
            except Exception as exc:
                raise AlgorithmError(
                    "The primitive job to evaluate the gradient failed!"
                ) from exc

            return weighted_energy_sum_gradient

        return evaluate_gradient

    def _check_circuit_num_qubits(
        self,
        operator: BaseOperator | SparsePauliOp,
        circuit: QuantumCircuit,
        circuit_type: str,
    ) -> QuantumCircuit:
        """Check that the number of qubits for the circuit passed matches
        the number of qubits  of the operator.
        """
        if operator.num_qubits != circuit.num_qubits:
            try:
                logger.info(
                    "Trying to resize %s to match operator on %s qubits.",
                    circuit_type,
                    operator.num_qubits,
                )
                circuit.num_qubits = operator.num_qubits
            except AttributeError as error:
                raise AlgorithmError(
                    f"The number of qubits of the {circuit_type} does not match the ",
                    f"operator, and the {circuit_type} does not allow setting the "
                    "number of qubits using `num_qubits`.",
                ) from error
        return circuit

    def _check_operator_ansatz(
        self, operator: BaseOperator | SparsePauliOp
    ) -> QuantumCircuit:
        """Check that the number of qubits of operator and ansatz match and that the ansatz is
        parameterized.
        """
        # set defaults
        if self.ansatz is None:
            ansatz = RealAmplitudes(num_qubits=operator.num_qubits, reps=6)
        else:
            ansatz = self.ansatz

        ansatz = self._check_circuit_num_qubits(
            operator=operator, circuit=ansatz, circuit_type="ansatz"
        )

        if ansatz.num_parameters == 0:
            raise AlgorithmError(
                "The ansatz must be parameterized, but has no free parameters."
            )

        return ansatz

    def _check_operator_initial_states(
        self,
        list_of_states: list[QuantumCircuit] | None,
        operator: BaseOperator | SparsePauliOp,
    ) -> QuantumCircuit:

        """Check that the number of qubits of operator and all the initial states match."""

        if list_of_states is None:
            initial_states = [
                QuantumCircuit(operator.num_qubits) for n in range(self.k)
            ]
            for n in range(self.k):
                initial_states[n].initialize(
                    Statevector.from_int(n, 2**operator.num_qubits)
                )

            warnings.warn(
                "No initial states have been provided to SSVQE, so they have been set to "
                "a subset of the computational basis states. This may result in unphysical "
                "results for some chemistry and physics problems."
            )

        else:
            initial_states = list_of_states
            if self.check_initial_states_orthogonal is True:
                stacked_states_array = np.hstack(
                    [np.asarray(Statevector(state)) for state in initial_states]
                )
                if not np.isclose(
                    stacked_states_array.transpose() @ stacked_states_array,
                    np.eye(self.k),
                ).any():
                    raise AlgorithmError(
                        "The set of initial states provided is not mutually orthogonal."
                    )

        for initial_state in initial_states:
            initial_state = self._check_circuit_num_qubits(
                operator=operator, circuit=initial_state, circuit_type="initial state"
            )

        return initial_states

    def _check_weight_vector(self, weight_vector: Sequence[float]) -> Sequence[float]:
        """Check that the number of weights matches the number of states."""
        if weight_vector is None:
            weight_vector = [self.k - n for n in range(self.k)]
        elif len(weight_vector) != self.k:
            raise AlgorithmError(
                "The number of weights provided does not match the number of states."
            )

        return weight_vector

    def _eval_aux_ops(
        self,
        ansatz: QuantumCircuit,
        aux_operators: ListOrDict[BaseOperator | SparsePauliOp],
    ) -> ListOrDict[tuple(complex, complex)]:
        """Compute auxiliary operator eigenvalues."""

        if isinstance(aux_operators, dict):
            aux_ops = list(aux_operators.values())
        else:
            aux_ops = aux_operators

        num_aux_ops = len(aux_ops)

        try:
            aux_job = self.estimator.run([ansatz] * num_aux_ops, aux_ops)
            aux_values = aux_job.result().values
            aux_values = list(zip(aux_values, [0] * len(aux_values)))

            if isinstance(aux_operators, dict):
                aux_values = dict(zip(aux_operators.keys(), aux_values))

        except Exception as exc:
            raise AlgorithmError(
                "The primitive job to evaluate the aux operator values failed!"
            ) from exc

        return aux_values

    def _build_ssvqe_result(
        self,
        optimizer_result: OptimizerResult,
        aux_operators_evaluated: ListOrDict[tuple[complex, tuple[complex, int]]],
        optimizer_time: float,
        operator: BaseOperator | SparsePauliOp,
        initialized_ansatz_list: list[QuantumCircuit],
    ) -> SSVQEResult:
        result = SSVQEResult()

        try:
            result.eigenvalues = (
                self.estimator.run(
                    initialized_ansatz_list,
                    [operator] * self.k,
                    [optimizer_result.x] * self.k,
                )
                .result()
                .values
            )

        except Exception as exc:
            raise AlgorithmError(
                "The primitive job to evaluate the eigenvalues failed!"
            ) from exc

        result.cost_function_evals = optimizer_result.nfev
        result.optimal_point = optimizer_result.x
        result.optimal_parameters = dict(
            zip(self.ansatz.parameters, optimizer_result.x)
        )
        result.optimal_value = optimizer_result.fun
        result.optimizer_time = optimizer_time
        result.aux_operators_evaluated = aux_operators_evaluated
        result.optimizer_result = optimizer_result

        return result

class SSVQEResult(VariationalResult, EigensolverResult):
    """SSVQE Result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals = None

    @property
    def cost_function_evals(self) -> int:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value

def _validate_initial_point(point, ansatz):
    expected_size = ansatz.num_parameters

    # try getting the initial point from the ansatz
    if point is None and hasattr(ansatz, "preferred_init_points"):
        point = ansatz.preferred_init_points
    # if the point is None choose a random initial point

    if point is None:
        # get bounds if ansatz has them set, otherwise use [-2pi, 2pi] for each parameter
        bounds = getattr(ansatz, "parameter_bounds", None)
        if bounds is None:
            bounds = [(-2 * np.pi, 2 * np.pi)] * expected_size

        # replace all Nones by [-2pi, 2pi]
        lower_bounds = []
        upper_bounds = []
        for lower, upper in bounds:
            lower_bounds.append(lower if lower is not None else -2 * np.pi)
            upper_bounds.append(upper if upper is not None else 2 * np.pi)

        # sample from within bounds
        point = algorithm_globals.random.uniform(lower_bounds, upper_bounds)

    elif len(point) != expected_size:
        raise ValueError(
            f"The dimension of the initial point ({len(point)}) does not match the "
            f"number of parameters in the circuit ({expected_size})."
        )

    return point

def _validate_bounds(ansatz):
    if hasattr(ansatz, "parameter_bounds") and ansatz.parameter_bounds is not None:
        bounds = ansatz.parameter_bounds
        if len(bounds) != ansatz.num_parameters:
            raise ValueError(
                f"The number of bounds ({len(bounds)}) does not match the number of "
                f"parameters in the circuit ({ansatz.num_parameters})."
            )
    else:
        bounds = [(None, None)] * ansatz.num_parameters

    return bounds

## Implementing the SSVQE algorithm
#############################################################################################################
# Define the SSVQE computation with varying ansatz depth
def run_ssvqe(matrix, ansatz, optimizer, seed):
    """
    Function to compute the zeromode and eigenvalues for different optimizers and ansatz depths,
    averaged over multiple independent SSVQE runs.

    Args:
        matrix (np.ndarray): The Hamiltonian matrix.
        ansatz (QuantumCircuit): The ansatz circuit.
        optimizer (Optimizer): Classical optimizer for SSVQE.
        seed (int): Seed value for reproducibility.
        eigenvalues_exact (list): List of exact eigenvalues for reference (classical values).

    Returns:
        zeromode (list): The zeromode statevector obtained from SSVQE.
        eigenvalues_vqd (list): List of eigenvalues obtained from SSVQE.
        num_func_calls (int): Number of function calls made by the optimizer.
    """

    # Define the qubit Hamiltonian
    qub_hamiltonian = SparsePauliOp.from_operator(matrix)

    # Initializing the Estimator
    estimator = Estimator()

    # Run the VQE algorithm
    @dataclass
    class SSVQELog:
        values: list = None
        parameters: list = None

        def __post_init__(self):
            self.values = []
            self.parameters = []

        # Update function to match the expected arguments
        def update(self, count, parameters, mean, _metadata):
            self.values.append(mean)
            self.parameters.append(parameters)

    log = SSVQELog()

    ssvqe = SSVQE(k=1, estimator = estimator, optimizer = optimizer, ansatz = ansatz, callback = log.update)
    result = ssvqe.compute_eigenvalues(qub_hamiltonian)

    # Get the optimizer runtime (not used in the current context, but can be logged)
    # time = result.optimizer_times

    # Extract the optimal parameters
    optimal_params = result.optimal_point
    final_circuit = ansatz.assign_parameters(optimal_params)
    zeromode_ssvqe = Statevector.from_instruction(final_circuit)
    zeromode = zeromode_ssvqe.data.tolist()

    # Extract the first and second eigenvalues from SSVQE results
    eigenvalues_ssvqe = result.eigenvalues

    # Get the number of function calls
    num_func_calls = result.cost_function_evals

    # Get the gate count and circuit depth of the variational ansatz
    decomposed_ansatz = ansatz.decompose()
    gate_count_dict = decomposed_ansatz.count_ops()
    gate_count = sum(gate_count_dict.values())
    circuit_depth = decomposed_ansatz.depth()  # Depth of the ansatz circuit

    return zeromode, eigenvalues_ssvqe, gate_count, circuit_depth, num_func_calls

## Estimating resource requirements for the SSVQE algorithm
def estimate_resources_first_eigenvalue_ssvqe(matrix, optimizers, num_qubits, target_relative_error_threshold, exact_second_eigenvalue):
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
            print(f"\nRunning SSVQE for optimizer-ansatz pair: {pair_name}")

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

                # Perform multiple independent SSVQE runs
                for run in range(10):  # 10 independent SSVQE runs
                    seed = run + 1

                    # Run VQD and capture additional metrics
                    zeromode, eigenvalues_quantum, gate_count, circuit_depth, func_calls = run_ssvqe(
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
