## Importing relevant libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import pandas as pd

## Functions to extract data and plot

def plot_metric(data, ax, metric_name, ylabel):

    # Extract the optimizers, ansatze, and the specified metric
    optimizers = ['SLSQP', 'P_BFGS', 'NELDER_MEAD', 'POWELL']
    ansatze = ['RealAmplitudes', 'TwoLocal', 'EfficientSU2']
    colors = ['blue', 'red', 'purple']  # Color map for ansatze
    markers = {'RealAmplitudes': 'o', 'TwoLocal': 's', 'EfficientSU2': '^'}

    # Data organization by optimizer and ansatz
    optimizer_data = {opt: {ansatz: [] for ansatz in ansatze} for opt in optimizers}
    failed_data = {opt: {ansatz: [] for ansatz in ansatze} for opt in optimizers}

    for entry in data:
        optimizer, value = entry
        opt, ansatz = optimizer.split('-')

        if value == 0:  # Mark as 'Did not converge'
            failed_data[opt][ansatz].append(entry)
        else:
            optimizer_data[opt][ansatz].append(value)

    # Plotting the metric for each optimizer and ansatz
    x_positions = np.arange(len(optimizers))
    offset = 0.1  # Base horizontal offset for non-converging cases
    vertical_offset = 0.5  # Vertical offset for crosses

    # Plot the failed cases (crosses) first
    for i, opt in enumerate(optimizers):
        for j, ansatz in enumerate(ansatze):
            failed_cases = failed_data[opt][ansatz]

            # Plot the non-converging cases
            if failed_cases:
                for k, _ in enumerate(failed_cases):
                    ax.scatter(
                        x_positions[i] + offset * (j - 1),  # Horizontal offset based on ansatz index
                        0 + vertical_offset * (k - len(failed_cases) // 2),  # Vertical offset to spread crosses
                        color=colors[j], marker='x', s=100, label=f'{ansatz} (NC)' if i == 0 and j == 0 else ""
                    )

    # Plot the metric values for converging cases
    for i, opt in enumerate(optimizers):
        for j, ansatz in enumerate(ansatze):
            values = optimizer_data[opt][ansatz]

            # Plot the metric values with specific markers for ansatze
            if values:
                ax.plot(
                    [x_positions[i]], values,
                    marker=markers[ansatz], markersize=9, linestyle='-', color=colors[j],
                    label=ansatz if i == 0 else ""  # Only add the ansatz label once
                )

    # Set axis properties
    ax.set_xticks(x_positions)
    ax.set_xticklabels(optimizers)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=17)
    ax.grid(True)
    ax.legend(fontsize=15)

## Same, but now plotting on a log scale
def plot_metric_log(data, ax, metric_name, ylabel):

    # Extract the optimizers, ansatze, and the specified metric
    optimizers = ['SLSQP', 'P_BFGS', 'NELDER_MEAD', 'POWELL']
    ansatze = ['RealAmplitudes', 'TwoLocal', 'EfficientSU2']
    colors = ['blue', 'red', 'purple']  # Color map for ansatze
    markers = {'RealAmplitudes': 'o', 'TwoLocal': 's', 'EfficientSU2': '^'}

    # Data organization by optimizer and ansatz
    optimizer_data = {opt: {ansatz: [] for ansatz in ansatze} for opt in optimizers}
    failed_data = {opt: {ansatz: [] for ansatz in ansatze} for opt in optimizers}

    for entry in data:
        optimizer, value = entry
        opt, ansatz = optimizer.split('-')

        if value == 0:  # Mark as 'Did not converge'
            failed_data[opt][ansatz].append(entry)
        else:
            optimizer_data[opt][ansatz].append(value)

    # Determine the minimum positive value for plotting
    min_positive_value = min(
        value for opt_data in optimizer_data.values()
        for ansatz_data in opt_data.values()
        for value in ansatz_data if value > 0
    )
    nc_marker_value = min_positive_value * 0.1  # Marker for non-converging cases

    x_positions = np.arange(len(optimizers))
    offset = 0.1  # Base horizontal offset for non-converging cases
    vertical_offset = 0.5  # Vertical offset for crosses

    # Plot the failed cases (crosses) first
    for i, opt in enumerate(optimizers):
        for j, ansatz in enumerate(ansatze):
            failed_cases = failed_data[opt][ansatz]

            # Plot the non-converging cases
            if failed_cases:
                for k, _ in enumerate(failed_cases):
                    ax.scatter(
                        x_positions[i] + offset * (j - 1),  # Horizontal offset based on ansatz index
                        nc_marker_value * (1 + 0.1 * k),  # Small vertical offset for spreading
                        color=colors[j], marker='x', s=100, label=f'{ansatz} (NC)' if i == 0 and j == 0 else ""
                    )

    # Plot the metric values for converging cases
    for i, opt in enumerate(optimizers):
        for j, ansatz in enumerate(ansatze):
            values = optimizer_data[opt][ansatz]

            # Plot the metric values with specific markers for ansatze
            if values:
                ax.plot(
                    [x_positions[i]], values,
                    marker=markers[ansatz], markersize=9, linestyle='-', color=colors[j],
                    label=ansatz if i == 0 else ""  # Only add the ansatz label once
                )

    # Set axis properties
    ax.set_xticks(x_positions)
    ax.set_xticklabels(optimizers)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=17)
    ax.grid(True)
    ax.legend(fontsize=15)

    # Set y-axis to log scale
    ax.set_yscale('log')

    # Adjust y-axis limits to include non-converging marker
    ax.set_ylim(bottom=nc_marker_value * 0.5)

# Function to extract data
def extract_metrics(data):
    """
    Extracts gate count, circuit depth, and number of function calls for different ansatz-optimizer pairs.
    If the optimal solution was not found (did not converge), assigns a value of 0.

    Parameters:
        data (dict): A dictionary where keys are ansatz-optimizer pairs (e.g., "RealAmplitudes-SLSQP") and
                     values are dictionaries with keys "gate_count", "circuit_depth", "num_function_calls".

    Returns:
        tuple: Three lists - gate_counts, circuit_depths, function_calls
               Each list contains tuples of the form (ansatz-optimizer, metric_value).
    """
    gate_counts = []
    circuit_depths = []
    function_calls = []

    # Iterate through the dictionary and extract the metrics
    for ansatz_optimizer, metrics in data.items():
        # Extract the metrics for each ansatz-optimizer pair, assigning 0 if not found or if 'No solution'
        gate_count = metrics.get("gate_count", 0)
        circuit_depth = metrics.get("circuit_depth", 0)
        num_function_calls = metrics.get("function_calls", 0)

        # Check for 'No solution' and set to 0 if necessary
        if gate_count == 'N/A':
            gate_count = 0
        if circuit_depth == 'N/A':
            circuit_depth = 0
        if num_function_calls == 'N/A':
            num_function_calls = 0

        # Append the extracted or default (0) values to their respective lists
        gate_counts.append((ansatz_optimizer, gate_count))
        circuit_depths.append((ansatz_optimizer, circuit_depth))
        function_calls.append((ansatz_optimizer, num_function_calls))

    return gate_counts, circuit_depths, function_calls

## Function to extract data to compare the relative errors in <x^2>

def extract_relative_errors(error_dict):
    # Define the optimizers and ansatze in the required order
    optimizers = ['SLSQP', 'P_BFGS', 'NELDER_MEAD', 'POWELL']
    ansatze = ['RealAmplitudes', 'TwoLocal', 'EfficientSU2']

    # Initialize the list for storing relative errors
    relative_errors = []

    # Loop through each optimizer and ansatz combination
    for optimizer in optimizers:
        for ansatz in ansatze:
            key = f"{optimizer}-{ansatz}"
            # Retrieve the error value from the dictionary or assign 0 if not found
            error_value = error_dict.get(key, 0)
            relative_errors.append((key, error_value))

    return relative_errors

## Function to compare performance metrics (SLSQP vs. P-BFGS)

def plot_metrics(ax, optimizer, metric_type, optimizer_label):
    # Plot gate count, circuit depth, and function calls for each ansatz type
    ansatz_types = ['RealAmplitudes', 'TwoLocal', 'EfficientSU2']
    colors = ['blue', 'red', 'purple']

    for i, ansatz in enumerate(ansatz_types):
        # Prepare data for each ansatz type
        gate_counts = []
        depths = []
        function_calls = []

        for N in N_values:
            key = f"{optimizer}-{ansatz}"
            gate_counts.append(metrics[N][key]['gate_count'])
            depths.append(metrics[N][key]['circuit_depth'])
            function_calls.append(metrics[N][key]['function_calls'])

        # Choose the corresponding metric (gate count, circuit depth, or function calls)
        if metric_type == 'gate_count':
            ax.plot(N_values, gate_counts, label=ansatz, color=colors[i], marker='o')
        elif metric_type == 'circuit_depth':
            ax.plot(N_values, depths, label=ansatz, color=colors[i], marker='o')
        elif metric_type == 'function_calls':
            ax.plot(N_values, function_calls, label=ansatz, color=colors[i], marker='o')

    ax.set_ylabel(f"{metric_type.replace('_', ' ').title()} [{optimizer_label}]", fontsize=18)
    ax.grid(True)
    ax.legend(fontsize=18)

## Function to estimate matrix sparsity

def estimate_sparsity(matrix):
    # Check if the matrix is a SciPy sparse matrix
    if issparse(matrix):
        non_zero_elements = matrix.count_nonzero()
        total_elements = matrix.shape[0] * matrix.shape[1]
    else:
        # For a NumPy array or dense matrix
        non_zero_elements = np.count_nonzero(matrix)
        total_elements = matrix.size

    sparsity = 1.0 - (non_zero_elements / total_elements)

    return sparsity

## Code to compare the performance metrics at multiple ansatz depths --- once data is available and stored appropriately

# Plot setup
fig, axes = plt.subplots(3, 4, figsize=(18, 14), constrained_layout=True, dpi=600)
layer_array = [1, 2, 3, 4, 5, 6]
colors = {'RealAmplitudes': 'blue', 'TwoLocal': 'red', 'EfficientSU2': 'purple'}
markers = {'RealAmplitudes': 'o', 'TwoLocal': 's', 'EfficientSU2': '^'}  # Circle, square, triangle

titles = {
    (1, 2): r'$a=1, b=2, \ell=1/2, \Gamma=1$',
    (-1, 2): r'$a=-1, b=2, \ell=1/2, \Gamma=1$',
    (1, 0): r'$a=1, b=0, \ell=1/2, \Gamma=1$',
}

# Store handles and labels for the common legend
handles, labels = [], []

# Loop through data for each parameter set
for row_idx, (params, ansatz_data) in enumerate(data.items()):
    for col_idx, N in enumerate([2, 4, 8, 16]):
        ax = axes[row_idx, col_idx]

        for ansatz, errors in ansatz_data.items():
            if N in errors:  # Ensure data exists for this N
                line, = ax.plot(
                    layer_array, errors[N],
                    marker=markers[ansatz], markersize=8, label=ansatz, color=colors[ansatz]
                )

                # Store a single instance of each handle/label for the common legend
                if row_idx == 0 and col_idx == 0:
                    handles.append(line)
                    labels.append(ansatz)

        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=18)

        # x and y axis labels
        if row_idx == 2:
            ax.set_xlabel('Number of layers', fontsize=20)
        if col_idx == 0:
            ax.set_ylabel(r'$\langle x^2 \rangle_{\mathrm{relative~error}}$', fontsize=20)
        if row_idx == 0:
            ax.set_title(fr'$N = {N}$', fontsize=20)

        ax.grid(True)

    # Add row titles
    axes[row_idx, 0].text(-0.5, 0.5, titles[params], va='center', ha='right',
                          fontsize=20, rotation=90, transform=axes[row_idx, 0].transAxes)

# Add a common legend at the top of the figure
fig.legend(handles, labels, loc='upper center', fontsize=18, ncol=3, markerscale=1.5, frameon=False, bbox_to_anchor = (0.5, 1.05))

# Save the plot
plt.savefig('Relative errors_multiple depths_FPE.pdf', bbox_inches = 'tight', dpi = 600)