## Importing relevant libraries

import numpy as np
import matplotlib.pyplot as plt

## Function to plot a particular performance metric

def convert_to_log_scale(values):
    """
    Convert values to a representation in powers of 10 (e.g., 0.04 becomes 10^{-2}).

    Args:
        values (list or np.array): List or array of numerical values.

    Returns:
        list: List of values in the format 10^{exponent} as strings.
    """
    powers_of_10 = []
    for value in values:
        if value == 0:
            powers_of_10.append("0")  # Handle zero separately
        else:
            exponent = int(np.floor(np.log10(abs(value))))
            powers_of_10.append(f"$10^{{{exponent}}}$")

    return powers_of_10

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
                    [x_positions[i]], np.abs(values),
                    marker=markers[ansatz], markersize=9, linestyle='-', color=colors[j],
                    label=ansatz if i == 0 else ""  # Only add the ansatz label once
                )

    # Set axis properties
    ax.set_xticks(x_positions)
    ax.set_xticklabels(optimizers)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.grid(True)
    ax.legend(fontsize=15)

    # Set y-axis to log scale
    ax.set_yscale('log')

    # # Adjust y-axis limits to include non-converging marker
    # ax.set_ylim(bottom=nc_marker_value * 0.05)

## Function to compare the performance metrics of different optimizer -- ansatz pairs at multiple ansatz depths (once the data has been collected and stored appropriately)

# Function to plot the data
def plot_data():
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi = 600)  # 3 rows, 4 columns

    # Define the x-axis values (array [1, 2, 3, 4, 5, 6])
    x_values = np.array([1, 2, 3, 4, 5, 6])

    # Prepare labels for the common legend
    lines = []
    labels = []

    # For each system volume (V = 1.1, 5.5, 10.5)
    for i, volume in enumerate(["V1.1", "V5.5", "V10.5"]):
        for j, qubit in enumerate([2, 3]):
            ax = axes[i, 2*j]  # Expectation values
            ax2 = axes[i, 2*j + 1]  # Relative errors

            # Prepare data for expectation values and relative errors
            exp_vals_2q = data[volume][f"{qubit}q"]["expectation_values"]["RealAmplitudes"]
            exp_vals_3q = data[volume][f"{qubit}q"]["expectation_values"]["TwoLocal"]
            exp_vals_4q = data[volume][f"{qubit}q"]["expectation_values"]["EfficientSU2"]

            rel_errors_2q = data[volume][f"{qubit}q"]["relative_errors"]["RealAmplitudes"]
            rel_errors_3q = data[volume][f"{qubit}q"]["relative_errors"]["TwoLocal"]
            rel_errors_4q = data[volume][f"{qubit}q"]["relative_errors"]["EfficientSU2"]

            # Ensure positive values for log scale plotting
            exp_vals_2q = np.abs(exp_vals_2q)  # Absolute values
            exp_vals_3q = np.abs(exp_vals_3q)
            exp_vals_4q = np.abs(exp_vals_4q)

            # Plot expectation values
            line1, = ax.plot(x_values, exp_vals_2q, label="RealAmplitudes", color=colors["RealAmplitudes"], marker='o')
            line2, = ax.plot(x_values, exp_vals_3q, label="TwoLocal", color=colors["TwoLocal"], marker='s')
            line3, = ax.plot(x_values, exp_vals_4q, label="EfficientSU2", color=colors["EfficientSU2"], marker='^')
            lines.extend([line1, line2, line3])

            ax.set_yscale('log')  # Log scale
            ax.set_title(fr"$N_{{\mathrm{{qubits}}}} = {qubit} ~(V = {volume[1:]})$", fontsize=17)
            ax.set_xlabel("Number of layers", fontsize=19) if i == 2 else ax.set_xlabel("")
            ax.set_ylabel(r"$\langle Q \rangle$", fontsize=19)
            ax.tick_params(axis='both', labelsize=16)
            ax.grid(True)  # Add grid lines

            # Plot relative errors
            ax2.plot(x_values, rel_errors_2q, label="RealAmplitudes", color=colors["RealAmplitudes"], marker='o')
            ax2.plot(x_values, rel_errors_3q, label="TwoLocal", color=colors["TwoLocal"], marker='s')
            ax2.plot(x_values, rel_errors_4q, label="EfficientSU2", color=colors["EfficientSU2"], marker='^')

            ax2.set_yscale('log')  # Log scale
            ax2.set_title(fr"$N_{{\mathrm{{qubits}}}} = {qubit} ~(V = {volume[1:]})$", fontsize=17)
            ax2.set_xlabel("Number of layers", fontsize=19) if i == 2 else ax2.set_xlabel("")
            ax2.set_ylabel(r"$\lambda_{1, \mathrm{relative ~error}}$", fontsize=19)
            ax2.tick_params(axis='both', labelsize=16)
            ax2.grid(True)  # Add grid lines

    # Add common legend at the top of the figure with an adjusted position
    fig.legend(lines, ["RealAmplitudes", "TwoLocal", "EfficientSU2"], loc="upper center", ncol=3, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 1.05))

    # Save the plot
    plt.tight_layout(pad=1.0)
    plt.savefig('Schlogl_multiple depths_zeromodes_eigenvalues.pdf', bbox_inches = 'tight', dpi = 600)

## Visualizing hardware experimental results
### Code to compare the hardware (experimental) solution infidelities and steady-state expectation values

# Infidelity calculation
def calculate_infidelity(fidelity):
    return 1 - fidelity**2

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(18, 10), dpi = 600)
colors = ['blue', 'red', 'purple']
markers = ['o', 's', '^']
ansatz_labels = ['RealAmplitudes', 'TwoLocal', 'EfficientSU2']

# First row: Infidelity plots
for i, (V, data) in enumerate(fidelities.items()):
    ax = axs[0, i]
    infidelities = {
        "SLSQP": [calculate_infidelity(data["SLSQP"][0]), calculate_infidelity(data["SLSQP"][1]), calculate_infidelity(data["SLSQP"][2])],
        "SPSA": [calculate_infidelity(data["SPSA"][0]), calculate_infidelity(data["SPSA"][1]), calculate_infidelity(data["SPSA"][2])]
    }
    x_positions = np.arange(len(infidelities))

    for j, optimizer in enumerate(["SLSQP", "SPSA"]):
        for k, infidelity in enumerate(infidelities[optimizer]):
            ax.scatter(
                x_positions[j] + k * 0.2 - 0.1,
                infidelity,
                color=colors[k],
                marker=markers[k],
                label=ansatz_labels[k] if j == 0 else None,
                s=80
            )

    ax.set_yscale('log')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["SLSQP", "SPSA"], fontsize=17)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_title(rf"$V = {V}$", fontsize=20)
    ax.tick_params(axis='y', labelsize=17)
    ax.grid(True)
    if i == 0:
        ax.set_ylabel(r"$1 - |\langle\psi_{0, \mathrm{classical}}|\psi_{0, \mathrm{VQD}}\rangle|^2$", fontsize=19)
    ax.legend(fontsize=16, loc='lower left', frameon=True)

# Second row: Expectation value plots
for i, (V, data) in enumerate(expectations.items()):
    ax = axs[1, i]
    x_positions = np.arange(len(data))

    for j, optimizer in enumerate(["SLSQP", "SPSA"]):
        for k, expectation in enumerate(data[optimizer]):
            ax.scatter(
                x_positions[j] + k * 0.2 - 0.1,
                expectation,
                color=colors[k],
                marker=markers[k],
                label=ansatz_labels[k] if j == 0 else None,
                s=80
            )

    ax.set_yscale('log')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["SLSQP", "SPSA"], fontsize=17)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.tick_params(axis='y', labelsize=17)
    ax.grid(True)
    if i == 0:
        ax.set_ylabel(r"$\langle Q \rangle$", fontsize=19)
    ax.legend(fontsize=16, loc='lower left', frameon=True)

# Add a common title
fig.suptitle(r"$N_{\mathrm{qubits}} = 2 ~(7.5 \times 10^4 ~\mathrm{shots}); \mathrm{Manila ~machine}$", fontsize=22, y = 0.94)

# Adjust layout and save the plot
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('Hardware noise analysis_Schlogl.pdf', bbox_inches = 'tight', dpi = 600)