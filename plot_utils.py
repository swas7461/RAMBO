import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_layer_sensitivity(df, model_id, loss_baseline, sensitive_layer_type):
    df = df[df['percentage_of_weights_flipped'] == 0.1]
    df = df[df['Layer_type'] == sensitive_layer_type]

    df_mag = df[df['Attack_type'] == 'Hybrid']
    df_mag = df_mag.reset_index()

    plt.rcParams.update({'font.size':20})
    plt.figure(figsize=(8,6))

    plt.plot(df_mag.index, df_mag['Model_loss'], marker='+', label='Post Attack Loss', color = 'red')
    plt.fill_between(df_mag.index, df_mag['Model_loss'], color='red', alpha=0.2)
    plt.axhline(y=loss_baseline, color='g', linestyle='--', linewidth=2,  label='Baseline loss')
    plt.xlabel('Layer Id')
    plt.ylabel('Model Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'./plot_results/{model_id}_layer_sensitivity.pdf')

def plot_layer_selection_loss(df, model_id, loss_baseline, selected_layers, selected_layers_new):
    df = df[df['percentage_of_weights_flipped'] == 1]
    df = df[df['Attack_type'] == 'Hybrid']

    for i in range(len(selected_layers)):
        df['Layer_type'][df['Layer_type'] == selected_layers[i]] = selected_layers_new[i]

    plt.rcParams.update({'font.size':18})
    # Create the box plot
    plt.figure(figsize=(8, 6))

    # Filter DataFrame for selected layers
    # Prepare the data for boxplot
    data = [df[df['Layer_type'] == layer]['Model_loss'] for layer in selected_layers_new]

    # Create the boxplot with matplotlib
    box = plt.boxplot(data, labels=selected_layers_new, patch_artist=True, showfliers=False)

    # Customize box colors
    for patch in box['boxes']:
        patch.set(facecolor='skyblue', alpha=0.5, edgecolor='blue', linewidth=1.5)

    # Add the baseline loss line
    plt.axhline(y=loss_baseline, color='g', linestyle='--', linewidth=1, label='Baseline loss')

    # Compute and plot the median line across boxes
    medians = [np.median(d) for d in data]
    plt.plot(range(1, len(selected_layers) + 1), medians, color='blue', linewidth=2, marker='o', label='Median (50th percentile)')

    # Labels and aesthetics
    plt.xlabel('Layer Type')
    plt.ylabel('Model Loss')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./plot_results/{model_id}_layer_selection.pdf')

def plot_layer_selection_loss_fn(df, model_id, selected_layers, selected_layers_new):
    df = df[df['percentage_of_weights_flipped'] == 1]

    plt.rcParams.update({'font.size':18})
    df_filtered = df[df['number_of_weights_flipped'] != 0]
    df_filtered = df_filtered[df_filtered['Layer_type'].isin(selected_layers)]
    df_sorted_desc = df_filtered.sort_values(by='loss_fn', ascending=False)

    df_subset = df_sorted_desc[df_sorted_desc['Layer_type'].isin(selected_layers)]
    df_mag = df_subset[df_subset['Attack_type'] == 'Hybrid']

    for i in range(len(selected_layers)):
        df['Layer_type'][df['Layer_type'] == selected_layers[i]] = selected_layers_new[i]

    plt.figure(figsize=(8, 6))
    data = [df_mag[df_mag['Layer_type'] == layer]['loss_fn'] for layer in selected_layers_new]

    # Create the boxplot with matplotlib
    box = plt.boxplot(data, labels=selected_layers_new, patch_artist=True, showfliers=False)

    # Customize box colors
    for patch in box['boxes']:
        patch.set(facecolor='skyblue', alpha=0.5, edgecolor='blue', linewidth=1.5)

    # Compute and plot the median line across boxes
    medians = [np.median(d) for d in data]
    plt.plot(range(1, len(selected_layers_new) + 1), medians, color='blue', linewidth=2, marker='o', label='Median (50th percentile)')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    # plt.title('Variation in Loss Function by Layer Type and Attack Type (Logarithmic Scale)')
    plt.xlabel('Layer Type')
    plt.ylabel('Loss Function (Log Scale)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'./plot_results/{model_id}_layer_loss_func.pdf')

def plot_alpha_ablation(subset_sensitivity, model_id, sensitive_layer):
    num_flipped = subset_sensitivity[sensitive_layer]['number_of_weights_flipped'][:-1]
    magnitude = subset_sensitivity[sensitive_layer]['Magnitude'][:-1]
    gradient = subset_sensitivity[sensitive_layer]['Gradient'][:-1]
    alpha_025 = subset_sensitivity[sensitive_layer]['alpha_0.25'][:-1]
    alpha_05 = subset_sensitivity[sensitive_layer]['alpha_0.5'][:-1]
    alpha_075 =     subset_sensitivity[sensitive_layer]['alpha_0.75'][:-1]

    plt.figure(figsize=(9, 6))
    plt.plot(num_flipped, magnitude, marker='o', label='Magnitude')
    plt.plot(num_flipped, gradient, marker='s', label='Gradient')
    plt.plot(num_flipped, alpha_025, marker='^', label='alpha=0.25')
    plt.plot(num_flipped, alpha_05, marker='v', label='alpha=0.5')
    plt.plot(num_flipped, alpha_075, marker='D', label='alpha=0.75')

    # plt.yscale('log')
    plt.xlabel('Number of Weights Flipped')
    plt.ylabel('Model Loss')
    plt.title('Loss vs. Number of Weights Flipped (A_log, Layer 47)')
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.savefig(f'./plot_results/{model_id}_alpha_ablation.pdf')

def plot_optimization(reduced_weight_indices, model_id):

    layer_name = list(reduced_weight_indices.keys())[0]
    progress_data = reduced_weight_indices[layer_name]['progress']
    iterations, num_indices, losses = zip(*progress_data)
    plt.rcParams.update({'font.size':18})

    fig, ax1 = plt.subplots(figsize=(8,6))

    # First y-axis (Number of Indices)
    line1, = ax1.plot(iterations, num_indices, marker='o', label='# Bit-Flips')
    ax1.set_xlabel('# Iterations')
    ax1.set_ylabel('# Bit-Flips')
    ax1.set_xticks(iterations[::3])
    # ax1.set_ylim([0, 10])

    # Second y-axis (Loss)
    ax2 = ax1.twinx()
    line2, = ax2.plot(iterations, losses, marker='x', color='r', label='Loss')
    ax2.set_ylabel('Model Loss')
    ax2.set_ylim([0, 30])

    line3 = ax2.axhline(y=4, color='g', linestyle='--', linewidth=2,  label='Threshold loss')

    # Combine legends from both axes
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='best', frameon=True)

    ax1.grid(True)
    fig.tight_layout()
    plt.savefig(f'./plot_results/{model_id}_exclusionary_optimization.pdf')
