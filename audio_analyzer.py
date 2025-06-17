import subprocess
import os
import yaml  # Import PyYAML for parsing YAML files
import matplotlib.pyplot as plt
import numpy as np
import csv  # Import CSV module for exporting data
import argparse

# Feature display limits for plotting
# Define y-limits for specific features to focus on the most relevant regions
FEATURE_YLIMS = {
    "bpm": (90, 130),  
    "spectral_energyband_low.mean": (-25, -10),
    "spectral_energyband_high.mean": (-40, -25),
}

FEATURES_TO_CONVERT_TO_DB = [
    "barkbands.mean",
    "barkbands.median",
    "barkbands.var",
    "melbands.mean",
    "melbands.median",
    "melbands.var",
    "spectral_energyband_low.mean",
    "spectral_energyband_middle_low.mean",
    "spectral_energyband_high.mean",
    #"beats_loudness.mean",
    # Add more features as needed
]

def lin2Db(values):
    """
    Converts a list of values to dB using the formula 10 * log10(value).
    Handles small or zero values by replacing them with a small positive value.
    """
    return [10 * np.log10(max(v, 1e-10)) for v in values]

def call_executable_with_args(executable_name, *args):
    """
    Calls an executable from the 'bin/' folder with the provided arguments.

    :param executable_name: Name of the executable file (e.g., 'my_executable.exe').
    :param args: Arguments to pass to the executable.
    :return: The output and error messages from the executable.
    """
    # Construct the full path to the executable
    executable_path = os.path.join("bin", executable_name)

    # Ensure the executable exists
    if not os.path.isfile(executable_path):
        raise FileNotFoundError(f"Executable '{executable_name}' not found in 'bin/' folder.")

    # Call the executable with arguments
    try:
        result = subprocess.run(
            [executable_path, *args],  # Command and arguments
            stdout=subprocess.PIPE,    # Capture standard output
            stderr=subprocess.PIPE,    # Capture standard error
            text=True                  # Decode output as text
        )
        return result.stdout, result.stderr
    except Exception as e:
        raise RuntimeError(f"Failed to execute '{executable_name}': {e}")

def call_executable_with_args_realtime(executable_name, *args):
    """
    Calls an executable from the 'bin/' folder with the provided arguments
    and streams the output in real-time.

    :param executable_name: Name of the executable file (e.g., 'my_executable.exe').
    :param args: Arguments to pass to the executable.
    """
    executable_path = os.path.join("bin", executable_name)

    if not os.path.isfile(executable_path):
        raise FileNotFoundError(f"Executable '{executable_name}' not found in 'bin/' folder.")

    try:
        process = subprocess.Popen(
            [executable_path, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        for line in process.stdout:
            print(line, end="")

        _, error = process.communicate()
        if error:
            print("Error:", error)
    except Exception as e:
        raise RuntimeError(f"Failed to execute '{executable_name}': {e}")

def parse_yaml_to_dict(yaml_file_path):
    """
    Parses a YAML file and returns its contents as a Python dictionary.

    :param yaml_file_path: Path to the YAML file.
    :return: A dictionary containing the parsed YAML data.
    """
    if not os.path.isfile(yaml_file_path):
        raise FileNotFoundError(f"YAML file '{yaml_file_path}' not found.")

    try:
        with open(yaml_file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise RuntimeError(f"Failed to parse YAML file '{yaml_file_path}': {e}")

def plot_rhythm_beats_position(data):
    """
    Plots the beats positions from the 'rhythm' field.

    :param data: Parsed YAML data as a dictionary.
    """
    beats_positions = data.get("rhythm", {}).get("beats_position", [])
    plt.figure(figsize=(10, 5))
    plt.plot(beats_positions, marker='o', linestyle='-', color='green')
    plt.xlabel("Beat Index")
    plt.ylabel("Position (seconds)")
    plt.title("Rhythm - Beats Positions")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_aggregated_means_in_subplots(aggregated_means, file_names):
    """
    Plots the aggregated 'mean' values for each low-level feature in subplots.

    :param aggregated_means: Dictionary of aggregated means.
    :param file_names: List of file names corresponding to the YAML files.
    """
    num_features = len(aggregated_means)
    num_cols = 2  # Number of columns in the subplot grid
    num_rows = (num_features + num_cols - 1) // num_cols  # Calculate rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 5 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, (feature, means) in enumerate(aggregated_means.items()):
        ax = axes[i]
        ax.bar(file_names, means, color='skyblue')
        ax.set_title(f"'{feature}' Comparison")
        ax.set_xticklabels(file_names, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel("Mean Value")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_lowlevel_means(data):
    """
    Plots all 'mean' values from the 'lowlevel' category in the parsed YAML data.

    :param data: Parsed YAML data as a dictionary.
    """
    if "lowlevel" not in data:
        raise KeyError("The 'lowlevel' field is missing in the YAML data.")

    
    lowlevel_data = data["lowlevel"]
    means = {}

    # Extract 'mean' values from the 'lowlevel' subfields
    for key, value in lowlevel_data.items():
        if isinstance(value, dict) and "mean" in value:
            if type(value["mean"]) is float:
                means[key] = value["mean"]
    
    # Plot the extracted 'mean' values
    plt.figure(figsize=(12, 6))
    plt.bar(means.keys(), means.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylabel("Mean Values")
    plt.title("Lowlevel Features - Mean Values")
    plt.tight_layout()
    plt.show()

def aggregate_lowlevel_means(data_list):
    """
    Aggregates 'mean' values from the 'lowlevel' category across multiple YAML files.

    :param data_list: List of parsed YAML data dictionaries.
    :return: A dictionary where keys are feature names and values are lists of means across files.
    """
    aggregated_means = {}

    for data in data_list:
        if "lowlevel" not in data:
            continue

        lowlevel_data = data["lowlevel"]
        for key, value in lowlevel_data.items():
            if isinstance(value, dict) and "mean" in value:
                if isinstance(value["mean"], (int, float)):
                    if key not in aggregated_means:
                        aggregated_means[key] = []
                    aggregated_means[key].append(value["mean"])

    return aggregated_means

def plot_aggregated_means(aggregated_means, file_names):
    """
    Plots the aggregated 'mean' values for comparison across multiple YAML files.

    :param aggregated_means: Dictionary of aggregated means.
    :param file_names: List of file names corresponding to the YAML files.
    """
    plt.figure(figsize=(14, 8))

    feature_names = list(aggregated_means.keys())
    x = range(len(feature_names))

    for i, file_name in enumerate(file_names):
        means = [aggregated_means[feature][i] if i < len(aggregated_means[feature]) else 0 for feature in feature_names]
        plt.plot(x, means, marker='o', label=file_name)

    plt.xticks(x, feature_names, rotation=45, ha='right', fontsize=10)
    plt.ylabel("Mean Values")
    plt.title("Comparison of Lowlevel Features Across Files")
    plt.legend()
    plt.tight_layout()
    plt.show()

def extract_lowlevel_means(data):
    """
    Extracts 'mean' values from the 'lowlevel' section of the parsed YAML data.

    :param data: Parsed YAML data as a dictionary.
    :return: A dictionary of feature names and their mean values.
    """
    if "lowlevel" not in data:
        raise KeyError("The 'lowlevel' field is missing in the YAML data.")

    lowlevel_data = data["lowlevel"]
    means = {}

    for key, value in lowlevel_data.items():
        if isinstance(value, dict) and "mean" in value:
            if isinstance(value["mean"], (int, float)):
                means[key] = value["mean"]

    return means

def identify_relevant_features(all_means):
    """
    Identifies the most relevant features based on variability across files.

    :param all_means: List of dictionaries containing feature means for each file.
    :return: A list of the most relevant features.
    """
    feature_values = {}
    for means in all_means:
        for feature, value in means.items():
            if feature not in feature_values:
                feature_values[feature] = []
            feature_values[feature].append(value)

    # Calculate variability (e.g., range) for each feature
    variability = {feature: np.ptp(values) for feature, values in feature_values.items()}
    sorted_features = sorted(variability, key=variability.get, reverse=True)

    # Select the top N most relevant features
    top_features = sorted_features[:20]  # Adjust the number of features as needed
    return top_features, feature_values

import os  # Import os to handle file paths

def plot_relevant_features(feature_values, file_names, relevant_features, output_file="relevant_features.png", columns=6):
    """
    Plots the most relevant features for comparison across files in a grid layout and saves the figure.

    :param feature_values: Dictionary of feature values across files.
    :param file_names: List of file names corresponding to the YAML files.
    :param relevant_features: List of the most relevant features.
    :param output_file: Name of the output file to save the figure.
    :param columns: Number of columns in the grid layout (default: 6).
    """
    num_features = len(relevant_features)
    if num_features == 0:
        print(f"No relevant features found to plot for {output_file}.")
        return

    # Calculate the number of rows needed
    rows = (num_features + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(20, 5 * rows))  # Adjust figure size for better visibility
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Generate a colormap for the bars
    colors = plt.cm.tab10(range(len(file_names)))  # Use a colormap with enough distinct colors

    for i, feature in enumerate(relevant_features):
        ax = axes[i]
        values = feature_values[feature]
        ax.bar(file_names, values, color=colors[:len(file_names)])  # Assign a unique color to each file
        ax.set_title(f"'{feature}' Comparison", fontsize=10)
        ax.set_ylabel("Value", fontsize=8)
        ax.set_xticks(range(len(file_names)))
        ax.set_xticklabels(file_names, rotation=45, ha='right', fontsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0)  # Add more space between subplots

    # Ensure the output directory exists
    os.makedirs("out", exist_ok=True)

    # Save the figure to the out directory
    output_path = os.path.join("out", output_file)
    plt.savefig(output_path, dpi=300)  # Save the figure as a PNG file with high resolution
    print(f"Figure saved as '{output_path}'")
    plt.close(fig)  # Close the figure to free memory

def extract_rhythm_features(data):
    """
    Extracts scalar features from the 'rhythm' section of the parsed YAML data.

    :param data: Parsed YAML data as a dictionary.
    :return: A dictionary of rhythm feature names and their values.
    """
    if "rhythm" not in data:
        raise KeyError("The 'rhythm' field is missing in the YAML data.")

    rhythm_data = data["rhythm"]
    features = {}

    for key, value in rhythm_data.items():
        # Only include scalar values (e.g., int, float)
        if isinstance(value, (int, float)):
            features[key] = value

    return features

def identify_relevant_rhythm_features(all_features):
    """
    Identifies the most relevant rhythm features based on variability across files.

    :param all_features: List of dictionaries containing rhythm features for each file.
    :return: A list of the most relevant rhythm features.
    """
    feature_values = {}
    for features in all_features:
        for feature, value in features.items():
            if feature not in feature_values:
                feature_values[feature] = []
            feature_values[feature].append(value)

    # Calculate variability (e.g., range) for each feature
    variability = {feature: np.ptp(values) for feature, values in feature_values.items()}
    sorted_features = sorted(variability, key=variability.get, reverse=True)

    # Select the top N most relevant features
    top_features = sorted_features[:10]  # Adjust the number of features as needed
    return top_features, feature_values

def plot_rhythm_features(feature_values, file_names, relevant_features, output_file="rhythm_features_comparison.png"):
    """
    Plots the most relevant rhythm features for comparison across files and saves the figure.

    :param feature_values: Dictionary of feature values across files.
    :param file_names: List of file names corresponding to the YAML files.
    :param relevant_features: List of the most relevant rhythm features.
    :param output_file: Name of the output file to save the figure.
    """
    num_features = len(relevant_features)
    fig, axes = plt.subplots(num_features, 1, figsize=(12, 5 * num_features))  # Adjust figure size for better visibility

    if num_features == 1:
        axes = [axes]  # Ensure axes is iterable for a single feature

    # Generate a colormap for the bars
    colors = plt.cm.tab10(range(len(file_names)))  # Use a colormap with enough distinct colors

    for i, feature in enumerate(relevant_features):
        ax = axes[i]
        values = feature_values[feature]
        ax.bar(file_names, values, color=colors[:len(file_names)])  # Assign a unique color to each file
        ax.set_title(f"Comparison of '{feature}' Across Files", fontsize=14)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_xticks(range(len(file_names)))
        ax.set_xticklabels(file_names, rotation=45, ha='right', fontsize=10)

    plt.tight_layout(pad=3.0)  # Add more space between subplots
    
    # Ensure the output directory exists
    os.makedirs("out", exist_ok=True)

    # Save the figure to the out directory
    output_path = os.path.join("out", output_file)
    plt.savefig(output_path, dpi=300)  # Save the figure as a PNG file with high resolution
    print(f"Figure saved as '{output_path}'")
    plt.close(fig)  # Close the figure to free memory

def extract_tonal_features(data):
    """
    Extracts scalar features from the 'tonal' section of the parsed YAML data.

    :param data: Parsed YAML data as a dictionary.
    :return: A dictionary of tonal feature names and their values.
    """
    if "tonal" not in data:
        raise KeyError("The 'tonal' field is missing in the YAML data.")

    tonal_data = data["tonal"]
    features = {}

    for key, value in tonal_data.items():
        # Only include scalar values (e.g., int, float)
        if isinstance(value, (int, float)):
            features[key] = value

    return features

def identify_relevant_tonal_features(all_features):
    """
    Identifies the most relevant tonal features based on variability across files.

    :param all_features: List of dictionaries containing tonal features for each file.
    :return: A list of the most relevant tonal features.
    """
    feature_values = {}
    for features in all_features:
        for feature, value in features.items():
            if feature not in feature_values:
                feature_values[feature] = []
            feature_values[feature].append(value)

    # Calculate variability (e.g., range) for each feature
    variability = {feature: np.ptp(values) for feature, values in feature_values.items()}
    sorted_features = sorted(variability, key=variability.get, reverse=True)

    # Select the top N most relevant features
    top_features = sorted_features[:5]  # Adjust the number of features as needed
    return top_features, feature_values

def plot_tonal_features(feature_values, file_names, relevant_features, output_file="tonal_features_comparison.png"):
    """
    Plots the most relevant tonal features for comparison across files and saves the figure.

    :param feature_values: Dictionary of feature values across files.
    :param file_names: List of file names corresponding to the YAML files.
    :param relevant_features: List of the most relevant tonal features.
    :param output_file: Name of the output file to save the figure.
    """
    num_features = len(relevant_features)
    fig, axes = plt.subplots(num_features, 1, figsize=(12, 5 * num_features))  # Adjust figure size for better visibility

    if num_features == 1:
        axes = [axes]  # Ensure axes is iterable for a single feature

    # Generate a colormap for the bars
    colors = plt.cm.tab10(range(len(file_names)))  # Use a colormap with enough distinct colors

    for i, feature in enumerate(relevant_features):
        ax = axes[i]
        values = feature_values[feature]
        ax.bar(file_names, values, color=colors[:len(file_names)])  # Assign a unique color to each file
        ax.set_title(f"Comparison of '{feature}' Across Files", fontsize=14)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_xticks(range(len(file_names)))
        ax.set_xticklabels(file_names, rotation=45, ha='right', fontsize=10)

    plt.tight_layout(pad=3.0)  # Add more space between subplots
    
    # Ensure the output directory exists
    os.makedirs("out", exist_ok=True)

    # Save the figure to the out directory
    output_path = os.path.join("out", output_file)
    plt.savefig(output_path, dpi=300)  # Save the figure as a PNG file with high resolution
    print(f"Figure saved as '{output_path}'")
    plt.close(fig)  # Close the figure to free memory

import numpy as np  # Import numpy for logarithmic calculations

def extract_sequences(data, section, include_mean=True):
    """
    Extracts sequences (e.g., arrays or lists) from a specific section of the parsed YAML data,
    including both 'mean' sequences and plain sequences.

    :param data: Parsed YAML data as a dictionary.
    :param section: The section to extract features from (e.g., 'lowlevel', 'rhythm', 'tonal').
    :param include_mean: Whether to include 'mean' sequences (default: True).
    :return: A dictionary of sequence feature names and their values.
    """
    if section not in data:
        raise KeyError(f"The '{section}' field is missing in the YAML data.")

    section_data = data[section]
    sequences = {}

    def find_sequences(prefix, obj):
        """
        Recursively searches for sequence features in the given object.

        :param prefix: The current key prefix (e.g., 'tonal.hpcp').
        :param obj: The current object to search.
        """
        if isinstance(obj, list):
            sequences[prefix] = obj  # Add plain sequences to the results
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if include_mean and key == "mean" and isinstance(value, list):
                    sequences[f"{prefix}.{key}"] = value  # Add 'mean' sequences to the results
                elif isinstance(value, list):
                    sequences[f"{prefix}.{key}"] = value  # Add plain sequences to the results
                elif isinstance(value, dict):
                    find_sequences(f"{prefix}.{key}", value)  # Recurse into nested dictionaries

    # Start the recursive search
    for key, value in section_data.items():
        find_sequences(key, value)

    return sequences

def plot_sequence_features(sequence_values, file_names, output_file="sequence_features_comparison.png", columns=6):
    """
    Plots sequence features for comparison across files in a grid layout and saves the figure.

    :param sequence_values: Dictionary of sequence feature values across files.
    :param file_names: List of file names corresponding to the YAML files.
    :param output_file: Name of the output file to save the figure.
    :param columns: Number of columns in the grid layout (default: 6).
    """
    # Filter out multidimensional sequences
    filtered_sequence_values = {
        feature: sequences
        for feature, sequences in sequence_values.items()
        if all(isinstance(seq, list) and all(isinstance(val, (int, float)) for val in seq) for seq in sequences)
    }

    num_features = len(filtered_sequence_values)
    if num_features == 0:
        print(f"No valid one-dimensional sequences found to plot for {output_file}.")
        return

    # Calculate the number of rows needed
    rows = (num_features + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(20, 5 * rows))  # Adjust figure size for better visibility
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, (feature, sequences) in enumerate(filtered_sequence_values.items()):
        ax = axes[i]
        for file_name, sequence in zip(file_names, sequences):
            # Convert to dB if the feature is in the list
            if feature in FEATURES_TO_CONVERT_TO_DB:
                sequence = lin2Db(sequence)

            ax.plot(sequence, label=file_name)  # Plot each file's sequence
        ax.set_title(f"{feature}", fontsize=10)
        ax.set_ylabel("Value (dB)" if feature in FEATURES_TO_CONVERT_TO_DB else "Value", fontsize=8)
        ax.set_xlabel("Index", fontsize=8)
        ax.legend(fontsize=6)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0)  # Add more space between subplots

    # Ensure the output directory exists
    os.makedirs("out", exist_ok=True)

    # Save the figure to the out directory
    output_path = os.path.join("out", output_file)
    plt.savefig(output_path, dpi=300)  # Save the figure as a PNG file with high resolution
    print(f"Figure saved as '{output_path}'")
    plt.close(fig)  # Close the figure to free memory

def plot_selected_features(data_list, file_names, selected_features, section=None, output_file="selected_features_comparison.png", columns=4):
    """
    Plots manually selected features for comparison across files and saves the figure.

    :param data_list: List of parsed YAML data dictionaries.
    :param file_names: List of file names corresponding to the YAML files.
    :param selected_features: List of manually selected feature names to plot.
    :param section: The section to extract features from (e.g., 'lowlevel', 'rhythm', 'tonal'). If None, search across all sections.
    :param output_file: Name of the output file to save the figure.
    :param columns: Number of columns in the grid layout (default: 4).
    """
    feature_values = {feature: [] for feature in selected_features}

    # Extract values for the selected features
    for data in data_list:
        if section:
            if section not in data:
                raise KeyError(f"The '{section}' field is missing in the YAML data.")
            sections_to_search = [section]
        else:
            sections_to_search = ["lowlevel", "rhythm", "tonal"]  # Search across all sections

        for feature in selected_features:
            value = None
            for sec in sections_to_search:
                if sec in data:
                    section_data = data[sec]
                    keys = feature.split(".")  # Handle nested keys
                    value = section_data
                    for key in keys:
                        if isinstance(value, dict) and key in value:
                            value = value[key]
                        else:
                            value = None
                            break
                    if value is not None:
                        break  # Stop searching if the feature is found
            feature_values[feature].append(value)

    # Plot the selected features
    num_features = len(selected_features)
    if num_features == 0:
        print(f"No features to plot for {output_file}.")
        return

    # Calculate the number of rows needed
    rows = (num_features + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(20, 5 * rows))  # Adjust figure size for better visibility
    
    # Ensure axes is always a flat list
    if num_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten the axes array for easy iteration
    
    # Generate a colormap for the bars
    colors = plt.cm.tab10(range(len(file_names)))  # Use a colormap with enough distinct colors

    for i, feature in enumerate(selected_features):
        ax = axes[i]
        values = feature_values[feature]

        # Filter out None values
        valid_values = [v if v is not None else 0 for v in values]

        # Check if the feature contains sequences
        if all(isinstance(v, list) for v in valid_values):
            # Plot sequences
            for file_name, sequence in zip(file_names, valid_values):
                # Convert to dB if the feature is in the list
                if feature in FEATURES_TO_CONVERT_TO_DB:
                    sequence = lin2Db(sequence)
                ax.plot(sequence, label=file_name)
            ax.set_ylabel("Value (dB)" if feature in FEATURES_TO_CONVERT_TO_DB else "Value", fontsize=12)
            ax.set_xlabel("Index", fontsize=12)
            ax.legend(fontsize=8)
        else:
            # Convert to dB if the feature is in the list
            if feature in FEATURES_TO_CONVERT_TO_DB:
                valid_values = lin2Db(valid_values)
            # Plot scalar values
            ax.bar(file_names, valid_values, color=colors[:len(file_names)])
            ax.set_ylabel("Value (dB)" if feature in FEATURES_TO_CONVERT_TO_DB else "Value", fontsize=8)
            ax.set_xticks(range(len(file_names)))
            ax.set_xticklabels(file_names, rotation=45, ha='right', fontsize=10)

        # Set y-limits for specific features if defined
        if feature in FEATURE_YLIMS:
            ax.set_ylim(FEATURE_YLIMS[feature])

        ax.set_title(f"'{feature}'", fontsize=14)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0)  # Add more space between subplots

    # Ensure the output directory exists
    os.makedirs("out", exist_ok=True)

    # Save the figure to the out directory
    output_path = os.path.join("out", output_file)
    plt.savefig(output_path, dpi=300)  # Save the figure as a PNG file with high resolution
    print(f"Figure saved as '{output_path}'")
    plt.close(fig)  # Close the figure to free memory

def plot_mfcc_means(parsed_data_list, file_names, output_file="mfcc_means_lineplot.png"):
    """
    Plots the z-score normalized MFCC mean vectors for each file as lines on the same plot, skipping the first two coefficients.
    The z-score output is divided by 3 to fit most values in the range [-1, 1].
    """
    plt.figure(figsize=(10, 6))
    for data, name in zip(parsed_data_list, file_names):
        mfcc_mean = None
        if "lowlevel" in data and "mfcc" in data["lowlevel"] and "mean" in data["lowlevel"]["mfcc"]:
            mfcc_mean = data["lowlevel"]["mfcc"]["mean"]
        if mfcc_mean is not None and len(mfcc_mean) > 2:
            mfcc_vals = np.array(mfcc_mean[2:])  # Skip mfcc[0] and mfcc[1]
            mean = np.mean(mfcc_vals)
            std = np.std(mfcc_vals)
            if std > 1e-10:
                normed = (mfcc_vals - mean) / std / 3  # Divide by 3 to scale to ~[-1, 1]
            else:
                normed = np.zeros_like(mfcc_vals)
            plt.plot(range(2, len(mfcc_mean)), normed, marker='o', label=name)
    plt.xlabel("MFCC Coefficient Index (starting from 2)")
    plt.ylabel("Normalized Value (z-score / 3)")
    plt.title("Z-score Normalized MFCC Mean Vectors (excluding MFCC[0] and MFCC[1]), scaled to ~[-1, 1]")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    os.makedirs("out", exist_ok=True)
    plt.savefig(os.path.join("out", output_file), dpi=300)
    plt.close()
    print(f"Figure saved as 'out/{output_file}'")


def plot_11_point_polygon(values, title="11-Point Polygon", output_file="11_point_polygon.png"):
    """
    Plots a polygon formed by 11 points (e.g., for visualizing MFCCs 2-12).
    Maps values from [-1, 1] to [0, 1] for the radius so that 0 is at the middle of the circle.
    :param values: List or array of 11 values.
    :param title: Title of the plot.
    :param output_file: Output filename.
    """
    if len(values) != 11:
        raise ValueError("Input must be a list or array of 11 values.")

    # Map values from [-1, 1] to [0, 1] for the radius
    radii = 0.5 * (np.array(values) + 1)
    radii = np.append(radii, radii[0])  # Close the polygon

    # Calculate angles for 11 points around a circle
    angles = np.linspace(0, 2 * np.pi, 12)  # 12 to close the polygon

    # Plot
    fig, ax = plt.subplots(subplot_kw={'polar': True})
    ax.plot(angles, radii, marker='o')
    ax.fill(angles, radii, alpha=0.25)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 11, endpoint=False))
    ax.set_xticklabels([str(i+2) for i in range(11)])  # Label as MFCC 2-12
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_file), dpi=300)
    plt.close()
    print(f"Figure saved as 'out/{output_file}'")

def plot_mfcc_polygon_for_each_file(parsed_data_list, file_names, output_dir="out"):
    """
    For each file, plot an 11-point polygon using the z-score normalized MFCC mean coefficients 2-12,
    scaled to ~[-1, 1] by dividing by 3.
    """
    os.makedirs(output_dir, exist_ok=True)
    for data, name in zip(parsed_data_list, file_names):
        mfcc_mean = None
        if "lowlevel" in data and "mfcc" in data["lowlevel"] and "mean" in data["lowlevel"]["mfcc"]:
            mfcc_mean = data["lowlevel"]["mfcc"]["mean"]

        if mfcc_mean is not None and len(mfcc_mean) > 12:
            mfcc_vals = np.array(mfcc_mean[2:13])  # Coefficients 2-12 (11 values)
            mean = np.mean(mfcc_vals)
            std = np.std(mfcc_vals)
            if std > 1e-10:
                normed = (mfcc_vals - mean) / std / 3  # Z-score, then scale to ~[-1, 1]
            else:
                normed = np.zeros_like(mfcc_vals)

            plot_11_point_polygon(
                normed,
                title=f"MFCC Polygon: {name}",
                output_file=f"{output_dir}/mfcc_polygon_{os.path.splitext(name)[0]}.png"
            )

def export_features_to_csv(data_list, file_names, selected_features, output_file="selected_features_global.csv"):
    """
    Exports the values of selected features to a CSV file, including descriptions,
    11 normalized MFCC coefficients (2-12), and their mapped values for circle plotting.
    """
    # Feature descriptions
    feature_descriptions = {
        "dynamic_complexity": "Measures the variation in loudness over time, reflecting dynamic range.",
        "spectral_complexity.mean": "Indicates the number of spectral peaks, reflecting timbral richness.",
        "hfc.mean": "High-frequency content, associated with brightness or sharpness of the sound.",
        "barkbands_kurtosis.mean": "Kurtosis of Bark bands, reflecting tonal sharpness or flatness.",
        "erbbands_kurtosis.mean": "Kurtosis of ERB bands, indicating spectral shape characteristics.",
        "beats_loudness.mean": "Average loudness of beats, reflecting rhythmic intensity.",
        "danceability": "Measures how suitable the track is for dancing, based on rhythm and tempo.",
        "tuning_equal_tempered_deviation": "Deviation from equal-tempered tuning, reflecting tuning accuracy.",
        "bpm": "Beats per minute, indicating the tempo of the track.",
        "spectral_energyband_low.mean": "Energy in the low-frequency band, reflecting bass presence.",
        "spectral_energyband_high.mean": "Energy in the high-frequency band, reflecting treble presence.",
    }
    # Add descriptions for normalized MFCCs 2-12
    mfcc_normed_labels = [f"mfcc_normed_{i}" for i in range(2, 13)]
    for i in range(2, 13):
        feature_descriptions[f"mfcc_normed_{i}"] = f"Z-score normalized and scaled MFCC coefficient {i} (mean/std/3) from the mean vector, reflects spectral envelope shape."
    # Add descriptions for mapped MFCCs 2-12 (for circle plotting)
    mfcc_circle_labels = [f"mfcc_circle_{i}" for i in range(2, 13)]
    for i in range(2, 13):
        feature_descriptions[f"mfcc_circle_{i}"] = f"Mapped (0.5*(x+1)) z-score normalized MFCC coefficient {i} for circle plotting."

     # Prepare a dictionary to store feature values
    feature_values = {feature: [] for feature in selected_features}

    # Extract values for the selected features and normalized MFCCs
    rows = []
    for data, file_name in zip(data_list, file_names):
        row = {"File Name": file_name}
        # Extract selected features
        for feature in selected_features:
            value = None
            for section in ["lowlevel", "rhythm", "tonal"]:
                if section in data:
                    section_data = data[section]
                    keys = feature.split(".")
                    value = section_data
                    for key in keys:
                        if isinstance(value, dict) and key in value:
                            value = value[key]
                        else:
                            value = None
                            break
                    if value is not None:
                        break  # Stop searching if the feature is found
            # Convert to dB if needed
            if feature in FEATURES_TO_CONVERT_TO_DB and value is not None:
                if isinstance(value, list):
                    value = lin2Db(value)
                else:
                    value = lin2Db([value])[0]
            row[feature] = value

        # Extract and normalize MFCC[2:13] for circle plotting
        mfcc_circle = [None] * 11
        if "lowlevel" in data and "mfcc" in data["lowlevel"] and "mean" in data["lowlevel"]["mfcc"]:
            mfcc_mean = data["lowlevel"]["mfcc"]["mean"]
            if mfcc_mean is not None and len(mfcc_mean) > 12:
                mfcc_vals = np.array(mfcc_mean[2:13])  # Coefficients 2-12 (11 values)
                mean = np.mean(mfcc_vals)
                std = np.std(mfcc_vals)
                if std > 1e-10:
                    mfcc_normed = ((mfcc_vals - mean) / std / 3)
                    mfcc_circle = (0.5 * (mfcc_normed + 1)).tolist()
                else:
                    mfcc_circle = [0.5] * 11
        for i, val in enumerate(mfcc_circle):
            row[f"mfcc_circle_{i+2}"] = val

        rows.append(row)

    # Write to CSV (transposed format with descriptions)
    output_path = os.path.join("out", output_file)
    os.makedirs("out", exist_ok=True)  # Ensure the output directory exists
    fieldnames = ["File Name"] + selected_features + [f"mfcc_circle_{i}" for i in range(2, 13)]
    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Write header row: feature name, description, then one column per file
        writer.writerow(["Feature Name", "Description"] + file_names)
        # Write each feature and its values
        for feature in selected_features + [f"mfcc_circle_{i}" for i in range(2, 13)]:
            description = feature_descriptions.get(feature, "No description available.")
            values = [row.get(feature, "") for row in rows]
            writer.writerow([feature, description] + values)

    print(f"CSV file with descriptions and mapped circle values saved as '{output_path}'")

# Example usage
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Audio feature analysis")
        parser.add_argument("--skip-extraction", action="store_true", help="Skip feature extraction step")
        args = parser.parse_args()

        samples = os.listdir("samples/")  # List files in the 'samples/' directory
        parsed_data_list = []
        file_names = []

        for sample in samples:
            if sample.endswith(".wav"):  # Process only .wav files
                bin = "streaming_extractor_music.exe"
                arg1 = os.path.join("samples", sample)
                arg2 = os.path.join("out", f"data_{sample.split('.wav')[0]}_raw.yml")
                arg3 = "profile.yml"

                if not args.skip_extraction:
                    # Call the executable
                    call_executable_with_args_realtime(bin, arg1, arg2, arg3)

                # Parse the output YAML file
                yaml_file_path = arg2
                parsed_data = parse_yaml_to_dict(yaml_file_path)
                parsed_data_list.append(parsed_data)
                file_names.append(sample)

        # Identify the most relevant features
        all_means = [extract_lowlevel_means(data) for data in parsed_data_list]
        relevant_features, feature_values = identify_relevant_features(all_means)

        # Plot the relevant features
        plot_relevant_features(feature_values, file_names, relevant_features, output_file="relevant_features_comparison.png")

        # Identify the most relevant rhythm features
        all_rhythm_features = [extract_rhythm_features(data) for data in parsed_data_list]
        relevant_rhythm_features, rhythm_feature_values = identify_relevant_rhythm_features(all_rhythm_features)

        # Plot the relevant rhythm features
        plot_rhythm_features(rhythm_feature_values, file_names, relevant_rhythm_features, output_file="rhythm_features_comparison.png")

        # Identify the most relevant tonal features
        all_tonal_features = [extract_tonal_features(data) for data in parsed_data_list]
        relevant_tonal_features, tonal_feature_values = identify_relevant_tonal_features(all_tonal_features)

        # Plot the relevant tonal features
        plot_tonal_features(tonal_feature_values, file_names, relevant_tonal_features, output_file="tonal_features_comparison.png")

        # Process 'mean' sequences for each section
        for section, output_file, convert_to_db in [
            ("lowlevel", "lowlevel_mean_sequences_comparison.png", False),  # Convert lowlevel to dB
            ("rhythm", "rhythm_mean_sequences_comparison.png", False),
            ("tonal", "tonal_mean_sequences_comparison.png", False),
        ]:
            all_sequences = {}
            for data in parsed_data_list:
                sequences = extract_sequences(data, section=section, include_mean=True)
                for key, value in sequences.items():
                    if key not in all_sequences:
                        all_sequences[key] = []
                    all_sequences[key].append(value)
    
            # Plot the 'mean' sequences for the current section
            if all_sequences:
                plot_sequence_features(all_sequences, file_names, output_file=output_file)

        # Manually select features to plot
        selected_features_lowlevel = [
            "average_loudness",
            "dissonance.mean",
            "dynamic_complexity",
            "hfc.mean",
            "pitch_salience.mean",
            "spectral_spread.mean",
            "spectral_complexity.mean",
            "spectral_rolloff.mean",
            "barkbands_kurtosis.mean",
            "barkbands.mean",
            "erbbands.mean",
            "erbbands_kurtosis.mean",
            "mfcc.mean",
            "spectral_contrast_valleys.mean",
            "silence_rate_30dB.mean",
            "zerocrossingrate.mean",
        ]

        selected_features_rythm = [
            "beats_count",
            "bpm",
            "beats_loudness.mean",
            "beats_loudness_band_ratio.mean",
            "bpm_histogram_first_peak_spread.mean",
            "bpm_histogram_first_peak_weight.mean",
            "danceability",
            "onset_rate",
        ]

        selected_features_tonal = [
            "tuning_equal_tempered_deviation",
            "tuning_nontempered_energy_ratio", 
            "chords_changes_rate",
            "hpcp.mean",
            "chords_histogram",
            "key_key",
            "key_strength",
            "key_scale"
        ]

        selected_features_global = [
            "dynamic_complexity",
            "spectral_complexity.mean",
            "hfc.mean",
            "barkbands_kurtosis.mean",
            "erbbands_kurtosis.mean",
            "beats_loudness.mean",
            "danceability",
            "tuning_equal_tempered_deviation",
            "bpm",
            "spectral_energyband_low.mean",
            "spectral_energyband_high.mean",
        ]


        # Plot the selected features
        plot_selected_features(parsed_data_list, file_names, selected_features_lowlevel, section="lowlevel", output_file="manually_selected_lowlevel_features_comparison.png")
        plot_selected_features(parsed_data_list, file_names, selected_features_rythm, section="rhythm", output_file="manually_selected_rhythm_features_comparison.png")
        plot_selected_features(parsed_data_list, file_names, selected_features_tonal, section="tonal", output_file="manually_selected_tonal_features_comparison.png")
        
        # Plot the selected global features
        plot_selected_features(
            parsed_data_list, 
            file_names, 
            selected_features_global, 
            section=None,  # Pass None since features span multiple sections
            output_file="manually_selected_global_features_comparison.png",
            columns=3
        )

        # Plot the z-score normalized MFCC mean vectors for each file
        plot_mfcc_means(parsed_data_list, file_names, output_file="mfcc_means_lineplot.png")

        # Plot the 11-point polygons for each file using the z-score normalized MFCC mean coefficients
        plot_mfcc_polygon_for_each_file(parsed_data_list, file_names)

        # Export the selected global features to CSV
        export_features_to_csv(parsed_data_list, file_names, selected_features_global, output_file="selected_features_global.csv")

    except Exception as e:
        print(f"Error: {e}")