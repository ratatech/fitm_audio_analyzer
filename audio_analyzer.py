import subprocess
import os
import yaml  # Import PyYAML for parsing YAML files
import matplotlib.pyplot as plt

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
    # Construct the full path to the executable
    executable_path = os.path.join("bin", executable_name)

    # Ensure the executable exists
    if not os.path.isfile(executable_path):
        raise FileNotFoundError(f"Executable '{executable_name}' not found in 'bin/' folder.")

    # Call the executable with arguments
    try:
        process = subprocess.Popen(
            [executable_path, *args],  # Command and arguments
            stdout=subprocess.PIPE,    # Capture standard output
            stderr=subprocess.PIPE,    # Capture standard error
            text=True                  # Decode output as text
        )

        # Stream the output in real-time
        for line in process.stdout:
            print(line, end="")  # Print each line as it is received

        # Wait for the process to complete and capture any errors
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
            data = yaml.safe_load(yaml_file)  # Parse YAML file into a dictionary
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse YAML file '{yaml_file_path}': {e}")

def plot_lowlevel_means(data):
    """
    Plots the 'mean' values from the 'lowlevel' field in the parsed YAML data.

    :param data: Parsed YAML data as a dictionary.
    """
    if "lowlevel" not in data:
        raise KeyError("The 'lowlevel' field is missing in the YAML data.")

    lowlevel_data = data["lowlevel"]
    means = {}
    
    # Extract 'mean' values from the 'lowlevel' subfields
    for key, value in lowlevel_data.items():
        if isinstance(value, dict) and "mean" in value:
            means[key] = value["mean"]

    # Plot the extracted 'mean' values
    plt.figure(figsize=(10, 6))
    plt.bar(means.keys(), means.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylabel("Mean Values")
    plt.title("Lowlevel Features - Mean Values")
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    try:
        samples = os.listdir("samples/")  # List files in the 'bin/' directory
        
        for sample in samples:
            bin = "streaming_extractor_music.exe"
            arg1 = "samples/" + sample
            arg2 = "out/data_" + sample.split(".wav")[0] + "_raw.yml"  # Ensure the output file has the correct path and extension
            arg3 = "profile.yml"

            # Call the executable
            call_executable_with_args_realtime(bin, arg1, arg2, arg3)

            # Parse the output YAML file
            yaml_file_path = arg2
            parsed_data = parse_yaml_to_dict(yaml_file_path)

            # Plot the 'mean' values from the 'lowlevel' field
            #plot_lowlevel_means(parsed_data)

    except Exception as e:
        print(f"Error: {e}")