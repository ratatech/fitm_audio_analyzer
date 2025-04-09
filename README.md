# FITM Audio Feature Analyzer

This project is a Python-based tool for analyzing audio files and extracting detailed audio features. It uses an external executable to process audio files and generates a YAML file containing the extracted features. The tool also includes functionality to parse the YAML file and visualize specific features using Python.

## Features
- **Audio Feature Extraction**: Extracts low-level, rhythm, tonal, and other audio features from audio files.
- **Real-Time Output Streaming**: Streams the output of the external executable in real-time.
- **YAML Parsing**: Parses the generated YAML file and converts it into a Python dictionary for further processing.
- **Data Visualization**: Plots specific features (e.g., `lowlevel.mean` values) using `matplotlib`.

## Requirements
- Python 3.7 or later
- Dependencies:
  - `pyyaml`: For parsing YAML files.
  - `matplotlib`: For plotting data.

Install the dependencies using:
```bash
pip install pyyaml matplotlib
```

## Usage

### 1. Prepare the Environment
- Place the audio files you want to analyze in the `samples/` directory.
- Ensure the external executable (e.g., `streaming_extractor_music.exe`) is located in the `bin/` directory.

### 2. Run the Script
Use the following command to analyze an audio file and extract features:
```bash
python audio_analyzer.py
```

### 3. Example Workflow
1. **Analyze an Audio File**:
   - The script will call the external executable to process the audio file (e.g., `samples/fetus.wav`) and generate a YAML file in the `out/` directory (e.g., `out/data_raw.yml`).

2. **Parse the YAML File**:
   - The script will parse the generated YAML file and load its contents into a Python dictionary.

3. **Visualize Features**:
   - The script will extract the `lowlevel.mean` values from the YAML file and plot them using `matplotlib`.

### 4. Customize the Script
- Modify the `bin`, `samples`, and `out` paths in the script to match your directory structure.
- Update the arguments passed to the executable as needed.

## Example Output
- **Real-Time Output**: The script streams the output of the external executable in real-time.
- **Parsed Data**: The parsed YAML data is printed to the console.
- **Visualization**: A bar chart of `lowlevel.mean` values is displayed.

## Directory Structure
```
fitm_audio_analyzer/
├── bin/
│   └── streaming_extractor_music.exe  # External executable
├── samples/
│   └── fetus.wav                      # Example audio file
├── out/
│   └── data_raw.yml                   # Extracted features (YAML file)
├── audio_analyzer.py                  # Main script
└── README.md                          # Project documentation
```

## Notes
- Ensure the external executable is compatible with your operating system.
- If you encounter permission issues with GitHub, ensure the correct GitHub account is configured for your repository.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.