import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

# Paths
base_path = "models/output/BASELINE/"
script_path = "map_generator.py"
input_map_path = "input_maps/defaultMap.xml"
num_maps = 100
episode_length = 64

# Get the current Python interpreter
python_interpreter = sys.executable

# List all .pt files in the directory
pt_files = [f for f in os.listdir(base_path) if f.endswith(".pt")]

def run_map_generator(pt_file):
    """Function to run map_generator.py for a single .pt file."""
    model_path = os.path.join(base_path, pt_file)
    command = [
        python_interpreter,  # Use the current interpreter
        script_path,
        "--input-map-path",
        input_map_path,
        "--model-path",
        model_path,
        "--save-maps",
        "--num-maps", "4",
        "--episode_length", "64",
        "--top-k", "5",
        "--use-baseline"
    ]
    
    # Print the command for debugging
    print(f"Running command: {' '.join(command)}")
    
    # Run the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command for {pt_file}: {e}")

# Run all commands in parallel
with ProcessPoolExecutor() as executor:
    executor.map(run_map_generator, pt_files)
