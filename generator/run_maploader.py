import subprocess
import os



def run_maploader(relative_map_path):
    """
    Run the MapLoader Java program from the maploader/ directory.

    Args:
        relative_map_path (str): The path to the map file relative to the generator/ directory.

    Returns:
        int: The exit code of the Java process.
    """
    # Define paths
    # maploader_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../maploader"))

     # Define the directories
    generator_dir = os.path.dirname(os.path.abspath(__file__))  # Path to generator/
    maploader_dir = os.path.abspath(os.path.join(generator_dir, "../maploader"))  # Path to maploader/

    # Resolve the absolute path to the map file
    absolute_map_path = os.path.abspath(os.path.join(generator_dir, relative_map_path))

    # Get the relative path to the map file from the maploader/ directory perspective
    map_path_for_java = os.path.relpath(absolute_map_path, maploader_dir)
   

    classpath = f"lib/*:src:."

    # Change working directory to maploader
    os.chdir(maploader_dir)

    # Java command to execute MapLoader
    java_command = [
        "java",
        "-cp",
        classpath,
        "MapLoader",
        map_path_for_java
    ]

    try:
        # Run the Java process
        process = subprocess.run(java_command, check=True, capture_output=True, text=True)

        # Print the output from the Java process
        print("Output:")
        print(process.stdout)
        print("Errors:")
        print(process.stderr)
        return process.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running MapLoader: {e}")
        print("Standard Output:", e.stdout)
        print("Standard Error:", e.stderr)
        return e.returncode


if __name__ == "__main__":
    # Path to the map file (relative to the generator/ directory)
    map_file = "../generator/tempMap.xml"  # Update this path as needed

    # Run the MapLoader
    exit_code = run_maploader(map_file)

    # Print the exit code
    print(f"Java process exited with code {exit_code}")
