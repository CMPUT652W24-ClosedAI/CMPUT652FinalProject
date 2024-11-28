import atexit
import signal
import subprocess
import os

class MapLoaderController:
    def __init__(self, classpath="lib/*:src:."):
        """
        Initialize the MapLoaderController.

        Args:
            classpath (str): The Java classpath, including dependencies and compiled classes.
        """
        self.classpath = classpath
        self.process = None
          # Register cleanup function to ensure the process is terminated
        atexit.register(self.stop_maploader)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """
        Signal handler to stop the MapLoader process on termination signals.
        """
        print(f"Received signal {signum}. Cleaning up MapLoader...")
        self.stop_maploader()

    def start_maploader(self, map_file):
        """
        Start or update the MapLoader process with a new map file.

        Args:
            map_file (str): The path to the map file to load.
        """
        generator_dir = os.path.dirname(os.path.abspath(__file__))  # Path to generator/
        maploader_dir = os.path.abspath(os.path.join(generator_dir, "../maploader"))  # Path to maploader/

        # Resolve the absolute path to the map file
        absolute_map_path = os.path.abspath(os.path.join(generator_dir, map_file))

        # Get the relative path to the map file from the maploader/ directory perspective
        map_path_for_java = os.path.relpath(absolute_map_path, maploader_dir)

        if self.process and self.process.poll() is None:
            # Update existing process
            # print("Updating MapLoader with new map file:", map_file)
            try:
                self.process.stdin.write(map_path_for_java + "\n")
                self.process.stdin.flush()
            except Exception as e:
                print(f"Error updating MapLoader: {e}")
        else:
            # Start a new process
            java_command = [
                "java",
                "-cp",
                self.classpath,
                "MapLoader"
            ]
            try:
                self.process = subprocess.Popen(
                    java_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=maploader_dir
                )
                # print("MapLoader started with map file:", map_file)
                self.process.stdin.write(map_path_for_java + "\n")
                self.process.stdin.flush()
            except Exception as e:
                print(f"Error starting MapLoader: {e}")

    def stop_maploader(self):
        """
        Stop the MapLoader process if it is running.
        """
        if self.process and self.process.poll() is None:
            print("Stopping MapLoader...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                print("MapLoader stopped.")
            except subprocess.TimeoutExpired:
                print("MapLoader did not stop gracefully. Killing process.")
                self.process.kill()
        else:
            print("MapLoader is not running.")

    def get_output(self):
        """
        Get the output from the MapLoader process.

        Returns:
            tuple: (stdout, stderr) output from the process.
        """
        if self.process:
            try:
                stdout, stderr = self.process.communicate(timeout=1)
                return stdout, stderr
            except subprocess.TimeoutExpired:
                return None, None
        return None, None
