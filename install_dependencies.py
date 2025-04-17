import platform
import subprocess
import sys
from pathlib import Path

def install_dependencies() -> None:
    """Install dependencies based on platform."""
    print("Installing dependencies...")
    
    # Install base requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Platform-specific installations
    system = platform.system()
    machine = platform.machine()
    
    print(f"\nDetected system: {system} {machine}")
    
    if system == "Darwin" and machine == "arm64":  # Apple Silicon
        print("Installing CuPy for Apple Silicon...")
        try:
            # Try installing CuPy for Apple Silicon
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "cupy"  # Use the base CuPy package for Apple Silicon
            ])
        except subprocess.CalledProcessError:
            print("Failed to install CuPy. GPU acceleration will not be available.")
            print("You can still use the application in CPU mode.")
    elif system == "Linux":
        print("Installing CuPy for Linux...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "cupy-cuda12x"  # Use appropriate CUDA version
            ])
        except subprocess.CalledProcessError:
            print("Failed to install CuPy. GPU acceleration will not be available.")
            print("You can still use the application in CPU mode.")
    else:
        print("Skipping CuPy installation for this platform")


if __name__ == "__main__":
    install_dependencies() 
