import os
import sys
import platform
import torch
import subprocess
import asyncio

def main():
    """Main entry point for the Streamlit application."""
    # Set the working directory to the project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Set environment variables for macOS
    if platform.system() == "Darwin":
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["TORCH_DISABLE_CUSTOM_CLASSES"] = "1"
        os.environ["STREAMLIT_SERVER_PORT"] = "8501"
        os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "500"
        os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
        os.environ["STREAMLIT_SERVER_ENABLE_XSRF"] = "false"
    
    # Initialize torch with single thread
    torch.set_num_threads(1)
    
    # Create a new event loop for macOS
    if platform.system() == "Darwin":
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run Streamlit using subprocess with specific environment
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "photo_ai/frontend/app.py",
        "--server.port=8501",
        "--server.address=localhost",
        "--server.headless=true",
        "--server.maxUploadSize=500",
        "--server.enableXsrfProtection=false",
        "--server.enableCORS=false",
        "--global.developmentMode=false"
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 