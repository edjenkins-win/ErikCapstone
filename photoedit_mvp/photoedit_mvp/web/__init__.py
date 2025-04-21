"""Web interface for the Photo Editor application."""

from .app import main as run_app

# Function to run the web app
def run_web_app():
    """Run the Streamlit web app."""
    import sys
    from streamlit.web import cli as st_cli
    from pathlib import Path
    
    app_path = Path(__file__).parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    st_cli.main()
