#!/usr/bin/env python
"""Direct runner for the Photo Editor web application."""

import streamlit.web.cli as st_cli
import sys
from pathlib import Path

def main():
    """Run the Streamlit web app directly."""
    app_path = Path(__file__).parent / "photoedit_mvp" / "web" / "app.py"
    sys.argv = ["streamlit", "run", str(app_path.resolve())]
    print(f"Running Streamlit app at: {app_path.resolve()}")
    st_cli.main()

if __name__ == "__main__":
    main() 