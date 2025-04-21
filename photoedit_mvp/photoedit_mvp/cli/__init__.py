"""Command-line interface for the Photo Editor application."""

from .commands import main

# Entry point for the CLI
def run_cli():
    """Run the CLI application."""
    import sys
    sys.exit(main())
