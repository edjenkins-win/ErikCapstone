"""Setup script for the Photo Editor application."""

from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Get the version from the package
with open(os.path.join("photoedit_mvp", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split('"')[1]
            break

setup(
    name="photoedit-mvp",
    version=version,
    author="Photo Editor Team",
    author_email="example@example.com",
    description="AI-driven photo editing tool that analyzes and enhances photos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/photoedit-mvp",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
    ],
    extras_require={
        "web": ["streamlit>=1.8.0"],
    },
    entry_points={
        "console_scripts": [
            "photo-analyze=photoedit_mvp.cli.commands:analyze_command",
            "photo-apply=photoedit_mvp.cli.commands:apply_command",
            "photo-edit=photoedit_mvp.cli:run_cli",
            "photo-web=photoedit_mvp.web:run_web_app",
        ],
    },
)
