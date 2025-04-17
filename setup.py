from setuptools import setup, find_packages

setup(
    name="photo_ai",
    version="0.1.0",
    packages=find_packages(include=['photo_ai', 'photo_ai.*']),
    install_requires=[
        "numpy",
        "opencv-python",
        "streamlit",
    ],
    python_requires=">=3.8",
) 