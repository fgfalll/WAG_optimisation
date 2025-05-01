from setuptools import setup, find_packages

setup(
    name="co2eor-optimizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scipy>=1.7",
        "pycuda>=2021.1",
        "optuna>=2.10",
        "lasio>=0.30",
        "matplotlib>=3.5"
    ],
    python_requires=">=3.8",
)