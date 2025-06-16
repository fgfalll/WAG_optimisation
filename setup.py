from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="co2eor-optimizer",
    version="0.2.0",
    description="CO₂ Enhanced Oil Recovery Optimization Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Reservoir Engineering Team",
    author_email="contact@co2eor-optimizer.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(include=["co2eor_optimizer*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scipy>=1.7",
        "matplotlib>=3.5",
        "optuna>=2.10",
        "scikit-learn>=1.0",
        "scikit-optimize>=0.9",
        "bayesian-optimization>=1.4",
        "lasio>=0.30",
        "ecl2df>=0.13",
        "resfo>=1.0",
        "plotly>=5.5",
        "pyqt6>=6.4",
        "qtpy>=2.3",
        "pycuda>=2021.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.1",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
    },
    package_data={
        "co2eor_optimizer": ["py.typed"],
    },
    entry_points={
        "console_scripts": [
            "co2eor-optimizer=co2eor_optimizer.cli:main",
        ],
    },
)