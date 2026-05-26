"""
setup.py
────────
Install the package in editable mode:
    pip install -e .

Or with optional extras:
    pip install -e ".[dev]"
    pip install -e ".[tensorrt]"
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = ""
readme = Path(__file__).parent / "README.md"
if readme.exists():
    long_description = readme.read_text(encoding="utf-8")

setup(
    name="cv-production-kit",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description=(
        "Production-ready CV pipeline: async inference, "
        "multi-object tracking, Kalman filtering, model monitoring"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cv-production-kit",
    license="MIT",

    # ── Packages ──────────────────────────────────
    packages=find_packages(exclude=["tests*", "scripts*", "notebooks*"]),

    python_requires=">=3.9",

    # ── Core dependencies ─────────────────────────
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "ultralytics>=8.0.0",
        "onnxruntime>=1.16.0",
        "onnx>=1.14.0",
        "onnxsim>=0.4.33",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
    ],

    # ── Optional extras ───────────────────────────
    extras_require={
        "gpu": [
            "onnxruntime-gpu>=1.16.0",
        ],
        "tensorrt": [
            "tensorrt>=8.6.0",
            "pycuda>=2022.1",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "pandas>=2.0.0",
        ],
    },

    # ── CLI entry points ──────────────────────────
    entry_points={
        "console_scripts": [
            "cvkit-run=scripts.run_pipeline:main",
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],

    include_package_data=True,
    package_data={"cv_kit": ["py.typed"]},
)