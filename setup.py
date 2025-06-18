"""
Setup configuration for TARA Universal Model.
Privacy-first conversational AI with emotional intelligence and professional domain expertise.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="tara-universal-model",
    version="1.0.0",
    author="Ramesh",
    author_email="ramesh@tara-ai.com",
    description="Privacy-first conversational AI with emotional intelligence and professional domain expertise",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ramesh/tara-universal-model",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Communications :: Chat",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "gpu": [
            "nvidia-ml-py>=11.515.0",
            "cupy-cuda11x>=12.0.0",
        ],
        "cloud": [
            "boto3>=1.28.0",
            "google-cloud-storage>=2.10.0",
            "azure-storage-blob>=12.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tara-train=tara_universal_model.training.cli:main",
            "tara-serve=scripts.serve_model:main",
            "tara-download=scripts.download_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tara_universal_model": [
            "configs/*.yaml",
            "configs/domains/*.yaml",
            "configs/training/*.yaml",
        ],
    },
    keywords=[
        "ai", "nlp", "conversational-ai", "emotional-intelligence", 
        "healthcare", "business", "education", "privacy", "local-ai",
        "transformers", "lora", "domain-adaptation"
    ],
    project_urls={
        "Bug Reports": "https://github.com/ramesh/tara-universal-model/issues",
        "Source": "https://github.com/ramesh/tara-universal-model",
        "Documentation": "https://tara-universal-model.readthedocs.io/",
    },
    zip_safe=False,
) 