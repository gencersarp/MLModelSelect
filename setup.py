from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlmodelselect",
    version="0.1.0",
    author="MLModelSelect Contributors",
    description="Vast range of ML Models for plug and use with advanced model comparison",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gencersarp/MLModelSelect",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.2.0",
        "scipy>=1.6.0",
    ],
)
