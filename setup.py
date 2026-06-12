from setuptools import setup, find_packages

setup(
    name="dna_classifier",
    version="0.1",
    packages=find_packages(),
    py_modules=["entrypoint", "train", "preprocess", "dataset", "evaluation_metrics"],
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0,<5.0.0",
        "accelerate>=0.20.0",
        "scikit-learn>=1.2.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "google-cloud-storage>=2.10.0",
        "biopython>=1.81",
        "einops>=0.6.1",
        "tqdm>=4.65.0",
        "pyfaidx>=0.7.0"
    ]
)
