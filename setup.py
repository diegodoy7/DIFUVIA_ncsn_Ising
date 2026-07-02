from setuptools import setup, find_packages

setup(
    name="difuvia",
    version="0.1.0",
    packages=find_packages(exclude=["experiments"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "pyyaml",
        "tqdm",
    ],
)
