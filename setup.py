# setup.py
from setuptools import setup, find_packages

setup(
    name="animus",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        # Add other dependencies
    ],
    author="Your Name",
    description="A social simulation framework using LLMs",
    python_requires=">=3.8",
)