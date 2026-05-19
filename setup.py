from setuptools import setup, find_packages
import os

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="rag_core",
    version="1.0.0",
    description="Shared RAG pipeline for careerbot and resumeanalyser",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
)
