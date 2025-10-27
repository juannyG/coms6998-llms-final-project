from setuptools import setup, find_packages

setup(
    name="coms6998-llms-final-project",
    version="0.1.0",
    description="COMS 6998 LLMs Final Project - Scaling Language Model Experiments",
    author="Can Kerem Akbulut - Rakene Chowdhury - Juan Gutierrez",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[], # Deps are in requirements.txt
)
