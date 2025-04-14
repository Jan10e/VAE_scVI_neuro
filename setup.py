# -*- coding: utf-8 -*-
from setuptools import find_packages
from setuptools import setup


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="SCVI in Neuro",
    version="0.1.0",
    description="Implementing SCVI VAE on neuro data",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Jantine Broek",
    author_email="jantine.broek@proton.me",
    url="https://github.com/Jan10e/vae_scvi_neuro",
    license=license,
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "matplotlib",
        "scvi-tools",
        "numpy",
        "scikit-learn",
        "scanpy",
        "seaborn",
    ],
    zip_safe=False,
    include_package_data=True,
)
