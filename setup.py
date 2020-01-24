from setuptools import setup
import glob
import os

setup(
    name="dense_basis_toolbelt",
    version="0.0.1",
    author="Kartheik Iyer",
    author_email="kartheik.iyer@dunlap.utoronto.ca",
    url = "https://github.com/kartheikiyer/dense_basis_toolbelt",
    packages=["dense_basis_toolbelt"],
    description="Extra tools for the Dense Basis package",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"], "dense_basis_toolbelt": ["trained_models/*.sed"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["matplotlib", "numpy", "scipy", "george", "sklearn", "torch", "ranger", "emcee"]
)
