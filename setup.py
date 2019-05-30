import os
from setuptools import setup, find_packages

_mydir = os.path.dirname(__file__)

setup(
    name='GGG-Inputs',
    desciption='Python code that creates the .mod and .vmr files used in GGG',
    author='Joshua Laughner, Sebastien Roche, Matt Kiel',
    author_email='jlaugh@caltech.edu',
    url='',
    packages=find_packages(),
)
