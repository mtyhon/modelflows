import os
import sys
import zipfile
from setuptools import setup
from setuptools.command.install import install

def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        return f.read()

install_requires = read_file("requirements.txt").splitlines()

version = {}
with open("modelflows/version.py") as fp:
    exec(fp.read(), version)


setup(
    name="modelflows",
    version=version['__version__'],
    author="Marc Hon",
    packages=["modelflows"],
    package_data={
        'modelflows': ['data/scaler/*'],
    },
    include_package_data=True,
    description="Conditional normalizing flows for emulating grids of stellar models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mtyhon/modelflows",
    license="MIT",
    install_requires=install_requires,
)
