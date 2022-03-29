#!/usr/bin/env python

import pathlib
from setuptools import setup

# Parent directory
HERE = pathlib.Path(__file__).parent

# The readme file
README = (HERE / "README.md").read_text()

setup(
    name="inthemoment",
    version="0.0.1",
    description="Testing ground for the method of moments",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Stephan Meighen-Berger",
    author_email="stephan.meighenberger@gmail.com",
    url='https://github.com/MeighenBergerS/inthemoment',
    license="MIT",
    install_requires=[
        "PyYAML",
        "numpy",
        "scipy",
    ],
    extras_require={
        "interactive": ["nbstripout", "matplotlib", "jupyter", "tqdm"],
    },
    packages=["inthemoment"],
    # package_data={'fennel': ["data/*.pkl"]},
    # include_package_data=True
)
