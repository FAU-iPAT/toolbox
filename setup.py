#! /opt/conda/bin/python3
""" General PyPI compliant setup.py configuration of the package """

# Copyright 2018 FAU-iPAT (http://ipat.uni-erlangen.de/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup, find_packages  # type: ignore


__author__ = 'Dominik Haspel'
__version__ = '0.1'
__copyright__ = '2018, FAU-iPAT'
__license__ = 'Apache-2.0'
__maintainer__ = 'Dominik Haspel'
__email__ = 'has@ipat.uni-erlangen.de'
__status__ = 'Development'


def get_readme() -> str:
    """
    Method to read the README.rst file

    :return: string containing README.md file
    """
    with open('README.md') as file:
        return file.read()


# ------------------------------------------------------------------------------
#   Call setup method to define this package
# ------------------------------------------------------------------------------
setup(
    name='dieFFT.toolbox',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description='Collection of useful tools when working with keras for neural net training',
    long_description=get_readme(),
    url='https://github.com/FAU-iPAT/toolbox',
    license=__license__,
    keywords='toolbox keras tensorflow nn neural net model custom layer data generator',
    include_package_data=True,
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires='>=3.5',
    install_requires=[
        'numpy>=1.13.1',
        'scipy>=0.19.0',
        'keras>=2.0.5',
        'tensorflow>=1.3.0',
    ],
    zip_safe=False)
