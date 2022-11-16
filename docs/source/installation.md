Installation
============

Required Dependencies
---------------------
- Python 3.8 or later
- numpy
- scipy
- matplotlib

Instructions
------------
The easiest way to install `calabru` is to use the python package index: ::

    pip install pycba

For users wishing to develop: ::

    git clone https://github.com/MonashSmartStructures/calabru.git
    cd calabru
    pip install -e .
    
For contributions, first fork the repo and clone from your fork. `Here <https://www.dataschool.io/how-to-contribute-on-github/>`_ is a good guide on this workflow.

Tests
-----
`calabru` comes with ``pytest`` functions to verify the correct functioning of the package. 
Users can test this using: ::

    python -m pytest

from the root directory of the package.

The test creates models from other modules. As such, the following dependency are required for tests:

- ospgrillage (https://github.com/MonashSmartStructures/ospgrillage.git)
- pyCBA (https://github.com/ccaprani/pycba.git)
