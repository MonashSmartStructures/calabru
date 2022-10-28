.. figure:: images/logo.png
   :alt: calabru_logo
   :align: center
   :scale: 50

Calabru
=======
`calabru` [1]_ is a calibration framework for model/analysis/data systems in the Python
environment.

Fun fact name "Calabru" is an irish

The calibration framework includes several state-of-the-art model updating methods such as:

- Sensitivity-based analysis
- Bayesian-based approach (in progress)

.. [1] Fun fact, "Calabru" means Calibration in the Irish language.

Installation
============
`calabru` is available on `pip` and can be installed as follows:

.. code-block::
    pip install calabru

Basic workflow
==============
The workflow for `calabru` includes:

1) Define the model/system/analysis as a python function handler.

    User first defined a finite element (FE) model in python and
    into a function handler as:

.. code-block::
    def handler():
        print("Im running FE model and getting results")
        results = "I'm some form of results from the FE model in a list"
        return results

The function handler needs to:

    - accept inputs as arguments.
    - output any form of results or measureables as a list.

2) Create in a main interface script, the main class of `calabru`.

    User made a script which calls `calabru` main class, and define
    some start parameters of the model

.. code-block::
    import calabru as cb

    calabru_obj = cb.ModelUpdating(function_handler=handler)


where the arguments are generally the function handler (``function_handle=``),
the starting parameters of the function handler (``param_list=``); and
the target response i.e. the objectives of the updating procedure (``target_list=``).


3) Run the calibration procedure.

    User do this by calling the function:

.. code-block::

    calabru_obj.update_model()

4) Get the properties of the updating procedure - more information can be found in Outputs.


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`