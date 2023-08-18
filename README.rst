.. -*- mode: rst -*-

|Codecov|_


.. |Codecov| image:: https://codecov.io/gh/sm1lla/hfs/master/graph/badge.svg?token=OGXIDWQC03
.. _Codecov: https://codecov.io/gh/sm1lla/hfs


====================================================
hfs - A library for hierarchical feature selection
====================================================

Introduction
=============

Welcome to the **hfs** repository! This library provides several hierarchical feature selection algorithms.


Getting Started
===================================================

1. Installation
-------------------------------------

The package cannot be installed with pip or conda yet so to create your package, you need to clone the ``hfs`` repository::

    $ git clone https://github.com/sm1lla/hfs.git

We recommend that you create a new virtual environment for hfs in which you install the required packages with::

    $ pip install -r requirements.txt

2. Usage
-------------------------------------------
Here is a simple example of how to use one of the hierarchical feature selection algorithms implemented in hfs:

.. code-block:: python

    from hfs import SHSELSelector
    
    # Initialize selector
    selector = SHSELSelector(hierarchy)

    # Fit selector and transform data
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)

Documentation
=============

For detailed information on how to use **hfs**, check out our complete documentation at https://hfs.readthedocs.io.

Contributing
============

We welcome contributions! If you would like to contribute to the project, 
feel free to create a pull request.


Happy feature selecting!






