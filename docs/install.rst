.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

============
Installation
============

Stable release
--------------

The latest stable release of wavespectra package
can be installed using pip or conda.

Install using pip
~~~~~~~~~~~~~~~~~~~

To install wavespectra, run this command in your terminal:

.. code-block:: console

   $ pip install wavespectra

For the full install which includes `netcdf4`_ and some other
extra libraries run this command:

.. code-block:: console

   $ pip install wavespectra[extra]

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

Install from conda
~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    $ conda install -c conda-forge wavespectra

From sources
------------

The sources for wavespectra can be downloaded from the `Github repo`_.

You can either clone the public repository using `git`_:

.. code-block:: console

    $ git clone https://github.com/wavespectra/wavespectra.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -o wavespectra.tar.gz -L https://github.com/wavespectra/wavespectra/tarball/master
    $ tar xzf wavespectra.tar.gz

.. note::

    The following commands assume that you are in the root directory of the
    source code. That is, the directory that contains the `pyproject.toml` file.

Once you have a copy of the source, you can install wavespectra with:

.. code-block:: console

    $ pip install .

Again, include the `[extra]` tag for the full install:

.. code-block:: console

   $ pip install './[extra]'


Notes to developers
-------------------

If you want to work on the wavespectra codebase, you can install it in
`development mode`_. This will allow you to edit the code and see the changes reflected
in the package without having to reinstall it.

To install wavespectra in development mode, run this command in your terminal:

.. code-block:: console

    $ pip install -e .

Include the extra dependencies by running:

.. code-block:: console

    $ pip install -e '.[extra]'

Tests
~~~~~

To run the tests, install the test dependencies:

.. code-block:: console

    $ pip install '.[test]'

and run the tests with:

.. code-block:: console

    $ pytest tests

docs
~~~~

To build the docs, install the docs dependencies:

.. code-block:: console

    $ pip install '.[docs]'

and build the docs with:

.. code-block:: console

    $ make docs



.. _netcdf4: https://unidata.github.io/netcdf4-python/netCDF4/index.html
.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _Github repo: https://github.com/wavespectra/wavespectra
.. _tarball: https://github.com/wavespectra/wavespectra/tarball/master
.. _development mode: https://setuptools.pypa.io/en/latest/userguide/development_mode.html
.. _sphinx: https://www.sphinx-doc.org/en/master/
.. _git: https://git-scm.com/