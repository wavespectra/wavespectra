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


.. note::

    Wavespectra requires a Fortran compiler such as `gfortran` available on the system
    when installing with `pip` in order to build the watershed partitioning module.
    Installation from conda-forge includes pre-compiled versions of the code so the
    compiler is not required.


From sources
------------

The sources for wavespectra can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/wavespectra/wavespectra

Or download the `tarball`_:

.. code-block:: console

    $ curl -o wavespectra.tar.gz -L https://github.com/wavespectra/wavespectra/tarball/master
    $ tar xzf wavespectra.tar.gz

Once you have a copy of the source, you can install it from the base project directory with:

.. code-block:: console

    $ pip install .

Again, include the [extra] tag for the full install:

.. code-block:: console

   $ pip install ./[extra]

please make sure a Fortran compiler is available when installing from source.


Building the docs
-----------------

To build the docs locally, first install the docs requirements:

.. code-block:: console

    $ pip install -r requirements/docs.txt

Then run the available makefile:

.. code-block:: console

    $ make docs

Alternatively, run the sphinx_ command directly from inside the docs folder:

.. code-block:: console

    $ cd docs
    sphinx-build -b html ./ _build

and open the index file `_build/index.html` with your browser.


.. _netcdf4: https://unidata.github.io/netcdf4-python/netCDF4/index.html
.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _Github repo: https://github.com/wavespectra/wavespectra
.. _tarball: https://github.com/wavespectra/wavespectra/tarball/master
.. _development mode: https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs
.. _sphinx: https://www.sphinx-doc.org/en/master/