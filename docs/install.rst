.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

============
Installation
============

Stable release
--------------

To install wavespectra, run this command in your terminal:

.. code-block:: console

   $ pip install wavespectra

For the full install which includes `netcdf4`_ and some other
extra libraries run this command:

.. code-block:: console

   $ pip install wavespectra[extra]

These are the preferred method to install wavespectra, as they will always install the most
recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

From sources
------------

The sources for wavespectra can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/wavespectra/wavespectra

Or download the `tarball`_:

.. code-block:: console

    $ curl -OL https://github.com/wavespectra/wavespectra/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install

or alternatively in `development mode`_ with:

.. code-block:: console

   $ pip install -e .

For the full installation also install the extra requirements:

.. code-block:: console

   $ pip install -r requirements/extra.txt

.. _netcdf4: https://unidata.github.io/netcdf4-python/netCDF4/index.html
.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _Github repo: https://github.com/wavespectra/wavespectra
.. _tarball: https://github.com/wavespectra/wavespectra/tarball/master
.. _development mode: https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs