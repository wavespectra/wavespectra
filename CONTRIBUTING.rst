.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/wavespectra/wavespectra/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

wavespectra could always use more documentation, whether as part of the
official wavespectra docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/wavespectra/wavespectra/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `wavespectra` for local development.

1. Fork the `wavespectra` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/wavespectra.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv wavespectra
    $ cd wavespectra/
    $ pip install -e '.[extra,test]'

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the linter
   and the tests::

    $ ruff check wavespectra tests
    $ ruff format --check wavespectra tests
    $ pytest tests

   ruff and pytest are installed with the ``test`` extra. The full python
   version matrix is tested by the CI when you push.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for all supported Python versions. The
   GitHub Actions workflow runs the test suite across the full version
   matrix when you push to your branch; make sure it passes.

Tips
----

To run a subset of tests::

$ pytest tests/test_specarray.py


Packaging Notes
---------------

Some of the packaging settings in ``pyproject.toml`` are not self-explanatory,
the reasoning behind them is recorded here.

``[project] dependencies``
    ``numpy`` is declared explicitly because the compiled ``specpart``
    extension links the numpy C API, even though numpy would also be installed
    transitively through ``scipy`` and ``xarray``.

``[tool.setuptools.packages.find] include``
    Packages are listed explicitly so that stray top-level directories created
    at build time, such as cibuildwheel's ``wheelhouse``, do not break the
    flat-layout auto-discovery.

``[tool.cibuildwheel] build``
    The CPython versions to build wheels for, which should be kept in sync
    with the supported range in the classifiers and in the test matrix of the
    testing workflow. Free-threaded builds use separate ``cp3XXt-*``
    identifiers so they are not built unless they are listed here.

``[tool.cibuildwheel] archs``
    64-bit only. 32-bit Windows wheels would be built by default but core
    dependencies such as matplotlib no longer ship win32 wheels to test the
    built wheels against.

``[tool.cibuildwheel] skip``
    musl-based linux wheels are not built, they can be added if there is
    demand for them.

``[tool.cibuildwheel] test-requires`` and ``test-command``
    Every built wheel is tested by running the partition test module, which
    exercises the compiled ``specpart`` extension end to end. This requires
    netCDF4 to be available for the wheels of every Python version built.

``[tool.pytest.ini_options] filterwarnings``
    netCDF4-python sets the ``shape`` attribute on numpy arrays, which numpy
    2.5 deprecated. It surfaces through the xarray netCDF4 backend whenever a
    netcdf file is written and is a third-party issue, nothing to fix in
    wavespectra.

``[tool.ruff.lint] extend-select``
    B015 and B018 flag comparisons and expressions whose result is discarded,
    which in tests almost always means a forgotten ``assert``.


Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

$ tbump <new-version>

and publish a release on GitHub. The GitHub Actions workflow will then run
the tests, build the wheels and the source distribution and deploy to PyPI.
