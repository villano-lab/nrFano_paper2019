# nrFano_paper2019
Analysis code/Jupyter notebook documentation for the 2019 NR Fano paper

# Python environment

The notebooks use quite a few libraries; you probably don't want to install them piecemeal!  If you're using the Anaconda python distribution you can set up an environment with the below conda commands.  If you're not using Anaconda python, consider switching?

```
conda env create -f nr_fano_env.yml
conda activate nr_fano
```

Rarely, the environment configuration may change.  To update your nr_fano environment:

```
conda env update --file nr_fano_env.yml
```

# Running tests

If you've made changes to the code, you can check that you haven't broken anything by:

```
cd analysis_notebooks
py.test
```

Note that the test will take a while, about half an hour.

If you've installed the nr_fano environment then you should be able to run these commands without installing any additional packages.

If you'd like to add a test, edit the `analysis_notebooks/test_local.py` file.
