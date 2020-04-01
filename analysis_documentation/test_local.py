import subprocess
import tempfile
import papermill as pm
import os.path as path

def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000",
                "--ExecutePreprocessor.kernel_name=python3",
                "--output", fout.name, path]
        subprocess.check_call(args)


def _exec_papermill(input_nb, args):
    output_nb = path.join('test',input_nb)
    pm.execute_notebook(input_nb, output_nb, parameters=args)


def test():
    print('Testing Jupyter notebooks...')
    _exec_notebook('nrFano_documentation.ipynb')
    _exec_notebook('lindhard_SiGe.ipynb')
    _exec_notebook('silicon_Fano.ipynb')
    #_exec_papermill('edelweiss_res.ipynb', None)
    #_exec_papermill('edelweiss_C_systematicErrors_allParameters.ipynb', {'Test': True})
