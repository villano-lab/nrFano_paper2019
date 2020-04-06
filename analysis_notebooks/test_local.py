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
    #_exec_notebook('nrFano_paper.ipynb')
    #_exec_notebook('5D_sigQ_parameterization.ipynb')
    _exec_notebook('silicon_Fano.ipynb')
    #_exec_notebook('QEr_2D_joint.ipynb')
    #_exec_notebook('ERNR_bands.ipynb')
    #_exec_notebook('stat_uncertainty_C.ipynb')
    #_exec_notebook('extracted_Fano.ipynb')
    #_exec_notebook('binning_systematic_Final.ipynb')
    #_exec_notebook('Qwidth_confirm.ipynb')
    #_exec_notebook('ms_correction.ipynb')
    #_exec_notebook('yield_width_compare.ipynb')
    #_exec_notebook('fitting_errors.ipynb')
    #_exec_notebook('bin_centering_correction.ipynb')
    #_exec_papermill('edelweiss_res.ipynb', None)
    #_exec_papermill('edelweiss_C_systematicErrors_allParameters.ipynb', {'Test': True})
