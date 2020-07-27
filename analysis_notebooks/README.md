# The following nortebooks are summaries of the main logic of the NR Fano paper [![Build Status](https://travis-ci.com/villano-lab/nrFano_paper2019.svg?branch=master)](https://travis-ci.com/villano-lab/nrFano_paper2019)

## Analysis Overview with Paper Figures
nrFano_paper.ipynb

## Statement, description, and verification of the Edelweiss resolutions <br/>
edelweiss_res.ipynb

## Extraction of error on Edelweiss "C" <br/>
edelweiss_C.ipynb

## Extraction of Edelweiss fit parameters and their errors using Edelweiss functions<br/>
### Allow A, B (the yield function), aH, V (the resolution fit and voltage), and m, C (the additional width) to vary
notebook: edelweiss_C_systematicErrors_allParameters.ipynb
MCMC sample chain: data/edelweiss_C_systematicErrors_sampler_nll_allpars_gausPrior.h5
Error in the fit: data/systematic_error_fits.h5

### Allow only m and C (the additional width) to vary
notebook: edelweiss_C_statisticalErrors.ipynb
MCMC sample chain: data/edelweiss_corr_C_systematicErrors_sampler_nll_allpars_gausPrior.h5
Error in the fit: data/systematic_error_fits.h5

## Extraction of Edelweiss fit paramters using first-order corrected NR and ER functions <br/>
### Allow A, B (the yield function), aH, V (the resolution fit and voltage), and m, C (the additional width) to vary
edelweiss_fit_allParameters_seriesCorr.ipynb
data/systematic_error_fits_corr.h5

### Allow only m and C (the additional width) to vary
edelweiss_C_statisticalErrors_corr.ipynb
data/statistical_error_fits_corr.h5

## Calculation of the 2-dimensional Y, Er joint distribution <br/>
QEr_2D_joint.ipynb <br/>
<!---addendum -- check normalization against Arvind's function <br/>
addendum -- do the Er integral analytically <br/> --->

## Extraction of NR, ER band countours from 2-dimensional Y, Er joint distribution <br/>
ERNR_bands.ipynb <br/>
Qwidth_confirm.ipynb <br/>

## Correction to extracted C for multiple scatters <br/>
ms_correction.ipynb <br/>
yield_width_compare.ipynb <br/> 
fitting_errors.ipynb <br/>
bin_centering_correction.ipynb <br/>

## Extraction of the Fano factor from the Edelweiss C <br/>
stat_uncertainty_C.ipynb <br/>
extracted_Fano.ipynb <br/>
binning_systematic_Final.ipynb <br/>

## Calculation of NR Fano factor based on Lindhard <br/>
to be named

## Comparison of Dougherty silicon yield variance measurement to Lindhard <br/>
silicon_Fano.ipynb

<!--- [comment]: # the following moved to a subsequent publication --->
<!--- [comment]: # ## Dark Matter limit comparison given different Fano estimates <br/> --->
<!--- [comment]: # to be named --->
