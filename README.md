# all_pulsar_spectra

This repository fits the spectra of all pulsars with more than four flux density measurements in the [pulsar_spectra](https://github.com/NickSwainston/pulsar_spectra) catalogue.
It will be updated every time there is a new release of [pulsar_spectra](https://github.com/NickSwainston/pulsar_spectra) to create an approximate catalogue of pulsar spectrums. The gallery of fits and some basic analysis can be viewed [here](https://pulsar-spectra.readthedocs.io/).

## Credit
If using any of the data, please cite [Swainston et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022PASA...39...56S/abstract)
and [Nicholas Swainston's thesis](https://catalogue.curtin.edu.au/discovery/search?vid=61CUR_INST:CUR_ALMA>) (link will be updated once it is published).
Chapter 6 of the thesis analysed version 2.0.0.

## Running
Install [pulsar_spectra](https://github.com/NickSwainston/pulsar_spectra) then fit all pulsars spectra with

```
fit_all_pulsars.py
```
Then update the docs with
```
make_docs.py
```
Then use [sphinx](https://www.sphinx-doc.org/en/master/) to create the html with
```
sphinx-build docs Html
```
Then you can view the docs by opening the `html/index.html` file in your browser.