# modelflows
Normalizing Flows for the Emulation of Grids of Stellar Models

<img src="https://github.com/mtyhon/modelflows/blob/main/images/varymassfeh.gif" width="500" height="500">

An interactive visualization can be found [here](https://mtyhon--modelflows-inyou.modal.run/).

A description of the emulator can be found in the paper, which is under review. Experiments in the paper are described in the notebooks found in this repo, which can be downloaded and ran locally.

The experiments require the grid of models [here](https://www.dropbox.com/scl/fi/yyfl98nrygg692t807uae/grid.zip?rlkey=6ausvk1wtk7r2vgc95cpdgv3m&dl=0) and the trained conditional normalizing flows (CNF<sub>dwarf</sub>, CNF<sub>giant</sub>, and CNF<sub>asfgrid</sub>) 
[here](https://www.dropbox.com/scl/fi/fvfzzcxojwv56bfable90/pretrained_flows.zip?rlkey=0pka2ddu9s0j61dk5xxsotmws&dl=0). Unzip these files into the main folder. The grid is optional, but necessary to visualize comparisons with its emulation.

If you are simply interested in the sampling the emulator to infer stellar parameters, see the following.

## Getting Started

A pip-installable package `modelflows` is available in Python 3. Run the following

```python
pip install modelflows
```

## Sampling From Grids of Stellar Models
First, import and initialize an `Emulator` instance.
```python
import modelflows as mf

emu = mf.Emulator(grid='asfgrid')
```
Currently supported grids are `asfgrid` and `cnfgiant`. If this is the first time creating an `Emulator` instance for a particular grid, the normalizing flow model will be downloaded to a local folder on your machine.

To sample from the emulator, first define a dictionary of observations. An example is as follows:

```python
obs = {
    'teff':     4728.,
    'teff_err': 80.,
    'dnu':      4.311,
    'dnu_err':  0.013,
    'numax':    41.39,
    'numax_err':0.54,
    'feh':      -0.15,
    'feh_err':  0.15
}
```

Then pass the dictionary to `Emulator.sample()` as follows:
```python
samples = emu.sample(
    num_posterior_samples=5000,
    batch_sample_size=100000,
    observations=obs,
    evstate=0,
    calc_intervals=True,
    plot_posterior=True
)
```

The use of a CUDA-capable GPU is highly recommended. The displayed arguments for `Emulator.sample()` are the following:
* `num_posterior_samples`: Number of unique draws of a grid parameter (set to age) to build the posterior distribution. Recommended at least 1000.
* `batch_sample_size`: Number of draws from the normalizing flow per iteration. These will need to fit in memory per iteration, so scale accordingly.
* `observations`: Dictionary of observations.
* `calc_intervals`: Prints highest density intervals of the posterior distribution using `arviz`.
* `plot_posterior`: Displays a corner plot of the posterior distribution.

#### Example posterior corner plot from the `asfgrid` emulation

<img src="https://github.com/mtyhon/modelflows/blob/main/images/example_posterior.png" width="500" height="500">

#### Accessing samples

`samples` provides a $N \times D + 1$ vector, where $N$ is the number of samples drawn from Sampling/Importance Resampling in the process of achieving `num_posterior_samples`. The last column represents sample likelihoods, while the first $D$ dimensions corresponds to posterior samples. `Emulator.labels` provides the identity of each column,


#### Supported observables for the `asfgrid` grid

`asfgrid` is defined only for stars with  $0.6M_{\odot}\leq M\leq5.5M_{\odot}$ for hydrogen shell-burning (RGB) subgiant stars with $\nu_{\mathrm{max}} \geq 300\mu\text{Hz}$ (`evstate=-1`),
lower RGB stars with $10 \mu\text{Hz} \leq  \nu_{\mathrm{max}} < 300 \mu\text{Hz}$ (`evstate=0`), upper RGB stars with $\nu_{\mathrm{max}} < 10\mu\text{Hz}$ (`evstate=1`), and core helium-burning (HeB) stars (`evstate=2`). 

Note that `evstate` is a keyword argument to `Emulator.sample()`. 

The following are observables that can be present in the dictionary of observations. Not all observables have to be specified for a given query. However, any given measurement must be accompanied by an uncertainty (e.g., `teff` with `teff_err`).

* `teff`: Effective temperature in K
* `teff_err`: Effective temperature uncertainty in K
* `dnu`: Asteroseismic large frequency separation in uHz
* `dnu_err`: Asteroseismic large frequency separation uncertainty in uHz
* `numax`: Asteroseismic frequency at maximum power in uHz
* `numax_err`: Asteroseismic frequency at maximum power uncertainty in uHz
* `feh`: Iron abundance in dex
* `feh_err`: Iron abundance uncertainty in dex

#### Supported observables for the `cnfgiant` grid

`cnfgiant` is defined only for hydrogen shell-burning stars with  $0.7M_{\odot}\leq M\leq2.5M_{\odot}$ (i.e., on the red giant branch). 

The following are observables that can be present in the dictionary of observations. Not all observables have to be specified for a given query. However, any given measurement must be accompanied by an uncertainty (e.g., `teff` with `teff_err`).

* `teff`: Effective temperature in K
* `teff_err`: Effective temperature uncertainty in K
* `dnu`: Asteroseismic large frequency separation in uHz
* `dnu_err`: Asteroseismic large frequency separation uncertainty in uHz
* `d02`: Asteroseismic small frequency separation in uHz
* `d02_err`: Asteroseismic small frequency separation uncertainty in uHz
* `numax`: Asteroseismic frequency at maximum power in uHz
* `numax_err`: Asteroseismic frequency at maximum power uncertainty in uHz
* `feh`: Iron abundance in dex
* `feh_err`: Iron abundance uncertainty in dex
* `lum`: Bolometric stellar luminosity in units of solar luminosity
* `lum_err`: Bolometric stellar luminosity uncertainty in units of solar luminosity

## TO-DO

- [ ] Batch version of `Emulator.sample()`
- [ ] Support for other grids


## 
