import torch
import numpy as np
import arviz as az

def infer_samples(flow, cvar):
    """
    Draws samples from the normalizing flow model.

    Args:
        flow: Normalizing flow model.
        cvar: Numpy array of size (batch x conditioning variable dims) containing the conditioning variables.
        num_samples: Number of samples per conditioning variable (default 1).

    Returns:
        ss2: Numpy array of size (batch x output dims) containing samples from the normalizing flow modelling the distribution.

    """

    with torch.no_grad():
        ss2, ss_logprobs = flow(torch.Tensor(cvar).to(device)).rsample_and_log_prob((1,))
        ss_logprobs = ss_logprobs.data.cpu().numpy().squeeze()
        ss2 = (ss2.data.cpu().numpy().squeeze())

    return ss2
    
def salpeter_imf(mass_range, alpha=2.35, size=1000):
    """
    Generate stellar masses according to the Salpeter IMF.

    Parameters:
    - mass_range: tuple or list, (min_mass, max_mass) in solar masses.
    - alpha: float, the exponent of the IMF.
    - size: int, number of stellar masses to generate.

    Returns:
    - masses: ndarray, generated stellar masses according to the Salpeter IMF.
    """
    min_mass, max_mass = mass_range
    # Convert mass limits to scale between 0 and 1 for inverse transform sampling
    min_mass_scaled = min_mass**(1.0 - alpha)
    max_mass_scaled = max_mass**(1.0 - alpha)

    # Uniform random numbers for inverse transform sampling
    random_samples = np.random.uniform(low=min_mass_scaled, high=max_mass_scaled, size=size)

    # Applying the inverse of the CDF to get masses distributed according to the Salpeter IMF
    masses = random_samples**(1.0 / (1.0 - alpha))

    return masses
    
def get_intervals(sir_samples):
    """
    Gets interval statistics from the samples.

    Parameters:
    - sir_samples: Array of weighted Sampling/Importance Resampling samples from the emulator.

    Returns:
    - hdis: Upper and lower HDI interval across each dimension.
    - median: Median of the posterior across each dimension.
    """

    hdis, medians = az.hdi(np.array(sir_samples[:,:-1])),  np.median(np.array(sir_samples[:,:-1]), axis=0)
    return hdis_medians
    
def gaussian_probability(observed, observed_err, predicted):
    """
    Calculate the Gaussian probability for a predicted value given an observed value and uncertainty.
    If observed value or error is None, return 1.
    
    Parameters:
    - observed: Input observed values.
    - observed_err: Input observed uncertainties.
    - predicted: Predicted values.
        
    Returns:
    - probabilities, 1 or Gaussian-weighted
    
    """
    if observed is None or observed_err is None:
        return 1.
    else:
        return 1. / (observed_err * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power(predicted - observed, 2) / observed_err**2).squeeze()
        
        
def calculate_luminosity_scaling_relation(teff, radius):
    """
    From a temperature and radius, determine luminosity in solar units.

    Args:
        teff: Effective temperature in Kelvin.
        radius: Radius in units of solar radii.

    Returns:
        lum: Bolometric luminosity in units of solar luminosities.

    """
    sol_teff = 5777  # Effective temperature of the Sun in Kelvin

    # Calculate the luminosity in solar units using the scaling relation
    lum = (radius ** 2) * ((teff / sol_teff) ** 4)

    return lum

