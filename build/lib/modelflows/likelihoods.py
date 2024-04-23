import numpy as np
from .utils import *

def likelihood_batch_asfgrid(ss2, observations):
    """
    Weight the output flow samples using a Gaussian centered about observed values
    with a FWHM of the observed values uncertainty.  Return only samples with non-zero
    weights.

    Args:
        ss2: Numpy array of size (batch x output dims) containing samples from the normalizing flow.
        observations: List of observed values.

    Returns:
        likelihood: Likelihood of samples being drawn based on combined weights from observables.
        ss2: All samples with non-zero weight.

    """

    
    if len(ss2.shape) == 2:
        ss2 = np.expand_dims(ss2, 0)
        
    predictions = {
        'dnu': 10**ss2[:,:,1],
        'numax': 10**ss2[:,:,2],
        'radius': 10**ss2[:,:,-2],
        'age': 10**ss2[:,:,-1],
        'teff': 10**(ss2[:,:,0]).squeeze(),
        'lum': calculate_luminosity_scaling_relation(10**(ss2[:,:,0]).squeeze(), 10**ss2[:,:,-2]).squeeze(),
    }

    
    obs_keys = ['dnu', 'numax', 'teff', 'lum']
    fac = np.ones_like(predictions['dnu']).squeeze()
    
    for obs in obs_keys:
        if obs in observations:
            prob_var = gaussian_probability(observations[obs], observations[f'{obs}_err'], predictions[obs])
            fac = fac * prob_var
     
    likelihood = fac 

    return likelihood.squeeze(), ss2.squeeze()[likelihood != 0]
    
    
def likelihood_batch_cnfgiant(ss2, fehvar, observations, teff_scaler=None):
    """
    Weight the output flow samples using a Gaussian centered about observed values
    with a FWHM of the observed values uncertainty.  Return only samples with non-zero
    weights.

    Args:
        ss2: Numpy array of size (batch x output dims) containing samples from the normalizing flow.
        fehvar: Candidate [Fe/H] values from the flow.
        observations: Dictionary of observed values.
        teff_scaler: Effective temperature StandardScaler.
        
        
    Returns:
        likelihood: Likelihood of samples being drawn based on combined weights from observables.
        ss2: All samples with non-zero weight.

    """

    if len(ss2.shape) == 2:
        ss2 = np.expand_dims(ss2, 0)

    predictions = {
        'dnu': 10**ss2[:,:,1],
        'radius': 10**ss2[:,:,-2],
        'age': 10**ss2[:,:,-1],
        'teff': 10**teff_scaler.inverse_transform(ss2[:,:,0]).squeeze(),
        'd02': 10**ss2[:,:,3],
        'lum': calculate_luminosity_scaling_relation(10**teff_scaler.inverse_transform(ss2[:,:,0]).squeeze(), 10**ss2[:,:,-2]).squeeze(),
        'feh': fehvar
    }
    
    obs_keys = ['dnu', 'teff', 'feh', 'd02', 'lum']
    fac = np.ones_like(predictions['dnu']).squeeze()

    for obs in obs_keys:
        if obs in observations:
            prob_var = gaussian_probability(observations[obs], observations[f'{obs}_err'], predictions[obs])
            fac = fac * prob_var
        
    likelihood = fac

    return likelihood.squeeze(), ss2.squeeze()[likelihood != 0]
