import torch
from .utils import *
from .likelihoods import *

def infer_samples(flow, cvar, device='cpu'):
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
    
    
def sample_batch_cnfgiant(flow, num_samples, observations, teff_scaler, device='cpu'):
    """
    Draws a batch of samples from the flow and weight the samples according to the observables.
    
    Args:
        flow: The conditional normalizing flow.
        num_samples: Number of samples to draw from the normalizing flow.
        observations: List of observed values.
        teff_scaler: Effective temperature StandardScaler.
        device: Device on which operations will be done on.
    
    Returns:
        candidate_param_prob: Vector of shape (9xN) of non-zero likelihood draws of input parameters.
    """    
    Zsun, Xsun = 0.0134, 0.7381
    num_marginals = num_samples   #  number of samples to be drawing from the flow

    ## Define initial parameters into the flow ##

    massvar = salpeter_imf((0.8, 2.5), size=num_marginals).T  # mass 
    fehvar = np.random.uniform(low=-4.934, high = -1.291, size=(num_marginals,)).T # logZ
    heliumvar = np.random.uniform(low=0.23, high=0.37, size=num_marginals) # Y
    alfvar = np.random.uniform(low=1, high=2.7, size=num_marginals) # alpha
    
    if ('numax' in observations) and ('numax_err' in observations):
        numaxvar = np.log10(np.random.normal(loc=observations['numax'], 
                                scale=observations['numax_err'], 
                                size=num_marginals)) # lognumax
    else:
        numaxvar = np.random.uniform(low=0.8, high = 2.477, size=(num_marginals,)).T
                                
                                
    ovcorevar =  np.random.uniform(low=0., high = 1., size=(num_marginals,)).T # f_ovcore
    ovenvvar =  np.random.uniform(low=0., high = 1., size=(num_marginals,)).T # f_ovenv

    cv = np.ones((num_marginals, 7))
    cv[:, 0] = massvar
    cv[:, 1] = fehvar
    cv[:,2] = heliumvar
    cv[:,3] = alfvar
    cv[:,4] = ovcorevar
    cv[:,5] = ovenvvar
    cv[:,6] = numaxvar

    candidate_X = 1 - (heliumvar + 10**fehvar)
    candidate_feh = np.log10((10**fehvar)/candidate_X) - np.log10(Zsun/Xsun)

    cvar = torch.Tensor(cv).to(device) ## create PyTorch tensor
    ss2 = infer_samples(flow, cvar, device=device) ## draw samples from the flow

    ## Weight samples ##
       
    combined_likelihood,  nonzero_samples, = likelihood_batch_cnfgiant(ss2, candidate_feh, observations, teff_scaler)  

    ## For simplicity we can discard samples with a probability so low they are effectively zero ## 

    remaining_likelihood = combined_likelihood[combined_likelihood != 0]
    candidate_mass = massvar[combined_likelihood != 0]
    candidate_Z = fehvar[combined_likelihood != 0]
    candidate_feh = candidate_feh[combined_likelihood != 0]
    candidate_Y = heliumvar[combined_likelihood != 0]
    candidate_alf = alfvar[combined_likelihood != 0]
    candidate_coreov = ovcorevar[combined_likelihood != 0]
    candidate_shellov = ovenvvar[combined_likelihood != 0]
    candidate_age = 10**nonzero_samples[:, -1]

    ## This defines the vector of remaining samples from which we will draw using weights defined by remaining_likelihood_feh

    candidate_param = [[a, m, f, z, y, alf, coreov, shellov] for a, m, f, z, y, alf, coreov, shellov in zip(candidate_age,
                                                                               candidate_mass,
                                                                               candidate_feh,
                                                                               candidate_Z, 
                                                                               candidate_Y,
                                                                               candidate_alf,
                                                                                candidate_coreov,
                                                                                candidate_shellov)]
                                                                                
    if len(candidate_param) != 0:
        candidate_param_prob = np.hstack([np.array(candidate_param), remaining_likelihood.reshape(-1,1) ])
    else:
        return None                                                                                
    
    return candidate_param_prob
    
    
    
    
def sample_batch_asfgrid(flow, num_samples, observations, teff_scaler=None, device='cpu', evstate=1):
    """
    Draws a batch of samples from the flow and weight the samples according to the observables.
    
    Args:
        flow: The conditional normalizing flow.
        num_samples: Number of samples to draw from the normalizing flow.
        observations: Dictionary of observed values.
        teff_scaler: Effective temperature scaling transform. Unused for asfgrid.
        device: Device on which operations will be done on.
    
    Returns:
        candidate_param_prob: Vector of shape (6xN) of non-zero likelihood draws of input parameters.
    """    

    num_marginals = num_samples   #  number of samples to be drawing from the flow

    ## Define initial parameters into the flow ##

    massvar= salpeter_imf((0.6, 5.5), size=num_marginals).T # mass samples from IMF
    
    if ('feh' in observations) and ('feh_err' in observations):
        fehvar =np.random.normal(loc=observations['feh'],
                             scale=observations['feh_err'], 
                             size=num_marginals).T # metallicity samples centered about observations
    else:
        fehvar = np.random.uniform(low=-3.0, high=0.4, size=num_marginals)
    
    cv = np.ones((num_marginals, 3))
    cv[:, 0] = massvar
    cv[:, 1] = fehvar
    cv[:, 2] = evstate

    cvar = torch.Tensor(cv).to(device) ## create PyTorch tensor
    ss2 = infer_samples(flow, cvar, device=device) ## draw samples from the flow

    ## Weight samples ##
    
    combined_likelihood,  nonzero_samples, = likelihood_batch_asfgrid(ss2, observations)  

    ## For simplicity we can discard samples with a probability so low they are effectively zero ## 

    remaining_likelihood = combined_likelihood[combined_likelihood != 0]
    candidate_mass = massvar[combined_likelihood != 0]
    candidate_feh = fehvar[combined_likelihood != 0]
    candidate_radius = 10**nonzero_samples[:, -2]
    candidate_teff = 10**nonzero_samples[:, -0]
    candidate_age = 10**nonzero_samples[:, -1]


    ## This defines the vector of remaining samples from which we will draw using weights defined by remaining_likelihood_feh

    candidate_param = [[a, m, f, r, t] for a, m, f, r, t in zip(candidate_age,
                                                                candidate_mass,
                                                                candidate_feh,
                                                                candidate_radius, 
                                                                candidate_teff)]
    
    if len(candidate_param) != 0:
        candidate_param_prob = np.hstack([np.array(candidate_param), remaining_likelihood.reshape(-1,1) ])
    else:
        return None
    
    return candidate_param_prob
    

