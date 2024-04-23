import torch, zuko, random, corner, os, requests
import numpy as np
import matplotlib.pyplot as plt

from .utils import *
from .sampler import *
from .likelihoods import *
from .config import FLOW_DIR, SCALER_DIR
from joblib import load
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_n

assert zuko.__version__ == '0.3.2'

def download_file_with_progress(url, filename):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024*1024

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

class Emulator:
    """The Conditional Normalizing Flow Emulating a Grid of Stellar Evolutionary Models"""
    
    def __init__(self, grid):
        assert grid in ['cnfgiant', 'asfgrid'], "grid must be 'cnfgiant' or 'asfgrid'"
        self.grid = grid
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.grid == 'cnfgiant':
            if not os.path.exists(FLOW_DIR):
                os.makedirs(FLOW_DIR)
            loadpath = os.path.join(FLOW_DIR, 'cnf_giant.checkpoint')
            
            if not os.path.exists(loadpath):
                print('Existing %s model not found. Downloading from source...' %self.grid)
                dropbox_url = 'https://www.dropbox.com/scl/fi/yo3e8bwc8zu23o25ynlzd/cnf_giant.checkpoint?rlkey=kv0vgovsda5jwep1m04dyfigp&dl=1'
                download_file_with_progress(dropbox_url, loadpath)
      
            checkpoint = torch.load(loadpath, self.device)
            self.flow = zuko.flows.NSF(features=8,  context=7, transforms=10, hidden_features=[256] * 10).to(self.device)
            self.flow.load_state_dict(checkpoint['model_state_dict'])
            self.teff_scaler =  load(os.path.join(SCALER_DIR, 'cnfgiant_teff.scaler'))
            self.d01_scaler = load(os.path.join(SCALER_DIR, 'cnfgiant_d01.scaler'))
            self.mms_shell_scaler = load(os.path.join(SCALER_DIR, 'cnfgiant_mms_shell.scaler'))
            self.mms_core_scaler = load(os.path.join(SCALER_DIR, 'cnfgiant_mms_core.scaler'))
            self.sampler_func = sample_batch_cnfgiant
            self.labels =  ['$\\tau$ (Gyr)', '$M$', '[Fe/H]', '$\\log_{10}(Z)$', '$Y$', '$\\alpha$', 
          '$f_{\\mathrm{ov, core}}$', '$f_{\\mathrm{ov, env}}$']
            
        elif self.grid == 'asfgrid':
            if not os.path.exists(FLOW_DIR):
                os.makedirs(FLOW_DIR)
            loadpath = os.path.join(FLOW_DIR, 'cnf_asfgrid.checkpoint')
          
            if not os.path.exists(loadpath):
                print('Existing %s model not found. Downloading from source...' %self.grid)
                dropbox_url = 'https://www.dropbox.com/scl/fi/owk1pscj8tg3irdrm16px/cnf_asfgrid.checkpoint?rlkey=wxijcgbsxxxcq6xbhe70w2jiy&dl=1'
                download_file_with_progress(dropbox_url, loadpath)
               
                  
            checkpoint = torch.load(loadpath, self.device)
            self.flow = zuko.flows.NSF(features=5,  context=3, transforms=8, hidden_features=[512] * 8).to(self.device)
            self.flow.load_state_dict(checkpoint['model_state_dict'])   
            self.teff_scaler = self.d01_scaler = self.mms_shell_scaler = self.mms_core_scaler = None                          
            self.sampler_func = sample_batch_asfgrid
            self.labels = ['$\\tau$ (Gyr)', '$M (M_{\\odot})$', '[Fe/H] (dex)', '$R (R_{\\odot})$', '$T_{\\mathrm{eff}}$ (K)']

            
    def sample(self, num_posterior_samples, batch_sample_size, observations, plot_posterior=False, calc_intervals=False, notebook=True, **kwargs):
        """
        Samples from the emulated grid.
    
        Args:
            num_posterior_samples: Number of unique samples to draw from the posterior distribution.
            batch_sample_size: Size of batch to sample from the normalizing flow.
            observations: Dictionary of observed values.
            plot_posterior: Boolean flag to draw a corner plot for the posterior or not.
            calc_intervals: Boolean flag to calculate intervals from the posterior, determine has [lower HDI, median, upper HDI].
            calc_intervals: Progress bar in a notebook environment instead of a CLI.   
            
        Returns:
            posterior_samples: Posterior samples from the emulated grid.
        """  
        if notebook:
            pbar = tqdm_n(total=num_posterior_samples)
        else:
            pbar = tqdm(total=num_posterior_samples)
    
        candidate_param_prob = self.sampler_func(self.flow, batch_sample_size, observations, teff_scaler=self.teff_scaler, device=self.device, **kwargs)
        sir_samples = np.array(random.choices(candidate_param_prob, 
                             weights=(candidate_param_prob[:, -1]), k=10000))
                             
        unique_age, unique_indices = np.unique(sir_samples[:,0], return_index=True)  
              
        pbar = tqdm(total=num_posterior_samples)
        pbar.update(len(unique_age))
        
        while len(unique_age) < num_posterior_samples:
            extra_param_prob = self.sampler_func(self.flow, batch_sample_size, observations, teff_scaler=self.teff_scaler, device=self.device, **kwargs)   
            if extra_param_prob is None:
                continue
            extra_samples = np.array(random.choices(extra_param_prob, 
                                 weights=(extra_param_prob[:, -1]), k=10000))
            sir_samples = np.concatenate([sir_samples, extra_samples])
            unique_age, unique_indices = np.unique(sir_samples[:,0], return_index=True)
            print(len(unique_age))
            pbar.update(len(unique_age) - pbar.n)
            pbar.set_description(f'Unique Samples: {len(unique_age)}')

        pbar.close()

        if plot_posterior:
            fig = plt.figure(figsize=(12,12))
            fs = 16
            corner.corner(sir_samples[:, :-1], labels=self.labels,label_kwargs={"fontsize": fs}, fig=fig, smooth = 1.,
              hist_kwargs={'density':True})

            for ax in fig.get_axes():
                ax.tick_params(axis='both', labelsize=fs-6) 
            plt.show()
        
        if calc_intervals:
            hdis, medians = az.hdi(np.array(sir_samples[:,:-1])),  np.median(np.array(sir_samples[:,:-1]), axis=0)
            for i, (hdi, median) in enumerate(zip(hdis,medians)):
                print('%s: %.2f +%.2f -%.2f' %(self.labels[i], median, median-hdi[0], hdi[1] - median))
        
        return sir_samples 
     

        
        
