import os
import sys
from tqdm import tqdm
import h5py
import numpy as np

from config import config_dict, intermediates_dir

sys.path.append("..")
from aatw.units_constants import *
from aatw.spectral import dnu

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"


if __name__=="__main__":
    
    #===== config =====
    include_sources = ['egrs']
    plot_name = 'egrs'
    average_over_grid_shift = True
    use_tqdm = True
    
    config_name = 'CHIME-nnu30-nra3-ndec3'
    config = config_dict[config_name]
    
    
    #==============================
    nu_arr = config['nu_arr']
    telescope = config['telescope']
    n_nu = len(nu_arr)
    n_ra = config['n_ra_grid_shift']
    n_dec = config['n_dec_grid_shift']
    n_sample = 100 if 'snr' in include_sources else 1
    
    g_arr_samples = np.zeros((n_ra, n_dec, n_sample, n_nu))
    
    if use_tqdm:
        pbar = tqdm(total=n_nu * n_ra * n_dec)
    for i_nu, nu in enumerate(nu_arr):
        for i_ra in range(n_ra):
            for i_dec in range(n_dec):
                
                prefix = f'{intermediates_dir}/{config_name}'
                postfix = f'inu{i_nu}-ira{i_ra}-idec{i_dec}'
                
                bkg = np.load(f'{prefix}/bkg/bkg-{postfix}.npy')[np.newaxis, ...]
                exposure = np.load(f'{prefix}/exposure/exposure-{postfix}.npy')[np.newaxis, ...]
                sig_samples = np.zeros((n_sample,) + bkg.shape[1:]) # (sample, ra, dec)
                
                if 'gsr' in include_sources:
                    sig_samples += np.load(f'{prefix}/gsr_JF/gsr-{postfix}.npy')[np.newaxis, ...]
                    
                if 'egrs' in include_sources:
                    sig_samples += np.load(f'{prefix}/egrs/egrs-{postfix}.npy')[np.newaxis, ...]

                for snr_key in include_sources:
                    if snr_key.startswith('snr'):
                        sig_samples += np.load(f'{prefix}/{snr_key}/snr-{postfix}.npy')
                
                SNratio_samples_map = (sig_samples / bkg) * np.sqrt(
                    2 * dnu(nu) * 1e6 * exposure * telescope.t_obs_days
                ) # (sample, ra, dec)
                
                SNratio_samples = np.sqrt(np.sum(SNratio_samples_map**2, axis=(1, 2))) # (sample,)
                g_arr_samples[i_ra, i_dec, :, i_nu] = (gagg_CAST/invGeV) / np.sqrt(SNratio_samples) # [GeV^-1]
                if use_tqdm:
                    pbar.update()
                else:
                    print(f'i_nu={i_nu}, i_ra={i_ra}, i_dec={i_dec}')
    
    if average_over_grid_shift:
        g_arr_samples = np.mean(g_arr_samples, axis=(0, 1)) # (sample, nu)
        
    os.makedirs(f"../outputs/plot_data/{config_name}", exist_ok=True)
    
    with h5py.File(f"../outputs/plot_data/{config_name}/{plot_name}.h5", 'w') as hf:
        hf.create_dataset('nu', data=nu_arr)
        hf.create_dataset('gagg', data=g_arr_samples)
