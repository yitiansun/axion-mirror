import os
import sys
from tqdm import tqdm
import h5py
import numpy as np

from config import pc_dict

sys.path.append("..")
import  axionmirror.units_constants as uc
from axionmirror.spectral import dnu

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"


if __name__=="__main__":
    
    #===== settings =====
    pc = pc_dict['CHIME']
    # ['egrs', 'gsr', 'snr-fullinfo', 'snr-partialinfo', 'snr-graveyard']
    include_sources = ['snr-graveyard']
    save_name = 'snrg'

    average_over_grid_shift = True
    n_sample = 100 if np.any(['snr' in s for s in include_sources]) else 1

    g_arr_samples = np.zeros((pc.n_ra_shift, pc.n_dec_shift, n_sample, pc.n_nu))

    
    pbar = tqdm(total=pc.n_nu * pc.n_ra_shift * pc.n_dec_shift)
    for i_nu, nu in enumerate(pc.nu_s):
        for i_ra in range(pc.n_ra_shift):
            for i_dec in range(pc.n_dec_shift):
                
                pc.build(i_nu, i_ra, i_dec)
                
                bkg = np.load(f'{pc.save_dir}/bkg/bkg-{pc.postfix}.npy')[np.newaxis, ...]
                exposure = pc.exposure_map[np.newaxis, ...]
                sig_samples = np.zeros((n_sample,) + bkg.shape[1:]) # (sample, dec, ra)
                
                if 'gsr' in include_sources:
                    sig_samples += np.load(f'{pc.save_dir}/gsrJF/gsrJF-{pc.postfix}.npy')[np.newaxis, ...]
                    
                if 'egrs' in include_sources:
                    sig_samples += np.load(f'{pc.save_dir}/egrs/egrs-{pc.postfix}.npy')[np.newaxis, ...]

                for snr_pop in [s for s in include_sources if s.startswith('snr')]:
                    sig_samples += np.load(f'{pc.save_dir}/{snr_pop}/{snr_pop}-{pc.postfix}.npy')
                
                snratio_samples_map = (sig_samples / bkg) * np.sqrt(
                    2 * dnu(pc.nu) * 1e6 * exposure
                ) # (sample, dec, ra)
                
                snratio_samples = np.sqrt(np.sum(snratio_samples_map**2, axis=(1, 2))) # (sample,)
                g_arr_samples[i_ra, i_dec, :, i_nu] = (uc.gagg_CAST/uc.invGeV) / np.sqrt(snratio_samples) # [GeV^-1]
                pbar.update()
    
    if average_over_grid_shift:
        g_arr_samples = np.mean(g_arr_samples, axis=(0, 1)) # (sample, nu)
        
    save_dir = f"../outputs/plot_data/{pc.name}"
    os.makedirs(save_dir, exist_ok=True)
    
    with h5py.File(f"{save_dir}/{save_name}.h5", 'w') as hf:
        hf.create_dataset('nu', data=pc.nu_s)
        hf.create_dataset('gagg', data=g_arr_samples)