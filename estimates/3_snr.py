import os
import sys
from tqdm import tqdm
import pickle
import argparse

import numpy as np
import jax.numpy as jnp

from config import config_dict, intermediates_dir

sys.path.append("..")
from axionmirror.units_constants import *
from axionmirror.snr import load_snr_list, add_image_to_map

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"


def snr(
    run_dir, snr_population=..., snr_list_realizations=..., Sgfgnu1GHz_threshold=0.,
    telescope=..., nu_arr=..., i_nu=..., i_ra_grid_shift=..., i_dec_grid_shift=..., **kwargs
):
    
    nu = nu_arr[i_nu]
    
    subrun_postfix = f'inu{i_nu}-ira{i_ra_grid_shift}-idec{i_dec_grid_shift}'
    
    coords_dict = pickle.load(open(f'{run_dir}/coords/coords-{subrun_postfix}.p', 'rb'))
    ra_edges = coords_dict['ra_edges']
    dec_edges = coords_dict['dec_edges']
    ra_s = coords_dict['ra_s']
    dec_s = coords_dict['dec_s']
    ra_grid = coords_dict['ra_grid']
    dec_grid = coords_dict['dec_grid']
    radec_flat = coords_dict['radec_flat']
    radec_shape = coords_dict['radec_shape']
    
    
    #========== Add snr to map ==========
    sig_temp_map_snr_realizations = []
    
    pixel_area_map = np.outer(
        (dec_edges[1:] - dec_edges[:-1]),
        (ra_edges[1:] - ra_edges[:-1])
    ) * np.cos(dec_s)[:,None] # [sr]
    
    n_sigma = 4
    use_gaussian_intg = False

    for snr_list in snr_list_realizations:
        
        sig_temp_map_snr = np.zeros_like(ra_grid)
        
        for snr in snr_list:
            snr_anti_ra  = snr.ra + np.pi if snr.ra < np.pi else snr.ra - np.pi
            snr_anti_dec = - snr.dec

            if snr.Sgnu(1000) > Sgfgnu1GHz_threshold:
                #---------- Gegenschein ----------
                #T_submap = snr.Sgnu(nu)*Jy * tmpl * c0**2 / (2 * nu**2 * pixel_area_submap * kb)
                T_submap_prefactor = snr.Sgnu(nu)*Jy * 1 * c0**2 / (2 * nu**2 * 1 * kb)
                # [K] = [MHz^2 g] [cm MHz]^2 [MHz]^-2 [cm^2 MHz^2 g K^-1]^-1
                add_image_to_map(
                    sig_temp_map_snr, ra_s, dec_s, ra_edges, dec_edges,
                    pixel_area_map,
                    source_ra = snr_anti_ra,
                    source_dec = snr_anti_dec,
                    image_sigma = snr.image_sigma, n_sigma=n_sigma,
                    T_submap_prefactor = T_submap_prefactor,
                    use_gaussian_intg=use_gaussian_intg, modify_mode='add'
                )

            if snr.Sfgnu(1000) > Sgfgnu1GHz_threshold:
                #---------- Front gegenschein ----------
                #T_submap = snr.Sgnu(nu)*Jy * tmpl * c0**2 / (2 * nu**2 * pixel_area_submap * kb)
                T_submap_prefactor = snr.Sfgnu(nu)*Jy * 1 * c0**2 / (2 * nu**2 * 1 * kb)
                # [K] = [MHz^2 g] [cm MHz]^2 [MHz]^-2 [cm^2 MHz^2 g K^-1]^-1
                add_image_to_map(
                    sig_temp_map_snr, ra_s, dec_s, ra_edges, dec_edges,
                    pixel_area_map,
                    source_ra = snr.ra,
                    source_dec = snr.dec,
                    image_sigma = snr.image_sigma_fg, n_sigma=n_sigma,
                    T_submap_prefactor = T_submap_prefactor,
                    use_gaussian_intg=use_gaussian_intg, modify_mode='add'
                )
                
        sig_temp_map_snr_realizations.append(sig_temp_map_snr.copy())
        
    sig_temp_map_snr_realizations = np.array(sig_temp_map_snr_realizations)
    np.save(f'{run_dir}/snr-{snr_population}/snr-{subrun_postfix}.npy', sig_temp_map_snr_realizations)

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config')
    parser.add_argument('--population', type=str, required=True, help='{fullinfo, partialinfo, graveyard}')
    args = parser.parse_args()
    
    config = config_dict[args.config]
    
    snr_list_samples = []
    for i_sample in tqdm(range(100)):
        snr_list_samples.append(
            load_snr_list(f"../outputs/snr/{args.population}_samples/{args.population}_{i_sample}.json")
        )
    
    os.makedirs(f'{intermediates_dir}/{args.config}/snr-{args.population}', exist_ok=True)

    pbar = tqdm(total=len(config['nu_arr']) * config['n_ra_grid_shift'] * config['n_dec_grid_shift'])
    for i_nu in range(len(config['nu_arr'])):
        for i_ra in range(config['n_ra_grid_shift']):
            for i_dec in range(config['n_dec_grid_shift']):
                
                snr(
                    run_dir=f'{intermediates_dir}/{args.config}',
                    snr_population=args.population,
                    snr_list_realizations=snr_list_samples,
                    Sgfgnu1GHz_threshold=1e-8, # [Jy]
                    i_nu=i_nu, i_ra_grid_shift=i_ra, i_dec_grid_shift=i_dec, **config
                )
                
                pbar.update()