import os
import sys
from tqdm import tqdm
import pickle
import argparse

import numpy as np
import jax.numpy as jnp
from scipy.signal import convolve2d
from astropy.io import fits

from config import config_dict, intermediates_dir

sys.path.append("..")
import axionmirror.units_constants as uc
from axionmirror.nfw import rho_integral
from axionmirror.spectral import prefac
from axionmirror.map_utils import antipodal_map
from axionmirror.egrs import egrs_list_keuhr

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"


def egrs(run_dir, telescope=..., nu_arr=..., smooth=False,
         i_nu=..., i_ra_grid_shift=..., i_dec_grid_shift=..., **kwargs):
    """Make signal maps for extragalactic radio sources"""
    
    nu = nu_arr[i_nu]
    subrun_postfix = f'inu{i_nu}-ira{i_ra_grid_shift}-idec{i_dec_grid_shift}'
    include_forwardschein = False
    
    #===== Coords =====
    coords_dict = pickle.load(open(f'{run_dir}/coords/coords-{subrun_postfix}.p', 'rb'))
    ra_s = coords_dict['ra_s'] # 0 - 2pi
    dec_s = coords_dict['dec_s']
    ra_edges = coords_dict['ra_edges']
    dec_edges = coords_dict['dec_edges']
    ra_grid = coords_dict['ra_grid']
    dec_grid = coords_dict['dec_grid']
    radec_flat = coords_dict['radec_flat']
    radec_shape = coords_dict['radec_shape']
    l_grid = coords_dict['l_grid']
    b_grid = coords_dict['b_grid']
    lb_flat = jnp.asarray(coords_dict['lb_flat'])
    
    #===== Extragalactic radio source map =====
    egrs_list = egrs_list_keuhr()
    egrs_map = np.zeros_like(ra_grid)

    for egrs in egrs_list:
        
        if egrs.dec < dec_edges[0] or egrs.dec > dec_edges[-1]:
            continue
        i_dec = np.searchsorted(dec_edges, egrs.dec) - 1
        i_ra  = np.searchsorted(ra_edges, egrs.ra) - 1
        delta_dec = np.diff(dec_edges)[i_dec]
        delta_ra  = np.diff(ra_edges)[i_ra]
        
        pixel_area = delta_dec * delta_ra * np.cos(egrs.dec) # [rad^2]
        I = egrs.Snu(nu)/pixel_area # [Jy sr^-1]
        # I = 2 * nu^2 kb T / c0^2
        # [MHz^2 g sr^-1] = [MHz^2] [cm^2 MHz^2 g] [cm^-2 MHz^-2]
        T = I * uc.Jy / (2 * nu**2 * uc.kb / uc.c0**2)
        
        egrs_map[i_dec, i_ra] += T
        
    #===== rho_DM =====
    rho_integral_map = rho_integral(lb_flat).reshape(radec_shape)
    gegen_map = prefac(nu) * antipodal_map(egrs_map) * rho_integral_map
    total_map = gegen_map
    if include_forwardschein:
        forward_map = prefac(nu) * egrs_map * rho_integral_map
        total_map += forward_map
    
    #===== smooth for numerical stability =====
    sigma = 1 # [pixel]
    n_sigma = 5
    n_pix_side = 2 * n_sigma * sigma - 1
    i_c = (n_pix_side - 1) / 2
    kernel = np.zeros((n_pix_side, n_pix_side))
    
    for i in range(n_pix_side):
        for j in range(n_pix_side):
            kernel[i, j] = np.exp( - ((i-i_c)**2 + (j-i_c)**2) / (2*sigma**2) )
    kernel /= np.sum(kernel)
            
    total_map = convolve2d(total_map, kernel, mode='same')
        
    #===== Save =====
    os.makedirs(f"{run_dir}/egrs", exist_ok=True)
    np.save(f'{run_dir}/egrs/egrs-{subrun_postfix}.npy', total_map)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config')
    parser.add_argument('--use_tqdm', action='store_true', help='Use tqdm if flag is set')
    args = parser.parse_args()
    
    config = config_dict[args.config]
    
    if args.use_tqdm:
        pbar = tqdm(total=len(config['nu_arr']) * config['n_ra_grid_shift'] * config['n_dec_grid_shift'])
    for i_nu in range(len(config['nu_arr'])):
        for i_ra in range(config['n_ra_grid_shift']):
            for i_dec in range(config['n_dec_grid_shift']):
                
                egrs(
                    run_dir=f'{intermediates_dir}/{args.config}', smooth=False,
                    i_nu=i_nu, i_ra_grid_shift=i_ra, i_dec_grid_shift=i_dec, **config
                )
                
                if args.use_tqdm:
                    pbar.update()
                else:
                    print(f'i_nu={i_nu}, i_ra={i_ra}, i_dec={i_dec}')