import os
import sys
import argparse

import numpy as np
from scipy.signal import convolve2d

from config import pc_dict

sys.path.append("..")
import axionmirror.units_constants as uc
from axionmirror.nfw import rho_integral
from axionmirror.spectral import prefac
from axionmirror.map_utils import antipodal_map
from axionmirror.egrs import egrs_list_keuhr

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"


def egrs(pc):
    """Makes signal maps for extragalactic radio sources (EGRSs)."""
    
    #===== settings =====
    smooth=False
    include_forwardschein = False
    
    #===== extragalactic radio source map =====
    egrs_list = egrs_list_keuhr()
    egrs_map = np.zeros_like(pc.ra_grid)

    for egrs in egrs_list:
        
        if egrs.dec < pc.dec_edges[0] or egrs.dec > pc.dec_edges[-1]:
            continue
        i_dec = np.searchsorted(pc.dec_edges, egrs.dec) - 1
        i_ra  = np.searchsorted(pc.ra_edges, egrs.ra) - 1
        delta_dec = np.diff(pc.dec_edges)[i_dec]
        delta_ra  = np.diff(pc.ra_edges)[i_ra]
        
        pixel_area = delta_dec * delta_ra * np.cos(egrs.dec) # [rad^2]
        I = egrs.Snu_no_extrap(pc.nu)/pixel_area # [Jy sr^-1]
        # I = 2 * nu^2 kb T / c0^2
        # [MHz^2 g sr^-1] = [MHz^2] [cm^2 MHz^2 g] [cm^-2 MHz^-2]
        T = I * uc.Jy / (2 * pc.nu**2 * uc.kb / uc.c0**2)
        
        egrs_map[i_dec, i_ra] += T
        
    #===== rho_DM =====
    rho_integral_map = rho_integral(pc.lb_flat).reshape(pc.radec_shape)
    gegen_map = prefac(pc.nu) * antipodal_map(egrs_map) * rho_integral_map
    total_map = gegen_map
    if include_forwardschein:
        forward_map = prefac(pc.nu) * egrs_map * rho_integral_map
        total_map += forward_map
    
    #===== smooth for numerical stability =====
    if smooth:
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
        
    #===== save =====
    temp_name = 'egrs'
    temp_map = total_map
    os.makedirs(f"{pc.save_dir}/{temp_name}", exist_ok=True)
    np.save(f'{pc.save_dir}/{temp_name}/{temp_name}-{pc.postfix}.npy', temp_map)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config')
    args = parser.parse_args()
    
    pc = pc_dict[args.config]
    pc.iter_over_func(egrs)