import os
import sys
from tqdm import tqdm
import pickle
import argparse

import numpy as np
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord

from config import config_dict, intermediates_dir

sys.path.append("..")
from axionmirror.map_utils import grid_edges
from axionmirror.units_constants import *

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

    
def generate_coords(
    i_nu=..., i_ra_grid_shift=..., i_dec_grid_shift=..., save_dir=...,
    telescope=..., nu_arr=..., n_ra_grid_shift=..., n_dec_grid_shift=...,
):
    
    nu = nu_arr[i_nu]
    
    def ra_pixel_size(ra): # [rad]
        return (c0 / nu) / telescope.size_ra / np.sqrt(telescope.eta_a)
    def dec_pixel_size(dec): # [rad]
        return (c0 / nu) / (telescope.size_dec * np.cos(dec - telescope.dec)) / np.sqrt(telescope.eta_a)

    #---------- grid shift ----------
    ra_mid = np.pi
    dec_mid = telescope.dec
    ra_grid_shift_step = ra_pixel_size(ra_mid) / n_ra_grid_shift
    dec_grid_shift_step = dec_pixel_size(dec_mid) / n_dec_grid_shift
    ra_shifted_mids = ra_mid + np.arange(n_ra_grid_shift) * ra_grid_shift_step
    dec_shifted_mids = dec_mid + np.arange(n_dec_grid_shift) * dec_grid_shift_step

    ra_shifted_mid = ra_shifted_mids[i_ra_grid_shift]
    dec_shifted_mid = dec_shifted_mids[i_dec_grid_shift]

    ra_edges = np.array(
        grid_edges(ra_pixel_size, telescope.ra_min, telescope.ra_max, a_mid=ra_shifted_mid)
    )
    dec_edges = np.array(
        grid_edges(dec_pixel_size, telescope.dec_min, telescope.dec_max, a_mid=dec_shifted_mid)
    )

    ra_s  = (ra_edges[:-1]  + ra_edges[1:] ) / 2
    dec_s = (dec_edges[:-1] + dec_edges[1:]) / 2
    ra_grid, dec_grid = np.meshgrid(ra_s, dec_s)
    radec_flat = np.stack([ra_grid.ravel(), dec_grid.ravel()], axis=-1)
    radec_shape = (len(dec_s), len(ra_s))

    coord_grid = SkyCoord(ra=ra_grid*u.rad, dec=dec_grid*u.rad, frame='icrs')
    l_grid = np.array(coord_grid.galactic.l.rad)
    b_grid = np.array(coord_grid.galactic.b.rad)
    lb_flat = np.stack([l_grid.ravel(), b_grid.ravel()], axis=-1)

    coords_name = f'coords-inu{i_nu}-ira{i_ra_grid_shift}-idec{i_dec_grid_shift}'
    
    coords_dict = {
        'ra_edges' : ra_edges,
        'dec_edges' : dec_edges,
        'ra_s' : ra_s,
        'dec_s' : dec_s,
        'ra_grid' : ra_grid,
        'dec_grid' : dec_grid,
        'radec_flat' : radec_flat,
        'radec_shape' : radec_shape,
        'l_grid' : l_grid,
        'b_grid' : b_grid,
        'lb_flat' : lb_flat,
    }
    
    pickle.dump(coords_dict, open(f'{save_dir}/{coords_name}.p', 'wb'))
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config')
    args = parser.parse_args()
    
    config = config_dict[args.config]
    
    os.makedirs(f"{intermediates_dir}/{args.config}/coords", exist_ok=True)
    
    pbar = tqdm(total=len(config['nu_arr']) * config['n_ra_grid_shift'] * config['n_dec_grid_shift'])
    for i_nu in range(len(config['nu_arr'])):
        for i_ra in range(config['n_ra_grid_shift']):
            for i_dec in range(config['n_dec_grid_shift']):
                
                generate_coords(
                    i_nu=i_nu, i_ra_grid_shift=i_ra, i_dec_grid_shift=i_dec,
                    save_dir=f"{intermediates_dir}/{args.config}/coords", **config
                )
                
                pbar.update()