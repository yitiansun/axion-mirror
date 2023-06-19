import sys
sys.path.append("..")

import numpy as np

from aatw.telescopes import CHIME

intermediates_dir = '/n/holyscratch01/iaifi_lab/yitians/all_sky_gegenschein/axions-against-the-world/outputs/intermediates'

config_dict = {}

config_dict['CHIME-nnu30-nra3-ndec3'] = {
    'telescope' : CHIME,
    'nu_arr' : np.linspace(CHIME.nu_min, CHIME.nu_max, 30), # [MHz]
    'n_ra_grid_shift' : 3,
    'n_dec_grid_shift' : 3, # set to 1, 1 to turn off
}