import sys

import numpy as np

sys.path.append("..")
from axionmirror.telescopes import CHIME, HIRAX_1024, HERA


intermediates_dir = '/n/holyscratch01/iaifi_lab/yitians/all_sky_gegenschein/axions-against-the-world/outputs/intermediates'


config_dict = {}

config_dict['CHIME-10-4-4'] = {
    'telescope' : CHIME,
    'nu_arr' : np.linspace(CHIME.nu_min, CHIME.nu_max, 10), # [MHz]
    'n_ra_grid_shift' : 4,
    'n_dec_grid_shift' : 4, # set to 1, 1 to turn off
}

# config_dict['CHIME-10-4-4'] = {
#     'telescope' : CHIME,
#     'nu_arr' : np.linspace(CHIME.nu_min, CHIME.nu_max, 30), # [MHz]
#     'n_ra_grid_shift' : 3,
#     'n_dec_grid_shift' : 3, # set to 1, 1 to turn off
# }

# config_dict['HIRAX-1024-10-4-4'] = {
#     'telescope' : HIRAX_1024,
#     'nu_arr' : np.linspace(HIRAX_1024.nu_min, HIRAX_1024.nu_max, 30), # [MHz]
#     'n_ra_grid_shift' : 3,
#     'n_dec_grid_shift' : 3, # set to 1, 1 to turn off
# }

# config_dict['HERA-10-4-4'] = {
#     'telescope' : HERA,
#     'nu_arr' : np.linspace(HERA.nu_min, HERA.nu_max, 30), # [MHz]
#     'n_ra_grid_shift' : 3,
#     'n_dec_grid_shift' : 3, # set to 1, 1 to turn off
# }


#===== old =====

config_dict['CHIME-nnu30-nra3-ndec3'] = {
    'telescope' : CHIME,
    'nu_arr' : np.linspace(CHIME.nu_min, CHIME.nu_max, 30), # [MHz]
    'n_ra_grid_shift' : 3,
    'n_dec_grid_shift' : 3, # set to 1, 1 to turn off
}
config_dict['CHIME-nnu10-nra5-ndec5'] = {
    'telescope' : CHIME,
    'nu_arr' : np.linspace(CHIME.nu_min, CHIME.nu_max, 10), # [MHz]
    'n_ra_grid_shift' : 5,
    'n_dec_grid_shift' : 5, # set to 1, 1 to turn off
}

config_dict['HIRAX-1024-nnu30-nra3-ndec3'] = {
    'telescope' : HIRAX_1024,
    'nu_arr' : np.linspace(HIRAX_1024.nu_min, HIRAX_1024.nu_max, 30), # [MHz]
    'n_ra_grid_shift' : 3,
    'n_dec_grid_shift' : 3, # set to 1, 1 to turn off
}

config_dict['HERA-nnu30-nra3-ndec3'] = {
    'telescope' : HERA,
    'nu_arr' : np.linspace(HERA.nu_min, HERA.nu_max, 30), # [MHz]
    'n_ra_grid_shift' : 3,
    'n_dec_grid_shift' : 3, # set to 1, 1 to turn off
}