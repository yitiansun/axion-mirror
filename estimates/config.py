import sys

import numpy as np

sys.path.append("..")
from axionmirror.telescopes import CHIME, HERA, CHORD, HIRAX256
from axionmirror.pixel import PixelConfig


wdir = '/n/holyscratch01/iaifi_lab/yitians/all_sky_gegenschein/axion-mirror/outputs/intermediates'

pc_dict = {}
pc_dict['CHIME'] = PixelConfig(CHIME, 10, 1, 1, wdir)
pc_dict['HERA']  = PixelConfig(HERA, 10, 4, 4, wdir)
pc_dict['CHORD'] = PixelConfig(CHORD, 10, 4, 4, wdir)
pc_dict['HIRAX'] = PixelConfig(HIRAX256, 10, 4, 4, wdir)