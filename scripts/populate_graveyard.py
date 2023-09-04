"""boo!"""

import os
import sys
import argparse
from tqdm import tqdm

sys.path.append("..")
from axionmirror.graveyard import sample_graveyard_snrs
from axionmirror.snr import dump_snr_list


if __name__ == "__main__":

    # parse var as a str
    parser = argparse.ArgumentParser()
    parser.add_argument('--var', type=str, default='', help="{'', 'ti1', 'tf30', 'tf300', 'srhigh', 'srlow'} variations flag")
    args = parser.parse_args()
    
    var_name = 'graveyard_samples'
    if args.var != '':
        var_name += f'_{args.var}'

    os.makedirs(f"../outputs/snr/{var_name}", exist_ok=True)
    for i in tqdm(range(100)):
        print(f'Generating graveyard sample {i}...', flush=True)
        snr_list = sample_graveyard_snrs(t_cutoff=2e5, verbose=0, build=True, var_flag=args.var)
        dump_snr_list(snr_list, f"../outputs/snr/{var_name}/graveyard_{i}.json")