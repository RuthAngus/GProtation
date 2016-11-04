#!/usr/bin/env python

import sys
sys.path.append('..')
from gprotation.aigrain import AigrainLightCurve
from gprotation.model import GPRotModel
from gprotation.config import POLYCHORD
sys.path.append(POLYCHORD)
import PyPolyChord.PyPolyChord as PolyChord

def fit_star(i, test=False):

    lc = AigrainLightCurve(i)
    mod = GPRotModel(lc)
    basename = str(i)
    if test:
        print('Will run polychord on star {}...')
    else:
        _ = PolyChord.run_nested_sampling(mod.polychord_lnpost, 5, 0,
                        prior=mod.polychord_prior,
                        file_root=basename)    

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('stars', nargs='*', type=int)
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    for i in args.stars:
        fit_star(i, test=args.test)


