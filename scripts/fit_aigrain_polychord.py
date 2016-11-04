import sys
sys.path.append('..')
from gprotation.aigrain import AigrainLightCurve
from gprotation.model import GPRotModel
from gprotation.config import POLYCHORD
sys.path.append(POLYCHORD)
import PyPolyChord.PyPolyChord as PolyChord

def fit_star(i):

    lc = AigrainLightCurve(i)
    mod = GPRotModel(lc)
    basename = str(i)
    _ = PolyChord.run_nested_sampling(mod.polychord_lnpost, 5, 0,
                    prior=mod.polychord_prior,
                    file_root=basename)    

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('stars', nargs='*', type=int)

    args = parser.parse_args()

    for i in args.stars:
        fit_star(i)


