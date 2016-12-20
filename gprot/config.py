import os

AIGRAIN_DIR = os.getenv('AIGRAIN_ROTATION', 
                        "../code/simulations/kepler_diffrot_full")
POLYCHORD = os.getenv('POLYCHORD', os.path.expanduser('~/PolyChord'))