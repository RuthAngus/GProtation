# GProtation
Measuring stellar rotation periods with Gaussian processes

This code is no longer being maintained. If you're interested in measuring rotation periods with Gaussian processes, I 
recommend using [celerite](https://celerite.readthedocs.io/en/latest/?badge=latest) or 
[exoplanet](https://exoplanet.readthedocs.io/en/stable/).

Installation
----------

    python setup.py install

You should also define an environment variable `AIGRAIN_ROTATION` that
points to the directory of data from the Aigrain+ "Hares and Hounds" paper.
That is,

    $ ls $AIGRAIN_ROTATION
    final               noise_free          soft
    kepler_diffrot_full.tar.gz  par             sun

To download Kepler data, you will also need [kplr](http://dan.iel.fm/kplr)
installed.

Fitting also uses [emcee3](https://github.com/dfm/emcee3), so you will need
to install that as well.

You'll also need to pip-install schwimmbad, acor and tqdm.

Running
--------
After installing, you should be able to run

    gprot-fit 1 --aigrain (-v)

to fit the GProtation model star 1 from the Aigrain set using MultiNest (`-v` for
verbose emcee3 running--recommended!).  If you have [kplr](http://dan.iel.fm/kplr) installed, you should also be able to run

    gprot-fit 42 --kepler -v

The number you provide will be assumed to be a KOI number if it is <10000, and a KIC
ID number if >10000.

Now with the default parameters, on a single core, these fits will take a long time
(e.g. >1hr).  If you have multiple cores available, use the `--ncores`
option to specify how many cores you want to want to use to do the fitting.

Additional parameters can be seen with `gprot-fit -h`.
