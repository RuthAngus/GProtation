# GProtation
Measuring stellar rotation periods with Gaussian processes

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

Running
--------
If you have MultiNest/PyMultiNest installed, you should be able to run 

    gprot-fit-aigrain 1 (-v)

to fit the GProtation model star 1 from the Aigrain set using MultiNest (`-v` for 
verbose MultiNest running).
