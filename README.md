# GProtation
Measuring stellar rotation periods with Gaussian processes

Installation
----------

    python setup.py install

You should also define an environment variable `AIGRAIN_ROTATION` that 
points to the directory of data from the Aigrain+ "Hares and Hounds" paper.

Running
--------
If you have MultiNest/PyMultiNest installed, you should be able to run 

    gprot-fit-aigrain 1

to fit the GProtation model star 1 from the Aigrain set using MultiNest.
