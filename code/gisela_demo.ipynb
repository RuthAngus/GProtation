{
 "metadata": {
  "name": "",
  "signature": "sha256:c6e6116795e49087f8a3c6eb68107fb046af9215ab49f183ba91d5cbbf8a80e3"
 },
 "nbformat": 3,
 "nbformat_minor": 0,
 "worksheets": [
  {
   "cells": [
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "import numpy as np\n",
      "import matplotlib.pyplot as plt\n",
      "%matplotlib inline\n",
      "import gisela\n",
      "from tools import bin_data, simple_acf\n",
      "import triangle\n",
      "from Kepler_ACF import corr_run"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 1
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Load example data set"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "id = \"0\"\n",
      "x, y = np.genfromtxt(\"simulations/%s.txt\" % id).T\n",
      "yerr = np.ones_like(y) * 1e-5\n",
      "plt.errorbar(x, y, yerr=yerr, fmt=\"k.\")"
     ],
     "language": "python",
     "metadata": {},
     "outputs": []
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Measure the rotation period using the ACF method in order to get a rough guess"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "p_init, acf, lags, flag = simple_acf(x, y)\n",
      "print p_init"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "1.16471805\n"
       ]
      }
     ],
     "prompt_number": 3
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Bin the data and truncate it for speed!"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "npts = int(p_init / 10. * 48)  # 10 points per period\n",
      "xb, yb, yerrb = bin_data(x, y, yerr, npts)\n",
      "cutoff = 100\n",
      "m = xb < cutoff\n",
      "xb, yb, yerrb = xb[m], yb[m], yerrb[m]"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 4
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Run the MCMC"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "gs = gisela.gisela(xb, yb, yerrb)\n",
      "theta_init = np.log([1., 1., 1., 1., p_init])\n",
      "sampler = gs.MCMC(id, theta_init);"
     ],
     "language": "python",
     "metadata": {},
     "outputs": []
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Make triangle plot"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "nwalkers, nsteps, ndims = np.shape(sampler)\n",
      "flat = np.reshape(sampler, (nwalkers * nsteps, ndims))\n",
      "mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(flat, [16, 50, 84], axis=0)))\n",
      "mcmc_result = np.array([i[0] for i in mcmc_result])\n",
      "print(\"\\n\", np.exp(np.array(mcmc_result[-1])), \"period (days)\", \"\\n\")\n",
      "\n",
      "fig_labels = [\"A\", \"l2\", \"l1\", \"s\", \"P\"]\n",
      "fig = triangle.corner(flat, labels=fig_labels)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": []
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Plot the MAP prediction"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "theta = np.exp(np.array(mcmc_result))\n",
      "k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], theta[4])\n",
      "gp = george.GP(k)\n",
      "gp.compute(x, yerr)\n",
      "xs = np.linspace(x[0], x[-1], 1000)\n",
      "mu, cov = gp.predict(y, xs)\n",
      "\n",
      "plt.errorbar(x, y, yerr=yerr, **reb)\n",
      "plt.xlabel(\"Time (days)\")\n",
      "plt.ylabel(\"Normalised Flux\")\n",
      "plt.plot(xs, mu)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": []
    }
   ],
   "metadata": {}
  }
 ]
}