import numpy

"""mpfit: Non-linear least squares minimisation using the
   Levenberg-Marquardt technique with optional fixed, limited or tied
   parameters.

   This module is intended primarily for fitting non-linear models to
   data.  The user should supply a Python function which, given the
   data and parameters, computes an array of weighted deviations
   between model and data. The user can then use the function
   mpfit.mpfit to minimise the sum of the squares of the deviates,
   i.e. the chi-squared.

   The function mpfit.mpfit takes as arguments the user defined
   function and a numpy array containing inital guesses for the
   parameters. The data, and any measurement uncertainties, are passed
   to the user-defined function using the functkw keyword of
   mpfit.mpfit. It returns an instance of the mpfit.results class,
   which contains the best-fit values of the parameters as well as the
   covariance matrix and formal parameter uncertainties.

   Here is a quick example using a user function defined in this module:
      import mpfit
      import numpy
      import pylab
      N = 100
      x = numpy.r_[0:10:N*1j]
      p0 = numpy.array([5.7, 0.22])
      noise = mpfit.F(x, p0) + numpy.random.randn(N) * 10
      err = numpy.ones(N) * 10
      pylab.errorbar(x, y, yerr = err)
      p1 = numpy.array([6,0.2])
      fa = {'x':x, 'y':y + noise, 'err':err}
      m = mpfit(mpfit.example_func, p1, functkw = fa)
      print 'parameters = ', m.params
      print 'parameters = ', m.pcerror 
# Note how the errors are underestimated because the two parameters
# are strongly correlated!
      pylab.plot(x, mpfit.F(x, m.params))
      
   The user function should be built according to a specific set of
   rules, which are described below and illustrated in the example_func
   function.

   Simple constraints can be placed on parameter values by using the
   fixed, limited & limits, and/or tied keywords of mpfit.mpfit, as
   described in the documentation of function mpfit itself.
   *** NB: in the present version, only the fixed keyword is
       functional and tested. The limited / limits keywords don't work
       properly and the tied keyword is untested. ***

   See the documentation for the individual functions and classes for
   more details, including some keywords not described here.

   USER_DEFINED FUNCTION:

   The user-defined function should take one mandatory argument, a
   numpy array of parameter values. It should also have a keyword
   fjac, which is set if analytical derivatives are desired (see
   below). Additional keywords can be used to pass the data, and
   optionally measurement errors. These are passed to the user-defined
   function via the functkw keyword to mpfit. 

   The user-defined function should return a 2-element list
   containing:
   - a status flag (negative if the call failed, zero otherwise)
   - an one-dimensional array of (optionally weighted) residuals
   - a 2-d array of analytical derivatives (if fjac is None, the latter
   is also None).

   An example user-defined function, named example_func, is supplied
   in this module.  The keyword parameters x, y, and err in this
   example are suggestive but not required. There are no restrictions
   on the number of dimensions in x, y or err, but the deviates
   must*be returned in a one-dimensional numpy array.
 
   ANALYTICAL DERIVATIVES:

   In the search for the best-fit solution, mpfit.mpfit by default
   calculates derivatives numerically via a finite difference
   approximation. The user-supplied function need not calculate the
   derivatives explicitly. In general, this is often easier and can be
   faster than computing the derivatives analytically. However, if you
   wish to compute them analytically, then set the autoderivative
   keyword of mpfitfun to 0.
  
   If analytical derivatives are requested, they should be returned in
   an M x N array, where M is the number of data points and N is the
   number of parameters. If this array was called dp, then dp[i,j]
   would be the derivative at the ith point with respect to the jth
   parameter. If the autoderivative keyword of mpfit.mpfit is zero, upon
   input to the user function, fjac is set to a vector with the same
   length as p, with a value of 1 for a parameter which is free, and a
   value of zero for a parameter which is fixed (and hence no
   derivative needs to be calculated).
 
   REFERENCES

   MINPACK-1, Jorge More, available from netlib (www.netlib.org).
   "Optimization Software Guide," Jorge More and Stephen Wright, 
      SIAM, *Frontiers in Applied Mathematics*, Number 14.
   More', Jorge J., "The Levenberg-Marquardt Algorithm:
      Implementation and Theory," in *Numerical Analysis*, ed. Watson,
      G. A., Lecture Notes in Mathematics 630, Springer-Verlag, 1977.

   IDL MPFIT documentation:
      http://www.physics.wisc.edu/~craigm/idl/fitting.html

   Mark River's Python MPFIT documentation:
      http://cars9.uchicago.edu/software/python/mpfit.html

   MODIFICATION HISTORY
 
   Translated from MINPACK-1 in FORTRAN, Apr-Jul 1998, Craig Markwardt

   Translated from MPFIT (Craig Markwardt's IDL package) to Python,
      August, 2002. Mark Rivers

   Updated to use Numpy and somewhat simplified, Jan 2010, Suzanne Aigrain

   This software is provided as is without any warranty
   whatsoever. Permission to use, copy, modify, and distribute
   modified or unmodified copies is granted, provided this copyright
   and disclaimer are included unchanged.
   """

class machar:
   """mpfit.machar: class to contain relevant machine limits"""
   def __init__(self):
      f = numpy.MachAr()
      self.machep = f.eps
      self.minnum = f.xmin
      self.maxnum = f.xmax         
      self.maxlog = numpy.log(self.maxnum)
      self.minlog = numpy.log(self.minnum)
      self.rdwarf = numpy.sqrt(self.minnum*1.5) * 10
      self.rgiant = numpy.sqrt(self.maxnum) * 0.1
MACH = machar()

class results:
   """mpfit.results: class to contain results of fitting process

   Attributes:

   .status
      An integer status code is returned.  All values greater than zero can
      represent success (however .status == 5 may indicate failure to
      converge). It can have one of the following values:
      -16
         A parameter or function value has become infinite or an undefined
         number.  This is usually a consequence of numerical overflow in the
         user's model function, which must be avoided.
      -15 to -1 
         These are error codes that either MYFUNCT or iterfunct may return to
         terminate the fitting process.  Values from -15 to -1 are reserved
         for the user functions and will not clash with MPFIT.
      0  Improper input parameters.
      1  Both actual and predicted relative reductions in the sum of squares
         are at most ftol.
      2  Relative error between two consecutive iterates is at most xtol
      3  Conditions for status = 1 and status = 2 both hold.
      4  The cosine of the angle between fvec and any column of the jacobian
         is at most gtol in absolute value.
      5  The maximum number of iterations has been reached.
      6  ftol is too small. No further reduction in the sum of squares is
         possible.
      7  xtol is too small. No further improvement in the approximate solution
         x is possible.
      8  gtol is too small. fvec is orthogonal to the columns of the jacobian
         to machine precision.
 
   .fnorm
      The value of the summed squared residuals for the returned parameter
      values.
 
   .covar
      The covariance matrix for the set of parameters returned by mpfit.
      The matrix is NxN where N is the number of  parameters.  The square root
      of the diagonal elements gives the formal 1-sigma statistical errors on
      the parameters if errors were treated "properly" in fcn.
      Parameter errors are also returned in .perror.
      To compute the correlation matrix, pcor, use this example:
         cov = mpfit.covar
         pcor = cov * 0.
         for i in range(n):
            for j in range(n):
               pcor[i,j] = cov[i,j]/numpy.sqrt(cov[i,i]*cov[j,j])
      If nocovar is set or mpfit terminated abnormally, then .covar is set to None.
 
   .niter
      The number of iterations completed.
 
   .pcerror
      The formal 1-sigma errors in each parameter, computed from the
      covariance matrix and reduced chi-squared of the fit. If a
      parameter is held fixed, or if it touches a boundary, then the
      error is reported as zero.
      If the fit is unweighted (i.e. no errors were given, or the weights
      were uniformly set to unity), then .perror will probably not represent
      the true parameter uncertainties.  
      NB: .perror contains the errors before scaling by the
      reduced-chi-squared value
   """
   def __init__(self):
      self.niter = 0
      self.params = None
      self.covar = None
      self.perror = None
      self.status = 0
      self.dof = None
      self.bestnorm = None
      self.pcerror = None

def defiter(fcn, x, iter, fnorm = None, functkw = None, names = None, \
               ifree = None, pformat = '%.10g', dof = 1):
   """Print out parameter values at each iteration"""
   if (fnorm == None):
      [status, fvec] = call(fcn, x, functkw, damp = damp, tied = tied)
      fnorm = enorm(fvec)**2
   print "Iter ", ('%6i' % iter), "   CHI-SQUARE = ", ('%.10g' % fnorm), \
       " DOF = ", ('%i' % dof)
   ## Determine which parameters to print
   xfree = x[ifree]
   nprint = len(xfree)
   if names != None: namef = names[ifree]
   for i in range(nprint):
      p = '   P' + str(i) + ' = '
      if (names != None):
         if namef[i].strip() != '': p = '   ' + namef[i] + ' = '
      print p + (pformat % xfree[i]) + '  '
   return(0)

def F(x, p):
   """Example user-defined model function."""
   return p[0] * x * numpy.exp(p[1] * x)

def FGRAD(x, p, j):
   """Example user-defined analytical derivatives function."""
   if j == 0: return F(x, p) / p[0]
   if j == 1: return F(x, p) * p[1]
   return None

def example_func(p, fjac=None, x=None, y=None, err=None):
   """Example user-defined residuals function to use with mpfitfun.
   
   Uses functions mpfit.F(x, p) to computes the model given the data x
   and the parameters p, and mpfit.FGRAD(x, p, j) to compute the
   derivative of the model with respect to parameter p."""
   model = F(x, p)
   residuals = y - model
   if err != None: residuals /= err
   if fjac != None: 
      pderiv = numpy.zeros((len(x), len(p)))
      for j in range(len(p)):
         if fjac[j] == 0: continue
         pderiv[:,j] = FGRAD(x, p, j)
   else:
      pderiv = None
   status = 0
   return [status, residuals, pderiv]

def mpfit(fcn, xall, functkw = {}, fixed = None, limited = None, \
             limits = None, names = None, tied = None, \
             ftol = 1.e-10, xtol = 1.e-10, gtol = 1.e-10, \
             damp = 0., maxiter = 200, factor = 100., nprint = 1, \
             iterfunct = defiter, iterkw = {}, nocovar = 0, \
             fastnorm = 0, rescale = 0, autoderivative = 1, quiet = 0, \
             diag = None, epsfcn = None):
   """Perform the minimisation.

   Arguments:

   fcn:
      The function to be minimized. The function should return the weighted
      deviations between the model and the data, as described above.
 
   xall:
      An array of starting values for each of the parameters of the model.
      The number of parameters should be fewer than the number of measurements.

   Keywords:
 
   functkw:
      A dictionary which contains the parameters to be passed to the
      user-supplied function specified by fcn via the standard Python
      keyword dictionary mechanism. This is the way you can pass additional
      data to your user-supplied function without using global variables.
      If functkw = {'xval':[1.,2.,3.], 'yval':[1.,4.,9.], 'errval':[1.,1.,1.] }
      then the user supplied function should be declared like this:
      def myfunct(p, fjac=None, xval=None, yval=None, errval=None): 
         Default: {} (No extra parameters are passed to the
         user-supplied function.)

   autoderivative:
      If this is set, derivatives of the function will be computed
      automatically via a finite differencing procedure.  If not set, then
      fcn must provide the (analytical) derivatives.
         Default: set (=1). To supply your own analytical derivatives,
         explicitly set autoderivative=0

   fixed: if set, should be a numpy array of lenght M where M is the
      number of parameters. Each element should be set to 0 if the
      corresponding parameter is to be allowed to vary, and 1
      otherwise.
         Default: None (No fixed parameters)

   limited: if set, should be a 2xM element numpy array, with the
      first column corresponding to lower limits and the second to
      upper limits.  Each element should be set to 1 if a limit is to
      be imposed, and 0 otherwise.
         Default: None (No limited parameters)
   *** Warning: Doesn't work!!! ***
 
   limits: if set, should be a 2xM element numpy array, containing the
      limiting values to use (only values corresponding to non-zero
      elements of limited are taken into consideration)
         Default: All limits are set to zero
   *** Warning: Doesn't work!!! ***

   tied: is set, should be a numpy string array of length M. Each
      element should be set to a string expression which "ties" the
      parameter to other free or fixed parameters (or an empty string
      if the corresponding parameter is not to be tied). Any
      expression involving constants and the parameter array p are
      permitted. For example, if parameter 2 is always to be twice
      parameter 1 then use: tied[2] = '2 * p[1]'. Since they are
      totally constrained, tied parameters are considered to be fixed.
         Default: None (no tied parameters)
   *** Warning: Not tested!!! ***
 
   fastnorm:
      Set this keyword to select a faster algorithm to compute sum-of-square
      values internally.  For systems with large numbers of data points, the
      standard algorithm can become prohibitively slow because it cannot be
      vectorized well.  By setting this keyword, mpfit will run faster, but
      it will be more prone to floating point overflows and underflows. Thus, setting
      this keyword may sacrifice some stability in the fitting process.
         Default: clear (=0)
              
   ftol:
      A nonnegative input variable. Termination occurs when both the actual
      and predicted relative reductions in the sum of squares are at most
      ftol (and status is accordingly set to 1 or 3).  Therefore, ftol
      measures the relative error desired in the sum of squares.
         Default: 1E-10

   gtol:
      A nonnegative input variable. Termination occurs when the cosine of
      the angle between fvec and any column of the jacobian is at most gtol
      in absolute value (and status is accordingly set to 4). Therefore,
      gtol measures the orthogonality desired between the function vector
      and the columns of the jacobian.
         Default: 1e-10
 
   iterkw:
      The keyword arguments to be passed to iterfunct via the dictionary
      keyword mechanism.  This should be a dictionary and is similar in
      operation to functkw.
         Default: {} (No arguments are passed.)
 
   iterfunct:
      The name of a function to be called upon each iteration of the
      mpfit routine. It should be declared in the following way:
         def iterfunct(myfunct, p, iter, fnorm, functkw=None, 
                       quiet=0, dof=None, [iterkw keywords here])
      iterfunct must accept all the keyword parameters functw and quiet.   
         myfunct:  The user-supplied function to be minimized,
         p:        The current set of model parameters
         iter:     The iteration number
         functkw:  The arguments to be passed to myfunct.
         fnorm:    The chi-squared value.
         quiet:    Set when no textual output should be printed.
         dof:      The number of degrees of freedom, normally the number of points
                   less the number of free parameters. 
      In implementation, iterfunct can perform updates to the terminal or
      graphical user interface, to provide feedback while the fit proceeds.
      If the fit is to be stopped for any reason, then iterfunct should return a
      a status value between -15 and -1. Otherwise it should return None
      (e.g. no return statement) or 0.
      In principle, iterfunct should probably not modify the parameter values,
      because it may interfere with the algorithm's stability.  In practice it
      is allowed.
         Default: an internal routine is used to print the parameter
         values. Set iterfunct=None if there is no user-defined
         routine and you don't want the internal default routine be
         called.
 
   maxiter:
      The maximum number of iterations to perform.  If the number is exceeded,
      then the status value is set to 5 and MPFIT returns.
         Default: 200 iterations
 
   nocovar:
      Set this keyword to prevent the calculation of the covariance matrix
      before returning (see COVAR)
         Default: clear (=0)  The covariance matrix is returned
 
   nprint:
      The frequency with which iterfunct is called.  A value of 1 indicates
      that iterfunct is called with every iteration, while 2 indicates every
      other iteration, etc.  Note that several Levenberg-Marquardt attempts
      can be made in a single iteration.
        Default value: 1
 
   quiet:
      Set this keyword when no textual output should be printed by mpfit
 
   damp:
      A scalar number, indicating the cut-off value of residuals where
      "damping" will occur.  Residuals with magnitudes greater than this
      number will be replaced by their hyperbolic tangent.  This partially
      mitigates the so-called large residual problem inherent in
      least-squares solvers (as for the test problem CURVI,
      http://www.maxthis.com/curviex.htm).
      A value of 0 indicates no damping.
         Default: 0
         Note: DAMP doesn't work with autoderivative=0
 
   xtol:
      A nonnegative input variable. Termination occurs when the relative error
      between two consecutive iterates is at most xtol (and status is
      accordingly set to 2 or 3).  Therefore, xtol measures the relative error
      desired in the approximate solution.
         Default: 1E-10
 
   Outputs:
 
   Returns an instance of class mpfit.results. See documentation for
   this class for details.
   """

   ## Initialise a few bits and bobs
   res = results()
   machar = MACH
   machep = MACH.machep
   if type(xall) != numpy.ndarray: xall = numpy.array(xall)
   npar = len(xall)
   fnorm = -1.
   fnorm1 = -1.
   if quiet != 0: nprint = -1

   ## Check input parameters for errors
   if (ftol <= 0) + (xtol <= 0) + (gtol <= 0) + (maxiter <= 0) + (factor <= 0):
      print 'ERROR: invalid value for one of the input keywords'
      return res
   ## Parameter damping doesn't work when user is providing their own
   ## gradients.
   if (damp != 0) and (autoderivative == 0):
      print \
          'ERROR: keywords DAMP and AUTODERIVATIVE are mutually exclusive'
      return res

   ## Fixed parameters?
   ifixed = numpy.zeros(npar, bool)
   if (fixed != None):
      if type(fixed) != numpy.ndarray: fixed = numpy.array(xall)
      if len(fixed) != npar:
         print 'ERROR: number of elements in FIXED and P must agree'
         return res
      ifixed = (fixed != 0)
   qfixed = ifixed.any()
   ## Tied parameters?
   itied = numpy.zeros(npar, bool)
   if (tied != None):
      if type(tied) != numpy.ndarray: tied = numpy.array(tied)
      if len(tied) != npar:
         print 'ERROR: number of elements in TIED and P must agree'
         return res
      if type(tied[0] != str):
         print 'ERROR: TIED must be a numpy string array'
         return res
      for i in range(npar):
         tied[i] = tied[i].strip()
      itied = (tied != '')
   qtied = itied.any()
   # Tied parameters are effectively fixed
   ifree = (ifixed == False) * (itied == False) 
   n = numpy.sum(ifree)
   if n == 0:
      print 'ERROR: no free parameters'
      return res
   x = xall[ifree]

   ## Limited parameters?
   ilimited = numpy.zeros((2,npar), bool)
   if (limited == None):
      if type(limited) != numpy.ndarray: limited = numpy.array(limited)
      if limits != None:
         print 'WARNING: ignoring LIMITS as LIMITED not supplied'
   else:
      if limits == None:
         print 'ERROR: must supply LIMITS if supplying LIMITED'
         return res
      if type(limits) != numpy.ndarray: limits = numpy.array(limits)
      #brackets needed
      if (numpy.shape(limited) != (2,npar)) + (numpy.shape(limits) != (2,npar)):
         print 'ERROR: LIMITED and LIMITS must be 2xN arrays ' + \
             '(N = no. parameters)'
         return res
      ilimited = (limited != 0)
   ## Extract valyes for free params only
   iulim = numpy.zeros(n, bool)
   ulim  = numpy.zeros(n)
   illim = numpy.zeros(n, bool)
   llim  = numpy.zeros(n)
   if ilimited.any():
      iulim = limited[1,ifree] != 0
      ulim  = limits[1,ifree]
      illim = limited[0,ifree] != 0
      llim  = limits[0,ifree]

   qlim = iulim.any() + illim.any()
   ## Check consistency of limits
   if (qlim == True):
#      print x
#      print illim
#      print llim
#      print (x < llim)
#      print (illim == True) * (x < llim)
#      print iulim
#      print ulim
#      print (x > ulim)
#      print (illim == True)
#      print (iulim == True) * (x > ulim)
      wh = ((iulim == True) * (x > ulim)) + ((illim == True) * (x < llim))
#      print wh
      if wh.any():
         print 'ERROR: input params. not within LIMITS'
         return res
      #added brackets
      #wh =  (iulim == True) * (illim == True) + llim >= ulim
      wh =  ((iulim == True) * (illim == True)) * (llim >= ulim)
      #added
      #print (iulim == True)
      #print (illim == True)
      #print llim >= ulim
      #wh = numpy.array([False,False])
      #print wh
      #
      if wh.any():
         print 'ERROR: inconsistent upper and lower LIMITS'
         return res

   ## Parameter names for printing
   if (names != None):
      if type(names) != numpy.ndarray: names = numpy.array(names)
      if len(names) != npar:
         print 'ERROR: number of elements in NAMES and P must agree'
         return res

   ## Finite differencing step, absolute and relative, and sidedness
   ## of derivative
#   step = parinfo(parinfo, 'step', default=0., n=npar)
   step = None
#   dstep = parinfo(parinfo, 'relstep', default=0., n=npar)
   dstep = None
#   dside = parinfo(parinfo, 'mpside',  default=0, n=npar)
   dside = None

   ## Maximum and minimum steps allowed to be taken in one iteration
#   maxstep = parinfo(parinfo, 'mpmaxstep', default=0., n=npar)
#   minstep = parinfo(parinfo, 'mpminstep', default=0., n=npar)
#   qmin = minstep * 0  ## Remove minstep for now!!
#   qmax = maxstep != 0
#   wh = (((qmin!=0.) & (qmax!=0.)) & (maxstep < minstep))
#   if (numpy.sum(wh) > 0):
#      print 'ERROR: MPMINSTEP is greater than MPMAXSTEP'
#      return res
#   wh = ((qmin!=0.) & (qmax!=0.))
   qminmax = 0

   ## Check rescaling parameters
   if (rescale != 0):
      print 'ERROR: DIAG parameter scales are inconsistent'
      if( len(diag) < n): return res
      wh = (diag <= 0)
      if (numpy.sum(wh) > 0): return res
      print ''

   ## Inital call to function
   res.params = xall.copy()
   [res.status, fvec] = call(fcn, res.params, functkw, damp = damp, tied = tied)
   if (res.status < 0):
      print 'ERROR: first call to "' + str(fcn) + '" failed'
      return res
   m = len(fvec)
   if (m < n):
      print 'ERROR: no. parameters must not exceed no. data points'
      return res
   fnorm = enorm(fvec)

   ## Initialize Levelberg-Marquardt parameter and iteration counter
   par = 0.
   res.niter = 1
   qtf = x * 0.
   res.status = 0

   ## Beginning of the outer loop
   while(1):

      ## If requested, call fcn to enable printing of iterates
      x = res.params[ifree]
      if (qtied): res.params = tie(res.params, tied)
      if (nprint > 0) and (iterfunct != None):
         if (((res.niter-1) % nprint) == 0):
            mperr = 0
            xnew0 = res.params.copy()
            dof = max(m - n, 0)
            res.status = iterfunct(fcn, res.params, res.niter, fnorm**2, \
                                  functkw = functkw, ifree = ifree, \
                                  names = names, dof = dof, **iterkw)
                               
            ## Check for user termination
            if (res.status < 0):  
               print 'WARNING: premature termination by ' + str(iterfunct)
               return res

            ## If parameters were changed (grrr..) then re-tie
            if (max(abs(xnew0-res.params)) > 0):
               if (qtied): res.params = tie(res.params, tied)
               x = res.params[ifree]

      ## Calculate the jacobian matrix
      res.status = 2
      fjac = fdjac2(fcn, x, fvec, step, iulim, ulim, dside, \
                       epsfcn = epsfcn, autoderivative = autoderivative, \
                       dstep = dstep, functkw = functkw, ifree = ifree, \
                       xall = res.params, damp = damp, tied = tied)
      if (fjac == None):
         print 'WARNING: premature termination by FDJAC2'
         return res

      ## Determine if any of the parameters are pegged at the limits
      if (qlim):
#         whlpeg = (illim + (x == llim))
         whlpeg = (illim * (x == llim))
         nlpeg = numpy.sum(whlpeg)
#         whupeg = (iulim + (x == ulim))
         whupeg = (iulim * (x == ulim))
         nupeg = numpy.sum(whupeg)
         ## See if any "pegged" values should keep their derivatives
         if (nlpeg > 0):
            ## Total derivative of sum wrt lower pegged parameters
            for i in range(nlpeg):
               sum = numpy.sum(fvec * fjac[:,whlpeg[i]])
               if (sum > 0): fjac[:,whlpeg[i]] = 0
         if (nupeg > 0):
            ## Total derivative of sum wrt upper pegged parameters
            for i in range(nupeg):
               sum = numpy.sum(fvec * fjac[:,whupeg[i]])
               if (sum < 0): fjac[:,whupeg[i]] = 0

      ## Compute the QR factorization of the jacobian
      [fjac, ipvt, wa1, wa2] = qrfac(fjac, pivot=1)

      ## On the first iteration if "diag" is unspecified, scale
      ## according to the norms of the columns of the initial jacobian
      if (res.niter == 1):
         if ((rescale == 0) or (len(diag) < n)):
            diag = wa2.copy()
            diag[diag==0] = 1.
      
         ## On the first iteration, calculate the norm of the scaled x
         ## and initialize the step bound delta 
         wa3 = diag * x
         xnorm = enorm(wa3)
         delta = factor * xnorm
         if (delta == 0.): delta = factor

      ## Form (q transpose)*fvec and store the first n components in qtf
      wa4 = fvec.copy()
      for j in range(n):
         lj = ipvt[j]
         temp3 = fjac[j,lj]
         if (temp3 != 0):
            fj = fjac[j:,lj]
            wj = wa4[j:]
            ## *** optimization wa4(j:*)
            wa4[j:] = wj - fj * numpy.sum(fj*wj) / temp3  
         fjac[j,lj] = wa1[j]
         qtf[j] = wa4[j]
      ## From this point on, only the square matrix, consisting of the
      ## triangle of R, is needed.
      fjac = fjac[:n,:n]
      fjac.shape = [n,n]
      temp = fjac.copy()
      for i in range(n):
         temp[:,i] = fjac[:,ipvt[i]]
      fjac = temp.copy()

      ## Check for overflow. This should be a cheap test here since
      ## FJAC has been reduced to a (small) square matrix, and the
      ## test is O(N^2).
#      wh = where(finite(fjac) EQ 0, ct)
#      if ct GT 0 then goto, FAIL_OVERFLOW

      ## Compute the norm of the scaled gradient
      gnorm = 0.
      if (fnorm != 0):
         for j in range(n):
            l = ipvt[j]
            if (wa2[l] != 0):
               sum = numpy.sum(fjac[0:j+1,j]*qtf[0:j+1])/fnorm
               gnorm = max([gnorm,abs(sum/wa2[l])])
               
      ## Test for convergence of the gradient norm
      if (gnorm <= gtol):
         res.status = 4
         break

      ## Rescale if necessary
      if (rescale == 0): diag = numpy.choose(diag>wa2, (wa2, diag))

      ## Beginning of the inner loop
      while(1):
         ## Determine the levenberg-marquardt parameter
         [fjac, par, wa1, wa2] = lmpar(fjac, ipvt, diag, qtf,
                                       delta, wa1, wa2, par = par)
         ## Store the direction p and x+p. Calculate the norm of p
         wa1 = -wa1

         if (qlim == 0) and (qminmax == 0):
            ## No parameter limits, so just move to new position WA2
            alpha = 1.
            wa2 = x + wa1
         else:

            ## Respect the limits. If a step were to go out of bounds,
            ## then we should take a step in the same direction but
            ## shorter distance. The step should take us right to the
            ## limit in that case.
            alpha = 1.
            if (qlim):
               ## Do not allow any steps out of bounds
               if (nlpeg > 0):
                  numpy.put(wa1, whlpeg, numpy.clip(
                        numpy.take(wa1, whlpeg), 0., max(wa1)))
               if (nupeg > 0):
                  numpy.put(wa1, whupeg, numpy.clip(
                        numpy.take(wa1, whupeg), min(wa1), 0.))

               dwa1 = abs(wa1) > machep
               whl = (dwa1 != 0.) * illim * ((x + wa1) < llim)
               if (numpy.sum(whl) > 0):
                  t = (llim[whl] - x[whl]) / wa1[whl]
                  alpha = min(alpha, min(t))
               whu = (dwa1 != 0.) * iulim * ((x + wa1) > ulim)
               if (numpy.sum(whu) > 0):
                  t = (ulim[whu] - x[whu]) / wa1[whu]
                  alpha = min(alpha, min(t))

            ## Obey any max step values.
            if (qminmax):
               nwa1 = wa1 * alpha
               whmax = ((qmax != 0.) * (maxstep > 0))
               if (numpy.sum(whmax) > 0):
                  mrat = max(nwa1[whmax] / maxstep[whmax])
                  if (mrat > 1): alpha = alpha / mrat

            ## Scale the resulting vector
            wa1 = wa1 * alpha
            wa2 = x + wa1

            ## Adjust the final output values. If the step put us
            ## exactly on a boundary, make sure it is exact.
            wh = (iulim != 0.) * (wa2 >= ulim * (1 - machep))
            if (numpy.sum(wh) > 0): wa2[wh] = ulim[wh]
            wh = (illim != 0.) * (wa2 <= llim * (1 + machep))
            if (numpy.sum(wh) > 0): wa2[wh] = llim[wh]
         # endelse
 
         wa3 = diag * wa1
         pnorm = enorm(wa3)

         ## On the first iteration, adjust the initial step bound
         if (res.niter == 1): delta = min([delta, pnorm])

         res.params[ifree] = wa2
 
         ## Evaluate the function at x+p and calculate its norm
         mperr = 0
         [res.status, wa4] = call(fcn, res.params, functkw, damp = damp, \
                                 tied = tied)
         if (res.status < 0):
            print 'WARNING: premature termination by "' + fcn + '"'
            return res
         fnorm1 = enorm(wa4)
  
         ## Compute the scaled actual reduction
         actred = -1.
         if ((0.1 * fnorm1) < fnorm): actred = - (fnorm1/fnorm)**2 + 1.

         ## Compute the scaled predicted reduction and the scaled
         ## directional derivative
         for j in range(n):
            wa3[j] = 0
            wa3[0:j+1] = wa3[0:j+1] + fjac[0:j+1,j] * wa1[ipvt[j]]

         ## Remember, alpha is the fraction of the full LM step
         ## actually taken
         temp1 = enorm(alpha * wa3) / fnorm
         temp2 = (numpy.sqrt(alpha * par) * pnorm) / fnorm
         prered = temp1 * temp1 + (temp2 * temp2) / 0.5
         dirder = -(temp1 * temp1 + temp2 * temp2)

         ## Compute the ratio of the actual to the predicted reduction.
         ratio = 0.
         if (prered != 0): ratio = actred / prered

         ## Update the step bound
         if (ratio <= 0.25):
            if (actred >= 0): temp = .5
            else: temp = .5 * dirder / (dirder + .5 * actred)
            if ((0.1 * fnorm1) >= fnorm) or (temp < 0.1): temp = 0.1
            delta = temp * min([delta, pnorm / 0.1])
            par = par / temp
         else: 
            if (par == 0) or (ratio >= 0.75):
               delta = pnorm / .5
               par = .5 * par

         ## Test for successful iteration
         if (ratio >= 0.0001): 
            ## Successful iteration.  Update x, fvec, and their norms
            x = wa2
            wa2 = diag * x
            fvec = wa4
            xnorm = enorm(wa2)
            fnorm = fnorm1
            res.niter = res.niter + 1
 
         ## Tests for convergence
         if ((abs(actred) <= ftol) * (prered <= ftol) \
                * (0.5 * ratio <= 1)): res.status = 1
         if delta <= xtol * xnorm: res.status = 2
         if ((abs(actred) <= ftol) * (prered <= ftol) \
                * (0.5 * ratio <= 1) * (res.status == 2)): res.status = 3
         if (res.status != 0): break

         ## Tests for termination and stringent tolerances
         if (res.niter >= maxiter): res.status = 5
         if ((abs(actred) <= machep) * (prered <= machep) \
                * (0.5*ratio <= 1)): res.status = 6
         if delta <= machep * xnorm: res.status = 7
         if gnorm <= machep: res.status = 8
         if (res.status != 0): break

         ## Repeat if iteration unsuccessful
         if (ratio >= 0.0001): break

      ## Check for over/underflow - SKIP FOR NOW
#      wh = where(finite(wa1) EQ 0 OR finite(wa2) EQ 0 OR finite(x) EQ 0, ct)
#      if ct GT 0 OR finite(ratio) EQ 0 then begin
#         print ('ERROR: parameter or function value(s) have become '+$
#                   'infinite# check model function for over- '+$
#                   'and underflow')
#         res.status = -16
#         break

      ## End of inner loop.

      if (res.status != 0): break;

   ## End of outer loop.

   ## Termination, either normal or user imposed.
   if (len(res.params) == 0): 
      print 'WARNING: len(params)=0'
      return res
   if (n == 0): res.params = xall.copy()
   else: res.params[ifree] = x
   if (res.status > 0):
      [status, fvec] = call(fcn, res.params, functkw, damp = damp, \
                               tied = tied)
      fnorm = enorm(fvec)
      m = len(fvec)
      dof = max(m - n, 0)
      res.dof = dof

   if ((fnorm != None) * (fnorm1 != None)):
      fnorm = max(fnorm, fnorm1)
      fnorm = fnorm**2.
      res.bestnorm = fnorm   
   
   res.covar = None
   res.perror = None
   ## (very carefully) set the covariance matrix COVAR
   if ((res.status > 0) * (nocovar == 0) * (n != None) \
          * (fjac != None) * (ipvt != None)):
      sz = numpy.shape(fjac)
      if ((n > 0) * (sz[0] >= n) * (sz[1] >= n) \
             * (len(ipvt) >= n)):
         cv = calc_covar(fjac[:n,:n], ipvt[:n])
         cv.shape = [n,n]
          
         ## Fill in actual covariance matrix, accounting for fixed
         ## parameters.
         nn = len(xall)
         res.covar = numpy.zeros((nn,nn))

         l = numpy.where(ifree)
         l = l[0]
         l = numpy.where(ifree)
         l = l[0]
         for i in range(n):
            res.covar[l, l[i]] = cv[:,i]

         ## Compute errors in parameters
         res.perror = numpy.zeros(nn)
         res.pcerror = numpy.copy(res.perror)
         d = numpy.diagonal(res.covar)
         wh = (d >= 0)
         res.perror[wh] = numpy.sqrt(d[wh])
         if (res.bestnorm != None) * (res.dof != None):
            res.pcerror = res.perror * numpy.sqrt(res.bestnorm/res.dof)
   return res


def tie(p, ptied = None):
   """Tie one parameter to another."""
   if (ptied == None): return p
   for i in range(len(ptied)):
      if ptied[i] == '': continue
      cmd = 'p[' + str(i) + '] = ' + ptied[i]
      exec(cmd)
   return p

def call(fcn, x, functkw, fjac = None, tied = None, damp = 0):
   """Call user function"""
   if tied != None:
      qtied = sum(tied != '')
      if (qtied): x = tie(x, tied)
   [status, f, pder] = fcn(x, fjac = fjac, **functkw)
   if (fjac == None):
      if (damp > 0):
         ## Apply the damping if requested.  This replaces the
         ## residuals with their hyperbolic tangent.  Thus residuals
         ## larger than DAMP are essentially clipped.
         f = numpy.tanh(f/damp)
      return [status, f]
   else:
      return [status, f, pder]

def fdjac2(fcn, x, fvec, step = None, ulimited = None, ulimit = None, \
              dside = None, epsfcn = None, autoderivative = 1, \
              functkw = None, xall = None, ifree = None, dstep = None, \
              damp = None, tied = None):
   """Compute the Jacobean"""
   machep = MACH.machep
   if epsfcn == None: epsfcn = machep
   if xall == None: xall = x
   nall = len(xall)
   if ifree == None: ifree = scipy.ones(nall, bool)
   eps = numpy.sqrt(max(epsfcn, machep))
   m = len(fvec)
   n = len(x)
   ## Fetch analytical derivative if requested
   if (autoderivative == 0):
      mperr = 0
      fjac = numpy.zeros(nall, float)
      ## Specify which parameters need derivatives
      fjac[ifree] = 1
      [status, fp, pder] = call(fcn, xall, functkw, fjac = fjac, \
                             damp = damp, tied = tied)
      if numpy.prod(pder.shape) != m * nall:
         print 'ERROR: Derivative matrix was not computed properly.'
         return None
      pder.shape = [m,nall]
      pder = -pder
      ## Select only the free parameters
      if len(ifree) < nall:
         pder = pder[:,ifree]
         pder.shape = [m, n]
         return(pder)
   ## Else compute derivative
   fjac = numpy.zeros([m, n], float)
   h = eps * abs(x)
   ## if STEP is given, use that
   if step != None:
      stepi = step[ifree]
      wh = (stepi > 0)
      h[wh] = stepi[wh]
   ## if relative step is given, use that
   if dstep != None:
      dstepi = dstep[ifree]
      wh = (dstepi > 0)
      h[wh] = abs(dstepi[wh])*x[wh]
   ## In case any of the step values are zero
   h[(h == 0)] = eps
   ## Reverse the sign of the step if we are up against the parameter
   ## limit, or if the user requested it.
   if (ulimited != None) + (ulimit != None):
      wh = (dside == -1) + ulimited * (x > ulimit-h)
      h[wh] = -h[wh]
   ## Loop through parameters, computing the derivative for each
   l = numpy.where(ifree)
   l = l[0]
   for j in range(n):
      xp = xall.copy()
      fp = fvec.copy()
      xp[l[j]] += h[j]
      [status, fp] = call(fcn, xp, functkw, damp = damp, tied = tied)
      if (status < 0): return None
      if dside == None:
         ## COMPUTE THE ONE-SIDED DERIVATIVE
         fjac[:,j] = (fp-fvec) / h[j]
      else:
         if abs(dside[j]) <= 1:
            ## COMPUTE THE ONE-SIDED DERIVATIVE
            fjac[:,j] = (fp-fvec) / h[j]
         else:
            ## COMPUTE THE TWO-SIDED DERIVATIVE
            xp[ifree[j]] -= h[j]
            mperr = 0
            [status, fm] = call(fcn, xp, functkw, damp = damp, tied = tied)
            if (status < 0): return None
            fjac[:,j] = (fp-fm) / (2 * h[j])
   return fjac

def enorm(vec, fastnorm = False):
   """Compute norm of a vector"""
   ## NOTE: it turns out that, for systems that have a lot of data
   ## points, this routine is a big computing bottleneck.  The
   ## extended computations that need to be done cannot be effectively
   ## vectorized.  The introduction of the FASTNORM configuration
   ## parameter allows the user to select a faster routine, which is
   ## based on numpy.sum() alone.  Very simple-minded sum-of-squares
   if fastnorm:
      ans = numpy.sqrt(numpy.sum(vec*vec))
   else:
      agiant = MACH.rgiant / len(vec)
      adwarf = MACH.rdwarf * len(vec)
      ## This is hopefully a compromise between speed and
      ## robustness. Need to do this because of the possibility of
      ## over- or underflow.
      mx = max(abs(max(vec)), abs(min(vec)))
      if mx == 0: return vec[0]*0.
      if (mx > agiant) * (mx < adwarf):
         ans = mx * numpy.sqrt(numpy.sum((vec/mx)*(vec/mx)))
      else:
         ans = numpy.sqrt(numpy.sum(vec*vec))
   return ans

def qrfac(a, pivot = 0):
   """Docstring TBD"""
   machep = MACH.machep
   m, n = a.shape
   ## Compute the initial column norms and initialize arrays
   acnorm = numpy.zeros(n)
   for j in range(n): acnorm[j] = enorm(a[:,j])
   rdiag = acnorm.copy()
   wa = rdiag.copy()
   ipvt = numpy.arange(n)
   ## Reduce a to r with householder transformations
   minmn = min(m, n)
   for j in range(minmn):
      if (pivot != 0):
         ## Bring the column of largest norm into the pivot position
         rmax = max(rdiag[j:])
         kmax = numpy.where(rdiag[j:] == rmax)
         kmax = kmax[0]
         ct = len(kmax)
         kmax += j
         if ct > 0:
            kmax = kmax[0]
            ## Exchange rows via the pivot only.  Avoid actually
            ## exchanging the rows, in case there is lots of memory
            ## transfer.  The exchange occurs later, within the body
            ## of MPFIT, after the extraneous columns of the matrix
            ## have been shed.
            if kmax != j:
               temp = ipvt[j] ; ipvt[j] = ipvt[kmax] ; ipvt[kmax] = temp
               rdiag[kmax] = rdiag[j]
               wa[kmax] = wa[j]
      ## Compute the householder transformation to reduce the jth
      ## column of A to a multiple of the jth unit vector
      lj = ipvt[j]
      ajj = a[j:,lj]
      ajnorm = enorm(ajj)
      if ajnorm == 0: break
      if a[j,j] < 0: ajnorm = -ajnorm         
      ajj = ajj / ajnorm
      ajj[0] += 1
      a[j:,lj] = ajj
      ## Apply the transformation to the remaining columns and update
      ## the norms
      ## NB: tried to optimize this by removing the loop, but it
      ## actually got slower.  Reverted to "for" loop to keep it
      ## simple.
      if (j+1 < n):
         for k in range(j+1, n):
            lk = ipvt[k]
            ajk = a[j:,lk]
            if a[j,lj] != 0: 
               a[j:,lk] = ajk - ajj * numpy.sum(ajk*ajj)/a[j,lj]
               if ((pivot != 0) + (rdiag[k] != 0)):
                  temp = a[j,lk] / rdiag[k]
                  rdiag[k] = rdiag[k] * numpy.sqrt(max((1.-temp**2), 0.))
                  temp = rdiag[k] / wa[k]
                  if ((0.05 * temp * temp) <= machep):
                     rdiag[k] = enorm(a[j+1:,lk])
                     wa[k] = rdiag[k]
      rdiag[j] = -ajnorm
   return [a, ipvt, rdiag, acnorm]

def qrsolv(r, ipvt, diag, qtb, sdiag):
   """Docstring TBD"""
   m, n = numpy.shape(r)
   ## Copy r and (q transpose)*b to preserve input and initialize s.
   ## in particular, save the diagonal elements of r in x.
   for j in range(n):
      r[j:n,j] = r[j,j:n]
   x = r.diagonal()
   wa = qtb.copy()
   ## Eliminate the diagonal matrix d using a givens rotation
   for j in range(n):
      l = ipvt[j]
      if (diag[l] == 0): break
      sdiag[j:] = 0
      sdiag[j] = diag[l]
      ## The transformations to eliminate the row of d modify only a
      ## single element of (q transpose)*b beyond the first n, which
      ## is initially zero.
      qtbpj = 0.
      for k in range(j,n):
         if (sdiag[k] == 0): break
         if (abs(r[k,k]) < abs(sdiag[k])):
            cotan = r[k,k] / sdiag[k]
            sine = 0.5 / numpy.sqrt(.25 + .25*cotan*cotan)
            cosine = sine * cotan
         else:
            tang = sdiag[k] / r[k,k]
            cosine = 0.5 / numpy.sqrt(.25 + .25*tang*tang)
            sine = cosine * tang 
         ## Compute the modified diagonal element of r and the
         ## modified element of ((q transpose)*b,0).
         r[k,k] = cosine*r[k,k] + sine*sdiag[k]
         temp = cosine*wa[k] + sine*qtbpj
         qtbpj = -sine*wa[k] + cosine*qtbpj
         wa[k] = temp
         ## Accumulate the transformation in the row of s
         if (n > k+1):
            temp = cosine * r[k+1:n,k] + sine * sdiag[k+1:n]
            sdiag[k+1:n] = - sine * r[k+1:n,k] + cosine * sdiag[k+1:n]
            r[k+1:n,k] = temp
      sdiag[j] = r[j,j]
      r[j,j] = x[j]
   ## Solve the triangular system for z. If the system is singular
   ## then obtain a least squares solution
   nsing = n
   wh = (sdiag == 0)
   if wh.any():
      nsing = wh[0]
      wa[nsing:] = 0
   if (nsing >= 1):
      wa[nsing-1] = wa[nsing-1]/sdiag[nsing-1] ## Degenerate case
      ## *** Reverse loop ***
      for j in range(nsing-2,-1,-1):  
         sum = numpy.sum(r[j+1:nsing,j]*wa[j+1:nsing])
         wa[j] = (wa[j]-sum) / sdiag[j]
   ## Permute the components of z back to components of x
   x[ipvt] = wa
   return (r, x, sdiag)

def lmpar(r, ipvt, diag, qtb, delta, x, sdiag, par = None):
   """Compute LM parameter"""
   dwarf = MACH.minnum
   m, n = numpy.shape(r)
   ## Compute and store in x the gauss-newton direction.  If the
   ## jacobian is rank-deficient, obtain a least-squares solution
   nsing = n
   wa1 = qtb.copy()
   wh = (r.diagonal() == 0)
   if wh.any():
      nsing = wh[0]
      wa1[nsing:] = 0
   if nsing > 1:
      ## *** Reverse loop ***
      for j in range(nsing-1,-1,-1):  
         wa1[j] = wa1[j] / r[j,j]
         if (j-1 >= 0):
            wa1[0:j] = wa1[0:j] - r[0:j,j] * wa1[j]
   ## Note: ipvt here is a permutation array
   x[ipvt] = wa1
   ## Initialize the iteration counter.  Evaluate the function at the
   ## origin, and test for acceptance of the Gauss-Newton direction
   iter = 0
   wa2 = diag * x
   dxnorm = enorm(wa2)
   fp = dxnorm - delta
   if (fp <= 0.1*delta):
      return [r, 0., x, sdiag]
   ## If the jacobian is not rank deficient, the Newton step provides a
   ## lower bound, parl, for the zero of the function.  Otherwise set
   ## this bound to zero.
   parl = 0.
   if nsing >= n:
      wa1 = diag[ipvt] * wa2[ipvt] / dxnorm
      wa1[0] /= r[0,0] ## Degenerate case 
      for j in range(1,n): ## Note "1" here, not zero
         sum = numpy.sum(r[0:j,j] * wa1[0:j])
         wa1[j] = (wa1[j] - sum) / r[j,j]
      temp = enorm(wa1)
      parl = ((fp / delta) / temp) / temp
   ## Calculate an upper bound, paru, for the zero of the function
   for j in range(n):
      sum = numpy.sum(r[0:j+1,j] * qtb[0:j+1])
      wa1[j] = sum / diag[ipvt[j]]
   gnorm = enorm(wa1)
   paru = gnorm / delta
   if paru == 0: paru = dwarf / min(delta,0.1)
   ## If the input par lies outside of the interval (parl,paru), set
   ## par to the closer endpoint
   par = min(max(par, parl), paru)
   if par == 0: par = gnorm / dxnorm
   ## Beginning of an interation
   while(1):
      iter += 1
      ## Evaluate the function at the current value of par
      if par == 0: par = max(dwarf, paru * 0.001)
      temp = numpy.sqrt(par)
      wa1 = temp * diag
      [r, x, sdiag] = qrsolv(r, ipvt, wa1, qtb, sdiag)
      wa2 = diag * x
      dxnorm = enorm(wa2)
      temp = fp
      fp = dxnorm - delta
      if ((abs(fp) <= 0.1 * delta) + \
             ((parl == 0) * (fp <= temp) * (temp < 0)) + \
             (iter == 10)): break;
      ## Compute the newton correction
      wa1 = diag[ipvt] * wa2[ipvt] / dxnorm
      for j in range(n-1):
         wa1[j] = wa1[j] / sdiag[j]
         wa1[j+1:n] = wa1[j+1:n] - r[j+1:n,j] * wa1[j]
      wa1[n-1] = wa1[n-1] / sdiag[n-1] ## Degenerate case
      temp = enorm(wa1)
      parc = fp / delta / temp / temp
      ## Depending on the sign of the function, update parl or paru
      if fp > 0: parl = max(parl, par)
      if fp < 0: paru = min(paru, par)
      ## Compute an improved estimate for par
      par = max(parl, par + parc)
   ## End of an iteration
   ## Termination
   return [r, par, x, sdiag]

def calc_covar(rr, ipvt=None, tol=1.e-14):
   """Compute covariance matrix"""
   if numpy.rank(rr) != 2:
      print 'ERROR: r must be a two-dimensional matrix'
      return -1
   s = numpy.shape(rr)
   n = s[0]
   if s[0] != s[1]:
      print 'ERROR: r must be a square matrix'
      return -1
   if (ipvt == None): ipvt = numpy.arange(n)
   r = rr.copy()
   r.shape = [n,n]
   ## For the inverse of r in the full upper triangle of r
   l = -1
   tolr = tol * abs(r[0,0])
   for k in range(n):
      if (abs(r[k,k]) <= tolr): break
      r[k,k] = 1./r[k,k]
      for j in range(k):
         temp = r[k,k] * r[j,k]
         r[j,k] = 0.
         r[0:j+1,k] = r[0:j+1,k] - temp*r[0:j+1,j]
      l = k
   ## Form the full upper triangle of the inverse of (r transpose)*r
   ## in the full upper triangle of r
   if l >= 0:
      for k in range(l+1):
         for j in range(k):
            temp = r[j,k]
            r[0:j+1,j] = r[0:j+1,j] + temp*r[0:j+1,k]
         temp = r[k,k]
         r[0:k+1,k] = temp * r[0:k+1,k]
   ## For the full lower triangle of the covariance matrix
   ## in the strict lower triangle or and in wa
   wa = numpy.repeat([r[0,0]], n)
   for j in range(n):
      jj = ipvt[j]
      sing = j > l
      for i in range(j+1):
         if sing: r[i,j] = 0.
         ii = ipvt[i]
         if ii > jj: r[ii,jj] = r[i,j]
         if ii < jj: r[jj,ii] = r[i,j]
      wa[jj] = r[j,j]
   ## Symmetrize the covariance matrix in r
   for j in range(n):
      r[0:j+1,j] = r[j,0:j+1]
      r[j,j] = wa[j]
   return r

