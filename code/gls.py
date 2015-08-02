import scipy
import scipy.linalg
import pylab
import mpfit
#from planet import orbit

mach = scipy.MachAr()
small = 10 * mach.eps

def sinefit(time, data, err = None, pmin = None, pmax = None, \
                nper = 500, return_periodogram = False, doplot = False):
    """Sine curve fit. Period by brute force, other pars linear.
    per, amp, phase, dc = sinefit(x, y, [err, pmin, pmax, nper])"""
    npts = len(time)
    if pmin is None:
        w = scipy.sort(time)
        dt = w[1:] - w[:npts-1]
        tstep = scipy.median(dt)
        pmin = 2 * tstep
    if pmax is None: pmax = (time.max() - time.min()) / 2.
    # even sampling in log space
    lpmin,lpmax = scipy.log10([pmin,pmax])
    lpers = scipy.r_[lpmin:lpmax:nper*1j]
    pers = 10.0**lpers
    #print 'pmin vs pnewmin', pmin, min(pers)
    #print 'pmax vs pnewmax', pmax, max(pers)

    if err == None:
        z = scipy.ones(npts)
        err = scipy.zeros(npts)
        mrk = '.'
    else:
        if len(err) is len(data):
            z = 1.0 / err**2
            mrk = '.'
        else:
            z = scipy.ones(npts)
            err = scipy.zeros(npts)
            mrk = '.'
    sumwt = z.sum()
    chi2_0 = scipy.sum((data-scipy.mean(data))**2*z)
    p_w = scipy.zeros(nper)
    a_w = scipy.zeros(nper)
    p_max = 0.0

    for i in scipy.arange(nper):
        arg = 2 * scipy.pi * time / pers[i]
        cosarg = scipy.cos(arg)
        sinarg = scipy.sin(arg)
        a = scipy.matrix([[scipy.sum(sinarg**2*z), scipy.sum(cosarg*sinarg*z), \
                               scipy.sum(sinarg*z)], \
                              [0, scipy.sum(cosarg**2*z), scipy.sum(cosarg*z)], \
                              [0, 0, sumwt]])
        a[1,0] = a[0,1]
        a[2,0] = a[0,2]
        a[2,1] = a[1,2]
        a[abs(a) < small] = 0.
        if scipy.linalg.det(a) < small: continue
        b = [scipy.sum(data*sinarg*z), scipy.sum(data*cosarg*z), \
                 scipy.sum(data*z)]
        c = scipy.linalg.solve(a,b)
        amp = (c[0]**2+c[1]**2)**(0.5)
        a_w[i] = amp
        phase = scipy.arctan2(c[1],c[0])
        dc = c[2]
        fit = amp * scipy.sin(arg + phase) + dc
        p_w[i] = (chi2_0 - scipy.sum((data-fit)**2 * z)) / chi2_0
        oper, oamp, ophase, odc = 0, 0, 0, 0
        if p_w[i] > p_max:
            p_max = p_w[i]
            oper = pers[i]
            oamp = amp
            ophase = phase
            odc = dc
            ofit = fit

    if doplot == False:
        if return_periodogram == False: return oper, oamp, ophase, odc
        else: return oper, oamp, ophase, odc, pers, p_w, a_w, chi2_0

    pylab.close('all')
    pylab.figure(1, figsize = (6,7), edgecolor = 'w')
    pylab.subplot(311)
    pylab.errorbar(time, data, err, fmt = 'k' + mrk, capsize = 0)
    pylab.xlabel('x')
    pylab.ylabel('y')
    np = (time.max()-time.min()) / oper
    if np < 20:
        x = scipy.r_[time.min():time.max():101j]
        pylab.plot(x, oamp * scipy.sin(2 * scipy.pi * x / oper + ophase) + odc, 'r')
    pylab.xlim(time.min(), time.max())
    pylab.subplot(312)
    pylab.loglog()
    pylab.axvline(oper, c = 'r')
    pylab.plot(pers, p_w, 'k-')
    pylab.xlabel('period')
    pylab.ylabel('reduced chi2')
    pylab.xlim(pers.min(), pers.max())
    pylab.subplot(313)
    ph = (time % oper) / oper
    pylab.errorbar(ph, data, err, fmt = 'k' + mrk, capsize = 0)
    x = scipy.r_[0:oper:101j]
    y = oamp * scipy.sin(2 * scipy.pi * x / oper + ophase) + odc
    pylab.plot(x/oper, y, 'r')
    pylab.xlim(0,1)
    pylab.xlabel('phase')
    pylab.ylabel('y')

    if return_periodogram == False: return oper, oamp, ophase, odc
    else: return oper, oamp, ophase, odc, pers, p_w, a_w, chi2_0

#def keplerian(t, par):
#    P, K, T0, V0, Ecc, omega = par
#    return V0 + orbit.radvel(t, P, K, T0, 0.0, Ecc, omega)

#def kep_func(p, x = None, y = None, err = None, fjac = None):
#    """Error function to minimize for Keplerian fitting"""
#    if err == None: return [0, keplerian(x, p) - y, None]
#    return [0, (keplerian(x, p) - y) / err, None]

#def kls(time, data, err = None, nph = 10, necc = 10, nper = 100, \
#            pmin = None, pmax = None, doplot = True, adjust = True):
#    '''Keplerian orbit fit based on GLS. Phase, eccentricity and
#    period by brute force, other parameters by means of normal
#    equations.'''
#
#    npts = len(time)
#    if len(err) is npts:
#        z = 1.0 / err**2
#    else:
#        z = scipy.ones(npts)
#        err = scipy.zeros(npts)
#    sumwt = z.sum()
#    chi2_0 = scipy.sum((data-scipy.mean(data))**2*z)
#    t_first = time.min()
#    t = time - t_first
#
#    if pmin is None:
#        w = scipy.sort(time)
#        dt = w[1:] - w[:npts-1]
#        tstep = scipy.median(dt)
#        pmin = 2 * tstep
#    if pmax is None: pmax = (time.max() - time.min()) / 2.
#    lpmin,lpmax = scipy.log10([pmin,pmax])
#    lpers = scipy.r_[lpmin:lpmax:nper*1j]
#    pers = 10.0**lpers
#    fmin = 1.0 / pmax
#    fmax = 1.0 / pmin
#    freqs = scipy.r_[fmin:fmax:nper*1j]
#    pers = 1.0 / freqs
#    phases = scipy.r_[0:1:(nph+1)*1j][:nph]
#    eccs = scipy.r_[0:1:(necc+1)*1j][:necc]
#
#    pylab.close('all')
#    pylab.figure(1)
#    chi2_per = scipy.zeros(nper)
#    chi2_min = chi2_0
#    for iper in scipy.arange(nper):
#        period = pers[iper]
#        chi2_2d = scipy.zeros((nph,necc))
#        chi2_ = chi2_0
#        for iph in scipy.arange(nph):
#            phase = phases[iph]
#            t0 = phase * period
#            for iecc in scipy.arange(necc):
#                ecc = eccs[iecc]
#                arg = orbit.truean(t, period, t0, ecc)
#                cosarg = scipy.cos(arg)
#                sinarg = scipy.sin(arg)
#                a = scipy.matrix([[scipy.sum(sinarg**2*z), scipy.sum(cosarg*sinarg*z), \
#                                       scipy.sum(sinarg*z)], \
#                                      [0, scipy.sum(cosarg**2*z), \
#                                           scipy.sum(cosarg*z)], \
#                                      [0, 0, sumwt]])
#                a[1,0] = a[0,1]
#                a[2,0] = a[0,2]
#                a[2,1] = a[1,2]
#                a[abs(a) < small] = 0.
#                if scipy.linalg.det(a) < small: continue
#                b = [scipy.sum(data*sinarg*z), scipy.sum(data*cosarg*z), \
#                         scipy.sum(data*z)]
#                c = scipy.linalg.solve(a,b)
#                fit = c[0] * sinarg + c[1] * cosarg + c[2]
#                chi2 = scipy.sum((data-fit)**2 * z)
#                if chi2 < chi2_:
#                    chi2_ = chi2
#                    K_ = scipy.sqrt(c[0]**2+c[1]**2)
#                    omega_ = scipy.arctan2(-c[0], c[1])
#                    gamma_ = c[2] - c[1] * ecc
#                    ecc_ = ecc
#                    ph_ = phase
#                chi2_2d[iph,iecc] = chi2
#        chi2_per[iper] = chi2_
#        if chi2_ < chi2_min:
#            chi2_min = scipy.copy(chi2_)
#            per_save = period
#            K_save = scipy.copy(K_)
#            omega_save = scipy.copy(omega_)
#            gamma_save = scipy.copy(gamma_)
#            ecc_save = scipy.copy(ecc_)
#            ph_save = scipy.copy(ph_)
#            chi2_2d_save = scipy.copy(chi2_2d)
#        print iper,nper
#
#    p_2d = (chi2_2d - chi2_0) / chi2_0
#    dchi2_per = (chi2_0 - chi2_per)
#    p_per = (chi2_0 - chi2_per) / chi2_0
#    p_min = p_per.min()
#    zre = (p_per / (1 - p_min))
#    z_per = zre * (npts - 5.) / 4.
#    pgt_per = (1 + (npts - 3.) * zre / 2.) * (1 + zre)**(-(npts-3.)/2.)
#
#    period = per_save
#    K = K_save
#    omega = omega_save
#    gamma = gamma_save
#    ecc = ecc_save
#    phase = ph_save
#    t0 = phase * period + t_first
#
#    if adjust == True:
#        pin = scipy.array([period, K, t0, gamma, ecc, omega])
#        print pin
#        fa = {'x': time, 'y': data, 'err': err}
#        m = mpfit.mpfit(kep_func, pin, functkw =  fa, quiet = True)
#        pout = m.params
#        print pout
#        period, K, t0, gamma, ecc, omega = pout
#        phase = (t0 - t_first) / period
#
#    if doplot == False: return period, t0, ecc, K, gamma, omega
#    fit = orbit.radvel(time, period, K, t0, gamma, ecc, omega)
#
#    xl1 = 0.1
#    xw1 = 0.85
#    yw1 = 0.55
#    yl1 = 1 - 0.05 - yw1
#    yw2 = 0.3
#    yl2 = yl1 - yw2
#
#    pylab.close('all')
#    pylab.figure(1)
#    ax1 = pylab.axes([xl1, yl1, xw1, yw1])
#    pylab.setp(ax1.xaxis.get_ticklabels(), visible = False)
#    pylab.errorbar(time, data, err, fmt = 'ko', capsize = 0)
#    npl = 1000
#    tmin = scipy.floor(t_first)
#    tmax = scipy.floor(time.max()) + 1
#    tpl = scipy.r_[tmin:tmax:npl*1j]
#    fpl = orbit.radvel(tpl, period, K, t0, gamma, ecc, omega)
#    pylab.plot(tpl, fpl, 'r-')
#    pylab.ylabel('RV')
#    ax2 = pylab.axes([xl1, yl2, xw1, yw2], sharex=ax1)
#    pylab.errorbar(time, data-fit, err, fmt = 'ko', capsize=0)
#    pylab.axhline(0.0, c = 'r')
#    pylab.ylabel('residuals')
#    pylab.xlabel('time')
#
#    pylab.figure(2)
#    ax1 = pylab.axes([xl1, yl1, xw1, yw1])
#    pylab.setp(ax1.xaxis.get_ticklabels(), visible = False)
#    phc = orbit.phase(time, period, t0)
#    pylab.errorbar(phc, data, err, fmt = 'ko', capsize = 0)
#    pylab.errorbar(phc+1, data, err, fmt = 'k.', capsize=0)
#    pylab.errorbar(phc-1, data, err, fmt = 'k.', capsize=0)
#    pylab.plot(phc+1, data, 'wo', mec = 'k')
#    pylab.plot(phc-1, data, 'wo', mec = 'k')
#    npl = 1000
#    tpl = scipy.r_[-0.5:1.5:npl*1j]
#    fpl = orbit.radvel(tpl, 1, K, 0.0, gamma, ecc, omega)
#    pylab.plot(tpl, fpl, 'r-')
#    pylab.ylabel('RV')
#    ax2 = pylab.axes([xl1, yl2, xw1, yw2], sharex=ax1)
#    pylab.errorbar(phc, data-fit, err, fmt = 'ko', capsize=0)
#    pylab.errorbar(phc+1, data-fit, err, fmt = 'k.', capsize=0)
#    pylab.errorbar(phc-1, data-fit, err, fmt = 'k.', capsize=0)
#    pylab.plot(phc+1, data-fit, 'wo', mec = 'k')
#    pylab.plot(phc-1, data-fit, 'wo', mec = 'k')
#    pylab.axhline(0.0, c = 'r')
#    pylab.xlim(-0.5,1.5)
#    pylab.ylabel('residuals')
#    pylab.xlabel('phase')
#
#    pylab.figure(3)
#    xoff = 0.15
#    xoffr = 0.03
#    xwi = 1 - xoff - xoffr
#    yoff = 0.1
#    yoffr = 0.03
#    ywi = (1 - yoff - yoffr) / 2.
#    ax1 = pylab.axes([xoff, 1-yoffr-ywi, xwi, ywi])
#    pylab.setp(ax1.xaxis.get_ticklabels(), visible = False)
#    pylab.axvline(period, c = 'r')
#    pylab.semilogx(pers, z_per, 'k-')
#    pylab.ylabel('normalised power z(P)')
#    axc = pylab.axes([xoff, 1-yoffr-2*ywi, xwi, ywi], sharex = ax1)
#    pylab.axvline(period, c = 'r')
#    pylab.loglog(pers, pgt_per, 'k-')
#    pylab.ylabel('false alarm prob. p(>z)')
#    pylab.xlabel('period')
#    pylab.xlim(pers.min(), pers.max())
#
#    pylab.figure(4)
#    p_2d = (chi2_0 - chi2_2d_save) / chi2_0
#    ph2d = scipy.resize(phases,(nph,necc))
#    dph = phases[1] - phases[0]
#    ph2d += dph / 2.
#    ecc2d = scipy.transpose(scipy.resize(eccs,(necc,nph)))
#    decc = eccs[1] - eccs[0]
#    ecc2d += decc / 2.
#    nlev = 10
#    im = pylab.imshow(p_2d, origin = 'lower', cmap = pylab.cm.get_cmap('gray'), \
#                          interpolation = 'nearest', extent = (0,1,0,1))
#    pylab.colorbar(im)
#    pylab.contour(ph2d, ecc2d, p_2d, 10, colors = 'chartreuse')
#    pylab.axvline(ecc, c='r')
#    pylab.axhline(phase, c='r')
#    pylab.xlabel('Eccentricity')
#    pylab.xlim(0,1)
#    pylab.ylabel('Phase')
#    pylab.ylim(0,1)
#    return period, t0, ecc, K, gamma, omega, data-fit
