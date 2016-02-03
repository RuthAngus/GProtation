import scipy, pyfits, atpy, glob, pylab, copy, os

import keplerdefs

#PlanetCands = atpy.Table('/Volumes/Data/kepler/Planet_Cands/Bahtalha12_PlanetCandidates.txt', type='ascii',name='PlanetCands')

BJDREF_t0 = 2454900
BJDREF_lc = 2454833

def GetLc(id = None, dir = None, tr_out = False, filename = None):
    if dir == None:
        print 'Must supply directory'
        return
    if id == None:
        print 'Must supply kepler id'
        return
    kic = scipy.copy(id)
    LcFiles = scipy.array([''])
    if filename != None:
        for fn in filename:
            LcFiles = scipy.append(LcFiles, glob.glob(fn)[0])
        LcFiles = LcFiles[1:]
        print LcFiles
    else: LcFiles = glob.glob('%s/kplr%09d-?????????????_llc.fits' % (dir,kic))
    nlc = len(LcFiles)
    if nlc == 0:
        print 'File not found for KIC %d' % kic
        if filename != None: 'Was looking for: %s' % filename
        else: print 'Was looking for: %s/kplr%09d-?????????????_llc.fits' % (dir,kic)
        return scipy.array([]), 0
    else: print 'Found %d long cadence lightcurve files for KIC %d' % (nlc, kic)
    tset = atpy.TableSet()
    if tr_out == True:
        tset.append(PlanetCands.where(scipy.floor(PlanetCands.KID) == kic))
    lc = atpy.Table(LcFiles[0], type = 'fits')
    hdulist = pyfits.open(LcFiles[0])
    quarter_index = scipy.zeros(len(lc.TIME), 'int') + hdulist[0].header['QUARTER']
    hdulist.close()
    for i in scipy.arange(nlc-1) + 1:
        temp_tab = atpy.Table(LcFiles[i], type = 'fits')
        temp_tab.PDCSAP_FLUX = temp_tab.PDCSAP_FLUX
        lc.append(temp_tab)
        hdulist = pyfits.open(LcFiles[i])
        new_index = scipy.zeros(len(lc.TIME)-len(quarter_index), 'int') + hdulist[0].header['QUARTER']
        hdulist.close()
        quarter_index = scipy.append(quarter_index, new_index)
    lc.add_column('Q', quarter_index)
    tset.append(lc)
    return tset, 1

def TransitPhase(tset):
    lc = tset.tables[1]
    time = lc.TIME
    nobs = len(time)
    lg = scipy.isfinite(time)
    pl = tset.tables[0]
    npl = len(pl.Period)
    phase = scipy.zeros((npl, nobs))
    inTr = scipy.zeros((npl, nobs), 'int')
    for ipl in scipy.arange(npl):
        period = pl.Period[ipl]
        t0 = pl.t0[ipl] + BJDREF_t0 - BJDREF_lc
        dur = pl.Dur[ipl] / 24. / period
        counter = 0
        while (time[lg] - t0).min() < 0:
            t0 -= period
            counter += 1
            if counter > 1000: break
        ph = ((time - t0) % period) / period
        ph[ph < -0.5] += 1
        ph[ph > 0.5] -= 1
        phase[ipl,:] = ph
        inTr[ipl,:] = (abs(ph) <= dur/1.5)
    return phase, inTr

def PlotLc(id=None, dir = None, quarter=None, tset = None):
    if id != None:
        tset, status = GetLc(id = id, dir = dir, tr_out = True)
        if tset is None: return
    elif tset == None:
        print 'no tset'
        return
    if quarter != None:
        tset.tables[1] = tset.tables[1].where(tset.tables[1].Q == quarter)
        if len(tset.tables[1].TIME) == 0:
            print 'No data for Q%d' % quarter
            return
    time = tset.tables[1].TIME
    phase, inTr = TransitPhase(tset)
    col = ['r','g','b','y','c','m','grey']
    npl, nobs = inTr.shape
    pylab.figure(1)
    pylab.clf()
    pylab.plot(time, tset.tables[1].PDCSAP_FLUX, 'k-')
    for ipl in scipy.arange(npl):
        list = inTr[ipl,:].astype(bool)
        pylab.plot(time[list], tset.tables[1].PDCSAP_FLUX[list], '.', c = col[ipl])
    l = scipy.isfinite(time)
    pylab.xlim(time[l].min(), time[l].max())
    ttl = 'KIC %d, P=' % (tset.tables[0].KID[0])
    for i in scipy.arange(npl):
        ttl = '%s%.5f ' % (ttl, tset.tables[0].Period[i])
    if quarter != None:
        ttl += 'Q%d' % quarter
    pylab.title(ttl)

    return


def copy_KOIs():
    input = scipy.loadtxt("%s/vsini_sample_stars.dat" % keplerdefs.ROOTDIR, unpack = True)
    kids = input[0].astype(int)

    for x in scipy.arange(len(kids)):
        lcfile = glob.glob('/Volumes/Data/kepler/Q1_corr_fits_jul/kplr%09d-?????????????_llc.fits' % kids[x])
        if len(lcfile) != 0:
            infile = lcfile[0]
            outfile = infile[:21]  + 'Q1_KOI_2' + infile[37:]
            stin = 'cp ' + infile + ' ' + outfile
            os.system(stin)
    return

# to read Chris's files
def RNFinder(KID):

    file = '%s/GlobalLCInfo.dat' % keplerdefs.ROOTDIR
    X2 = scipy.genfromtxt(file)
    ind = scipy.where(X2.T[1] == KID)
    ind = ind[0]

    print 'ind', ind
    if len(ind) == 0: return scipy.array([-999])

    return X2.T[:,ind]

def get_allQ(dir = None, kic = None):

    if dir == None:
        print 'Must supply directory'
        return
    if id == None:
        print 'Must supply kepler id'
        return

    LcFiles = glob.glob('%s/kplr%09d-?????????????_llc.fits' % (dir,kic))
    nlc = len(LcFiles)
    if nlc == 0:
        print 'File not found for KIC %d' % kic
        print 'Was looking for: %s/kplr%09d-?????????????_llc.fits' % (dir,kic)
        return scipy.array([]), 0
    else:
        print 'Found %d long cadence lightcurve files for KIC %d' % (nlc, kic)

    tset = atpy.TableSet()
    for i in scipy.arange(nlc):
        temp_tab = atpy.Table(LcFiles[i], type = 'fits')
        tset.append(temp_tab)


    tset.add_keyword('length', nlc)
    return tset, 1
