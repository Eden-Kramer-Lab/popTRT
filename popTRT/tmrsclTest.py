import numpy as _N
from kassdirs import resFN
import matplotlib.pyplot as _plt

def timeRescaleTest(fr, spks01, dt, TR, m, nohist=False, loP=0.00001):
    """
    t in units of 1.
    """

    Lspkts   = []
    for tr in xrange(TR):
        Lspkts.append(_N.where(spks01[tr] == 1)[0])
        #print "%(tr)d   %(l)d" % {"l" : len(Lspkts[tr]), "tr" : tr}

    if m > 1:
        dtm      = dt/m
        frm, mspkts = zoom(fr, Lspkts, m, nohist=nohist, loP=loP)
    else:
        frm      = fr
        mspkts   = Lspkts
        dtm      = dt

    if len(fr.shape) == 1:
        T  = fr.shape[0]
        TR = 1
        fr.reshape(T, 1)
    else:
        TR = fr.shape[0]
        T  = fr.shape[1]

    zs = []
    rts= []
    Nm1 = 0

    for tr in xrange(TR):
        N  = len(mspkts[tr])
        rt = _N.empty(N)    #  rescaled time
        rt[0] = _N.trapz(frm[tr, 0:mspkts[tr][0]])*dtm

        for i in xrange(1,  N):
            rt[i] = rt[i-1]+_N.trapz(frm[tr, mspkts[tr][i-1]:mspkts[tr][i]])*dtm
        # for i in xrange(N):
        #     rt[i] = _N.trapz(frm[tr, 0:mspkts[tr][i]])*dtm

        #  this means that small ISIs are underrepresented
        taus = _N.diff(rt)
        zs.extend((1 - _N.exp(-taus)).tolist())
        if N > 0:   #  this trial has spikes
            Nm1 += N - 1
        rts.append(rt)

    zss  = _N.sort(_N.array(zs))
    ks  = _N.arange(1, Nm1 + 1)
    bs  = (ks - 0.5) / Nm1         #  needs
    bsp = bs + 1.36 / _N.sqrt(Nm1)
    bsn = bs - 1.36 / _N.sqrt(Nm1)
    x   = _N.linspace(1./Nm1, 1, Nm1)
    return x, zss, bs, bsp, bsn, rts

def zoom(fr, spkts, m, nohist=False, loP=0.0001):
    """
    fr = [  29,  30,  0]
    sp = [   0,   1,  0]   spike occurs at last point fr is high
    m   multiply time by
    """
    if len(fr.shape) == 1:
        T  = fr.shape[0]
        TR = 1
        fr.reshape(T, 1)
    else:
        TR = fr.shape[0]
        T  = fr.shape[1]

    frm = _N.empty((TR, T*m))
    x   = _N.linspace(0, 1, T*m, endpoint=False)  #  don't include x=1
    Lmspkts = []
    if not nohist:
        for tr in xrange(TR):
            Lmspkts.append(_N.empty(len(spkts[tr]), dtype=_N.int))

            lt  = -1
            for i in xrange(len(spkts[tr])):
                sti = spkts[tr][i]
                frm[tr, (lt+1)*m:(sti+1)*m] = _N.interp(x[(lt+1)*m:(sti+1)*m],
                                                        x[(lt+1)*m:(sti+1)*m:m],
                                                        fr[tr, lt+1:sti+1])
                #  somewhere in [spkts[i]*m:(spkts[i]+1)*m]
                Lmspkts[tr][i] = sti*m + int(_N.random.rand()*m)
                frm[tr, Lmspkts[tr][i]+1:(sti+1)*m] = loP
                lt = sti
    else:
        for tr in xrange(TR):
            Lmspkts.append(_N.empty(len(spkts[tr]), dtype=_N.int))
            frm[tr] = _N.interp(x, x[::m], fr[tr])
            for i in xrange(len(spkts[tr])):
                Lmspkts[tr][i] = spkts[tr][i]*m
        
    return frm, Lmspkts

