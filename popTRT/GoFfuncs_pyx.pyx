import numpy as _N
from EnDedirs import resFN, datFN
import kdeutil as _ku
import time as _tm
import matplotlib.pyplot as _plt
import hc_bcast as _hb
cimport hc_bcast as _hb
from libc.stdio cimport printf
from libc.math cimport sqrt

mz_CRCL = 0
mz_W    = 1

class GoFfuncs:
    kde   = None
    lmd   = None
    lmd0  = None
    mltMk = 1    #  multiply mark values to 

    marksObserved = None   #  observed in this encode epoch

    #  xp   position grid.  need this in decode
    xp    = None
    xpr   = None   # reshaped xp
    dxp   = None

    #  current posterior model parameters
    u_    = None
    covs_ = None
    f_    = None
    q2_   = None
    l0_   = None

    #  initting fitMvNorm
    kde      = False
    Bx       = None;     bx     = None;     Bm    = None

    tetfile  = "marks.pkl"
    usetets  = None
    utets_str= ""

    tt0      = None
    tt1      = None

    maze     = None

    dbgMvt   = False
    spdMult   = 0.5

    Nx       = 121

    xLo      = 0
    xHi      = 3
    mLo      = -2
    mHi      = 8
    chmins   = None

    sts_per_tet = None
    _sts_per_tet = None
    svMkIntnsty = None   #  save just the mark intensities

    ##  X_   and _X
    def __init__(self, Nx=61, kde=False, bx=None, Bx=None, Bm=None, mkfn=None, encfns=None, K=None, xLo=0, xHi=3, maze=mz_CRCL, spdMult=0.1, ignorespks=False, chmins=None, rotate=False):
        """
        """
        oo = self
        oo.Nx = Nx
        oo.maze = maze
        oo.kde = kde
        if chmins is not None:
            oo.chmins = chmins
        else:
            oo.chmins = _N.ones(K)*-10000

        oo.spdMult = spdMult
        oo.ignorespks = ignorespks
        oo.bx = bx;   oo.Bx = Bx;   oo.Bm = Bm
        #  read mkfns
        _sts   = []#  a mark on one of the several tetrodes
        oo._sts_per_tet = []

        #  rotation about axis 1
        th1 = _N.pi/4
        rot1  = _N.array([[1, 0, 0,            0],
                          [0, 1, 0,            0],
                          [0, 0, _N.cos(th1),  _N.sin(th1)],
                          [0, 0, -_N.sin(th1), _N.cos(th1)]])

        #  roation about axis 4
        th4  = (54.738/180.)*_N.pi
        rot4  = _N.array([[1, 0, 0,            0],
                          [0, _N.cos(th4), _N.sin(th4), 0],
                          [0, -_N.sin(th4), _N.cos(th4), 0],
                          [0,            0,      0, 1]])


        th3   = (60.0/180.)*_N.pi
        rot3  = _N.array([[_N.cos(th3), _N.sin(th3), 0, 0],
                          [-_N.sin(th3), _N.cos(th3), 0, 0],
                          [0,            0,      1, 0],
                          [0,            0,      0, 1]]
        )

        _dat = _N.loadtxt(datFN("%s.dat" % mkfn))
        if K is None:
            K   = _dat.shape[1] - 2
            dat = _dat
        else:
            dat = _dat[:, 0:2+K]
            
        oo.mkpos = dat
        spkts = _N.where(dat[:, 1] == 1)[0]

        if rotate:
            for t in spkts:
                dat[t, 2:] = _N.dot(rot3, _N.dot(rot4, dat[t, 2:]))
                    
        oo._sts_per_tet.append(spkts)
        _sts.extend(spkts)
        oo.sts = _N.unique(_sts)

        oo.mdim  = K
        oo.pos  = dat[:, 0]         #  length of 

        if not kde:
            oo.mdim  = K
        
        oo.xLo = xLo;     oo.xHi = xHi
        
        ####  spatial grid for evaluating firing rates
        oo.xp   = _N.linspace(oo.xLo, oo.xHi, oo.Nx)  #  space points
        oo.xpr  = oo.xp.reshape((oo.Nx, 1))
        #  bin space for occupation histogram.  same # intvs as space points
        oo.dxp   = oo.xp[1] - oo.xp[0]
        oo.xb    = _N.empty(oo.Nx+1)
        oo.xb[0:oo.Nx] = oo.xp - 0.5*oo.dxp
        oo.xb[oo.Nx] = oo.xp[-1]+ 0.5*oo.dxp
        ####

        #oo.lmdFLaT = oo.lmd.reshape(oo.Nx, oo.Nm**oo.mdim)
        oo.dt = 0.001

    def LmdMargOvrMrks(self, enc_t0, enc_t1, uFE=None, prms=None):
        """
        0:t0   used for encode
        Lmd0.  
        """
        oo = self
        #####  
        if oo.kde:  #  also calculate the occupation. Nothing to do with LMoMks
            ibx2 = 1./ (oo.bx*oo.bx)        
            occ    = _N.sum((1/_N.sqrt(2*_N.pi*oo.bx*oo.bx))*_N.exp(-0.5*ibx2*(oo.xpr - oo.pos[enc_t0:enc_t1])**2), axis=1)*oo.dxp  #  this piece doesn't need to be evaluated for every new spike
            #  _plt.hist(mkd.pos, bins=mkd.xp) == _plt.plot(mkd.xp, occ)  
            #  len(occ) = total time of observation in ms.
            oo.occ = occ  
            oo.iocc = 1./occ  

            ###  Lam_MoMks  is a function of space
            oo.Lam_MoMks = _ku.Lambda(oo.xpr, oo.tr_pos, oo.pos[enc_t0:enc_t1], oo.Bx, oo.bx, oo.dxp, occ)
        else:  #####  fit mix gaussian
            l0s   = prms[uFE][0]    #  M x 1
            us    = prms[uFE][1]
            covs  = prms[uFE][2]
            M     = covs.shape[0]

            cmps   = _N.zeros((M, oo.Nx))
            for m in xrange(M):
                var = covs[m, 0, 0]
                ivar = 1./var
                cmps[m] = (1/_N.sqrt(2*_N.pi*var)) * _N.exp(-0.5*ivar*(oo.xp - us[m, 0])**2)
            oo.Lam_MoMks   = _N.sum(l0s*cmps, axis=0)

    def prepareDecKDE(self, t0, t1, telapse=0):
        #preparae decoding step for KDE
        oo = self

        sts = _N.where(oo.mkpos[t0:t1, 1] == 1)[0] + t0

        oo.tr_pos = _N.array(oo.mkpos[sts, 0])
        oo.tr_marks = _N.array(oo.mkpos[sts, 2:])


    #####################   GoF tools
    def getGridDims(self, uFE, g_M, prms, smpsPerSD, obsvd_mks):
        # for MoG and GT, we know 
        oo = self

        cdef int nt = 0
        l0   = _N.array(prms[uFE][0])
        print l0
        nonzero = _N.where(l0 > 0.001)[0]
        Mnz    = len(nonzero)

        us     = _N.array(prms[uFE][1])
        covs   = _N.array(prms[uFE][2])

        los    = _N.empty((Mnz, oo.mdim))
        his    = _N.empty((Mnz, oo.mdim))
        cand_grd_szs = _N.empty((Mnz, oo.mdim))

        for m in xrange(Mnz):
            mnz    = nonzero[m]
            his[m]    = us[mnz] + 5*_N.sqrt(_N.diagonal(covs[mnz]))  #  unit
            los[m]    = us[mnz] - 5*_N.sqrt(_N.diagonal(covs[mnz]))  #  unit

            print _N.sqrt(_N.diagonal(covs[mnz]))
            cand_grd_szs[m] = _N.sqrt(_N.diagonal(covs[mnz]))/smpsPerSD

        all_his = _N.max(his, axis=0)
        all_los = _N.min(los, axis=0)

        obsvd_his = _N.max(obsvd_mks, axis=0)
        obsvd_los = _N.min(obsvd_mks, axis=0)

        print "--------------"
        print all_his
        print obsvd_his
        print "--------------"
        print all_los
        print obsvd_los
        print "--------------"
        
        #nObsvd = obsvd_mks.shape[0]
        # for n in xrange(nObsvd):
        #     if (obsvd_mks[n, 0] > all_his[0]):
        #         print "too big %d in 0th dim" % n
        #     if (obsvd_mks[n, 0] < all_los[0]):
        #         print "too small %d in 0th dim" % n
        #     if (obsvd_mks[n, 1] > all_his[1]):
        #         print "too big %d in 1st dim" % n
        #     if (obsvd_mks[n, 1] < all_los[1]):
        #         print "too small %d in 1st dim" % n
        #     if (obsvd_mks[n, 2] > all_his[2]):
        #         print "too big %d in 2nd dim" % n
        #     if (obsvd_mks[n, 2] < all_los[2]):
        #         print "too small %d in 2nd dim" % n
        #     if (obsvd_mks[n, 3] > all_his[3]):
        #         print "too big %d in 3rd dim" % n
        #     if (obsvd_mks[n, 3] < all_los[3]):
        #         print "too small %d in 3rd dim" % n

        grd_szs = _N.min(cand_grd_szs, axis=0)

        amps  = all_his - all_los
        g_Ms = _N.array(amps / grd_szs, dtype=_N.int)

        g_M_max = _N.max(g_Ms)
        print g_Ms
        print g_M_max
        mk_ranges = _N.empty((oo.mdim, g_M_max))


        for im in xrange(oo.mdim):
            mk_ranges[im, 0:g_Ms[im]] = _N.linspace(all_los[im], all_his[im], g_Ms[im], endpoint=True)

        return g_Ms, mk_ranges
            



        


    #####################   GoF tools
    def mark_ranges(self, g_M):
        oo = self
        mk_ranges = _N.empty((oo.mdim, g_M))

        #mgrid     = _N.empty([oo.nTets] + [g_M]*oo.mdim)

        for im in xrange(oo.mdim):
            sts = _N.where(oo.mkpos[:, 1] == 1)[0]

            mL  = _N.min(oo.mkpos[sts, 2+im])
            mH  = _N.max(oo.mkpos[sts, 2+im])
            A   = mH - mL
            mk_ranges[im] = _N.linspace(mL - 0.03*A, mH + 0.03*A, g_M, endpoint=True)

        return mk_ranges

    def rescale_spikes(self, prms, uFE, t0, t1, kde=False):
        """
        uFE    which epoch fit to use for encoding model
        prms posterior params
        use params to decode marks from t0 to t1
        """
        print "epoch used for encoding: %d" % uFE
        oo = self
        ##  each 

        disc_pos = _N.array((oo.pos - oo.xLo) * (oo.Nx/(oo.xHi-oo.xLo)), dtype=_N.int)
                                
        i2pidcovs  = []
        i2pidcovsr = []

        if not kde:
            l0s = _N.array(prms[uFE][0])
            us  = _N.array(prms[uFE][5])
            covs= _N.array(prms[uFE][6])

            M   = covs.shape[0]

            iSgs= _N.linalg.inv(covs)
            i2pidcovs = (1/_N.sqrt(2*_N.pi))**(oo.mdim+1)*(1./_N.sqrt(_N.linalg.det(covs)))
            l0sr = _N.array(l0s)
        else:
            ibx2   = 1. / (oo.bx * oo.bx)
            sptl   = -0.5*ibx2*(oo.xpr - oo.tr_pos)**2  #  this piece doesn't need to be evalu


        fxdMks = _N.empty((oo.Nx, oo.mdim+1))  #  for each pos, a fixed mark
        fxdMks[:, 0] = oo.xp

        rscld = []

        for t in xrange(t0+1, t1): # start at 1 because initial condition
            if (oo.mkpos[t, 1] == 1):
                fxdMks[:, 1:] = oo.mkpos[t, 2:]
                if kde:   #  kerFr returns lambda(m, x) for all x
                    mkint = _ku.kerFr(fxdMks[0, 1:], sptl, oo.tr_marks, oo.mdim, oo.Bx, oo.Bm, oo.bx, oo.dxp, oo.occ)
                else:     #  evalAtFxdMks returns lambda(m, x) for all x
                    mkint = _hb.evalAtFxdMks_new(fxdMks, l0s, us, iSgs, i2pidcovs, M, oo.Nx, oo.mdim + 1)*oo.dt

                lst = [_N.sum(mkint[disc_pos[t0+1:t1]]), _N.sum(mkint[disc_pos[t0+1:t]])]
                lst.extend(oo.mkpos[t, 2:].tolist())

                rscld.append(lst)

        return rscld

    def max_rescaled_T_at_mark_MoG(self, mrngs, long[::1] g_Ms, prms, uFE, long t0, long t1, char iskde):
        """
        method to calculate boundary depends on the model
        uFE    which epoch fit to use for encoding model
        prms posterior params
        use params to decode marks from t0 to t1

        mrngs     # nTets x mdim x M
        """
        oo = self
        ##  each 

        #disc_pos = _N.array((oo.pos - oo.xLo) * (oo.Nx/(oo.xHi-oo.xLo)), dtype=_N.int)
        pos_hstgrm_t0t1, bns = _N.histogram(oo.pos[t0:t1], _N.linspace(oo.xLo, oo.xHi, oo.Nx+1, endpoint=True))

        cdef long[::1] mv_pos_hstgrm_t0t1 = pos_hstgrm_t0t1
        cdef long* p_pos_hstgrm_t0t1    = &mv_pos_hstgrm_t0t1[0]

        #cdef long[::1] mv_disc_pos = disc_pos
        #cdef long* p_disc_pos    = &mv_disc_pos[0]
        cdef double[:, ::1] mv_mrngs = mrngs
        cdef double* p_mrngs     = &mv_mrngs[0, 0]

        sptl= []

        #  4dim
        #ii = ii0*g_Ms[1]*g_Ms[2]*g_Ms[3]+ ii1*g_Ms[2]*g_Ms[3]+ ii2*g_Ms[3]+ ii3
        #ii = ii0*g_M1 + ii1*g_M2 + ii2*g_M3 + ii3
        #  2dim
        #ii = ii0*g_Ms[1]+ ii1
        #ii = ii0*g_M1
        cdef long g_M1, g_M2, g_M3
        cdef long g_M = _N.max(g_Ms)
        if oo.mdim  == 2:
            g_M1 = g_Ms[1]
        elif oo.mdim  == 4:
            g_M1 = g_Ms[1]*g_Ms[2]*g_Ms[3]
            g_M2 = g_Ms[2]*g_Ms[3]
            g_M3 = g_Ms[3]

        l0   = _N.array(prms[uFE][0])
        nonzero = _N.where(l0 > 0.002)[0]
        
        us   = _N.array(prms[uFE][1])
        covs = _N.array(prms[uFE][2])
        fs   = _N.array(prms[uFE][3])
        q2s  = _N.array(prms[uFE][4])

        cdef long M  = covs.shape[0]
        cdef long Mnz  = covs.shape[0]

        iSgs = _N.linalg.inv(prms[uFE][6])
        l0dt = _N.array(prms[uFE][0]*oo.dt)

        i2pidcovs = _N.array((_N.sqrt(2*_N.pi)**(oo.mdim+1))*_N.sqrt(_N.linalg.det(prms[uFE][6])))

        l0dt_i2pidcovs = l0dt/i2pidcovs

        iCovs        = _N.linalg.inv(covs)
        iq2s          = _N.array(1./q2s)

        cdef char* p_O01
        cdef char[::1] mv_O011
        cdef char[:, ::1] mv_O012
        cdef char[:, :, :, ::1] mv_O014
        cdef double* p_O
        cdef double[::1] mv_O1
        cdef double[:, ::1] mv_O2
        cdef double[:, :, :, ::1] mv_O4

        if oo.mdim == 1:
            O = _N.zeros(g_Ms[0])   #  where lambda is near 0, so is O
            mv_O1   = O
            p_O    = &mv_O1[0]
        elif oo.mdim == 2:
            O = _N.zeros([g_Ms[0], g_Ms[1]])
            mv_O2   = O
            p_O    = &mv_O2[0, 0]
        elif oo.mdim == 4:
            O = _N.zeros([g_Ms[0], g_Ms[1], g_Ms[2], g_Ms[3]])
            mv_O4   = O
            p_O    = &mv_O4[0, 0, 0, 0]
            O01 = _N.zeros([g_Ms[0], g_Ms[1], g_Ms[2], g_Ms[3]], dtype=_N.uint8)   #  where lambda is near 0, so is O
            mv_O014 = O01
            p_O01 = &mv_O014[0, 0, 0, 0]


        mk = _N.empty(oo.mdim)
        cdef double[::1] mv_mk = mk
        cdef double* p_mk         = &mv_mk[0]
        cdef double[::1] mv_xp    = oo.xp
        cdef double* p_xp         = &mv_xp[0]

        #disc_pos_t0t1 = _N.array(disc_pos[t0+1:t1])
        #cdef long[::1] mv_disc_pos_t0t1 = disc_pos_t0t1
        #cdef long* p_disc_pos_t0t1      = &mv_disc_pos_t0t1[0]

        cdef long ooNx = oo.Nx
        cdef long pmdim = oo.mdim + 1
        cdef long mdim = oo.mdim
        cdef double ddt = oo.dt

        qdr_mk    = _N.empty(M)
        cdef double[::1] mv_qdr_mk = qdr_mk
        cdef double* p_qdr_mk      = &mv_qdr_mk[0]
        qdr_sp    = _N.empty((M, oo.Nx))
        cdef double[:, ::1] mv_qdr_sp = qdr_sp
        cdef double* p_qdr_sp      = &mv_qdr_sp[0, 0]

        cdef double[:, ::1] mv_us = us
        cdef double* p_us   = &mv_us[0, 0]
        cdef double[::1] mv_fs = fs
        cdef double* p_fs   = &mv_fs[0]
        cdef double[::1] mv_iq2s = iq2s
        cdef double* p_iq2s   = &mv_iq2s[0]

        cdef double[:, :, ::1] mv_iSgs = iSgs
        cdef double* p_iSgs   = &mv_iSgs[0, 0, 0]
        cdef double[::1] mv_l0dt_i2pidcovs = l0dt_i2pidcovs
        cdef double* p_l0dt_i2pidcovs   = &mv_l0dt_i2pidcovs[0]
        cdef double[:, :, ::1] mv_iCovs = iCovs
        cdef double* p_iCovs   = &mv_iCovs[0, 0, 0]

        LLcrnr = mrngs[:, 0]   # lower left hand corner
        cdef double LLcrnr0=mrngs[0, 0]
        cdef double LLcrnr1=mrngs[1, 0]
        cdef double LLcrnr2=mrngs[2, 0]
        cdef double LLcrnr3=mrngs[3, 0]

        dm = _N.array(_N.diff(mrngs)[:, 0])
        cdef double[::1] mv_dm = dm   #  memory view
        cdef double *p_dm         = &mv_dm[0]

        CIF_at_grid_mks = _N.zeros(oo.Nx)
        cdef double[::1] mv_CIF_at_grid_mks = CIF_at_grid_mks
        cdef double*     p_CIF_at_grid_mks  = &mv_CIF_at_grid_mks[0]

        cdef int tt, ii, i0, i1, i2, i3, u0, u1, u2, u3, w0, w1, w2, w3
        cdef long nx, m
        cdef double mrngs0, mrngs1, mrngs2, mrngs3
        cdef int icnt = 0, cum_icnt 

        cdef double tt1, tt2

        #_hb.CIFspatial_nogil(p_xp, p_l0dt_i2pidcovs, p_fs, p_iq2s, p_qdr_sp, M, ooNx, ddt)

        Mnz  = len(nonzero)
        for c in xrange(M):   # pre-compute this
            qdr_sp[c] = (oo.xp - fs[c])*(oo.xp - fs[c])*iq2s[c]
        for c in nonzero:
            tt1 = _tm.time()
            icnt = 0
            cK = c*oo.mdim
            printf("doing cluster %d\n" % c)
            u0 = <int>((p_us[cK] - LLcrnr0) / p_dm[0])
            u1 = <int>((p_us[cK+1] - LLcrnr1) / p_dm[1])
            u2 = <int>((p_us[cK+2] - LLcrnr2) / p_dm[2])
            u3 = <int>((p_us[cK+3] - LLcrnr3) / p_dm[3])

            w0 = <int>((sqrt(covs[c, 0, 0]) / p_dm[0])*4)
            w1 = <int>((sqrt(covs[c, 1, 1]) / p_dm[1])*4)
            w2 = <int>((sqrt(covs[c, 2, 2]) / p_dm[2])*4)
            w3 = <int>((sqrt(covs[c, 3, 3]) / p_dm[3])*4)

            #printf("us %d %d %d %d   ws %d %d %d %d\n", u0, u1, u2, u3, w0, w1, w2, w3)

            #  so i look from 
            with nogil:
                for i0 in xrange(u0 - w0, u0 + w0+1):
                    for i1 in xrange(u0 - w0, u0 + w0+1):
                        for i2 in xrange(u0 - w0, u0 + w0+1):
                            for i3 in xrange(u0 - w0, u0 + w0+1):
                                ii = i0*g_M1+ i1*g_M2+ i2*g_M3+ i3
                                if p_O01[ii] == 0:
                                    p_O01[ii] = 1

                                    icnt += 1
                                    #  mrngs   # mdim x g_M
                                    p_mk[0] = p_mrngs[i0]
                                    p_mk[1] = p_mrngs[g_M + i1]   #  mrngs is 4 vecs of dim g_M
                                    p_mk[2] = p_mrngs[2*g_M + i2]
                                    p_mk[3] = p_mrngs[3*g_M + i3]
                                    _hb.CIFatFxdMks_nogil(p_mk, p_xp, p_l0dt_i2pidcovs, p_us, p_iCovs, p_fs, p_iq2s, p_CIF_at_grid_mks, p_qdr_mk, p_qdr_sp, Mnz, ooNx, mdim, ddt)
                                    p_O[ii] = 0

                                    for nn in xrange(ooNx):
                                        p_O[ii] += p_pos_hstgrm_t0t1[nn]*p_CIF_at_grid_mks[nn]
                                    ##  summing over entire path is VERY slow.  we get roughly 100x speed up when using histogram
                                    #for tt in xrange(0, t1-t0-1):
                                    #    p_O[ii] += p_CIF_at_grid_mks[p_disc_pos_t0t1[tt]]

            tt2 = _tm.time()
            printf("done   %.4f, icnt  %d\n", (tt2-tt1), icnt)
        return O




    # def max_rescaled_T_at_mark_KDE(self, mrngs, long g_M, prms, uFE, long t0, long t1, long rad, char iskde, smpld_marks=None):
    #     """
    #     uFE    which epoch fit to use for encoding model
    #     prms posterior params
    #     use params to decode marks from t0 to t1

    #     mrngs     # nTets x mdim x M
    #     """
    #     print "epoch used for encoding: %d" % uFE
    #     oo = self
    #     ##  each 

    #     disc_pos = _N.array((oo.pos - oo.xLo) * (oo.Nx/(oo.xHi-oo.xLo)), dtype=_N.int)
    #     cdef long[::1] mv_disc_pos = disc_pos
    #     cdef long* p_disc_pos    = &mv_disc_pos[0]
    #     cdef double[:, :, ::1] mv_mrngs = mrngs
    #     cdef double* p_mrngs     = &mv_mrngs[0, 0, 0]

    #     oo.svMkIntnsty = []
    #     l0dt_i2pidcovs = []
    #     us  = []
    #     covs= []
    #     fs  = []
    #     q2s = []
    #     M   = []
    #     iSgs= []
    #     #i2pidcovs = []
    #     #i2pidcovsr = []
    #     sptl= []
    #     cdef long g_M2 = g_M*g_M
    #     cdef long g_M3 = g_M*g_M2
    #     cdef long g_M4 = g_M*g_M3

    #     if iskde == 0:
    #         for nt in xrange(oo.nTets):
    #             #l0s.append(prms[nt][uFE][0])
    #             us.append(prms[nt][uFE][1])
    #             covs.append(prms[nt][uFE][2])
    #             fs.append(prms[nt][uFE][3])
    #             q2s.append(prms[nt][uFE][4])
    #             #fs_us.append(prms[nt][uFE][5])
    #             #q2s_covs.append()

    #             M.append(covs[nt].shape[0])

    #             iSgs.append(_N.linalg.inv(prms[nt][uFE][6]))
    #             l0dt = (prms[nt][uFE][0]*oo.dt)
    #             i2pidcovs = (_N.sqrt(2*_N.pi)**(oo.mdim+1))*_N.sqrt(_N.linalg.det(prms[nt][uFE][6]))

    #             l0dt_i2pidcovs.append(l0dt/i2pidcovs)
    #             #i2pidcovsr.append(i2pidcovs.reshape((M, 1)))
    #             #l0dt = _N.array(l0s[0][:, 0]*oo.dt)  # for nt==0
    #     else:
    #         ibx2   = 1. / (oo.bx * oo.bx)
    #         for nt in xrange(oo.nTets):
    #             sptl.append(-0.5*ibx2*(oo.xpr - oo.tr_pos[nt])**2)  #  this piece doesn't need to be evalu

    #     nt    = 0

    #     cdef char* p_O01
    #     cdef char[::1] mv_O011
    #     cdef char[:, ::1] mv_O012
    #     cdef char[:, :, :, ::1] mv_O014
    #     cdef double* p_O
    #     cdef double[::1] mv_O1
    #     cdef double[:, ::1] mv_O2
    #     cdef double[:, :, :, ::1] mv_O4

    #     cdef long i0, i1, i2, i3
    #     if oo.mdim == 1:
    #         O = _N.zeros(g_M)   #  where lambda is near 0, so is O
    #         O01 = _N.zeros(g_M, dtype=_N.uint8)   #  where lambda is near 0, so is O
    #         mv_O011 = O01
    #         p_O01 = &mv_O011[0]
    #         mv_O1   = O
    #         p_O    = &mv_O1[0]
    #     elif oo.mdim == 2:
    #         O = _N.zeros([g_M, g_M])
    #         O01 = _N.zeros([g_M, g_M], dtype=_N.uint8)   #  where lambda is near 0, so is O
    #         mv_O012 = O01
    #         p_O01 = &mv_O012[0, 0]
    #         mv_O2   = O
    #         p_O    = &mv_O2[0, 0]
    #     elif oo.mdim == 4:
    #         O = _N.zeros([g_M, g_M, g_M, g_M])
    #         O01 = _N.zeros([g_M, g_M, g_M, g_M], dtype=_N.uint8)   #  where lambda is near 0, so is O
    #         mv_O014 = O01
    #         p_O01 = &mv_O014[0, 0, 0, 0]
    #         mv_O4   = O
    #         p_O    = &mv_O4[0, 0, 0, 0]

    #     #mk = _N.empty((oo.Nx, oo.mdim+1))
    #     mk = _N.empty(oo.mdim)
    #     #mk[:, 0] = oo.xp
    #     cdef double[::1] mv_mk = mk
    #     cdef double* p_mk         = &mv_mk[0]
    #     cdef double[::1] mv_xp    = oo.xp
    #     cdef double* p_xp         = &mv_xp[0]

    #     #  temp
    #     mxval = _N.zeros((g_M, oo.Nx))
    #     cdef double[:, ::1] mxval_mv = mxval   #  memory view
    #     cdef double *p_mxval         = &mxval_mv[0, 0]

    #     disc_pos_t0t1 = _N.array(disc_pos[t0+1:t1])
    #     cdef long[::1] mv_disc_pos_t0t1 = disc_pos_t0t1
    #     cdef long* p_disc_pos_t0t1      = &mv_disc_pos_t0t1[0]


    #     cdef long ooNx = oo.Nx
    #     cdef long Mnt
    #     cdef long pmdim = oo.mdim + 1
    #     cdef long mdim = oo.mdim
    #     cdef double ddt = oo.dt
    #     if iskde == 0:
    #         usnt          = _N.array(us[nt])
    #         iSgsnt        = _N.array(iSgs[nt])
    #         iCovsnt        = _N.linalg.inv(covs[nt])
    #         l0dt_i2pidcovsnt   = _N.array(l0dt_i2pidcovs[nt])

    #         fsnt          = _N.array(fs[nt])
    #         iq2snt          = _N.array(1./q2s[nt])
    #         Mnt             = M[nt]

    #     qdr_mk    = _N.empty(Mnt)
    #     cdef double[::1] mv_qdr_mk = qdr_mk
    #     cdef double* p_qdr_mk      = &mv_qdr_mk[0]
    #     qdr_sp    = _N.empty((Mnt, oo.Nx))
    #     cdef double[:, ::1] mv_qdr_sp = qdr_sp
    #     cdef double* p_qdr_sp      = &mv_qdr_sp[0, 0]

    #     cdef double[:, ::1] mv_usnt = usnt
    #     cdef double* p_usnt   = &mv_usnt[0, 0]
    #     cdef double[::1] mv_fsnt = fsnt
    #     cdef double* p_fsnt   = &mv_fsnt[0]
    #     cdef double[::1] mv_iq2snt = iq2snt
    #     cdef double* p_iq2snt   = &mv_iq2snt[0]

    #     cdef double[:, :, ::1] mv_iSgsnt = iSgsnt
    #     cdef double* p_iSgsnt   = &mv_iSgsnt[0, 0, 0]
    #     cdef double[::1] mv_l0dt_i2pidcovsnt = l0dt_i2pidcovsnt
    #     cdef double* p_l0dt_i2pidcovsnt   = &mv_l0dt_i2pidcovsnt[0]
    #     cdef double[:, :, ::1] mv_iCovsnt = iCovsnt
    #     cdef double* p_iCovsnt   = &mv_iCovsnt[0, 0, 0]

    #     #cdef double[::1] mv_l0dt_i2pidcovsnt = l0dt
    #     #cdef double* p_l0dt      = &mv_l0dt[0]

    #     LLcrnr = mrngs[nt, :, 0]   # lower left hand corner

    #     dm = _N.array(_N.diff(mrngs[0])[:, 0])
    #     cdef double[::1] mv_dm = dm   #  memory view
    #     cdef double *p_dm         = &mv_dm[0]

    #     cdef double[:, ::1] mv_smpld_marks
    #     cdef double* p_smpld_marks

    #     cdef long s, i0_l, i0_h, i1_l, i1_h, i2_l, i2_h, i3_l, i3_h
    #     CIF_at_grid_mks = _N.empty(oo.Nx)
    #     cdef double[::1] mv_CIF_at_grid_mks = CIF_at_grid_mks
    #     cdef double*     p_CIF_at_grid_mks  = &mv_CIF_at_grid_mks[0]

    #     cdef long tt, ii, ii0, ii1, ii2, ii3
    #     cdef long nx
    #     cdef double mrngs0, mrngs1, mrngs2, mrngs3
    #     cdef long icnt, cum_icnt 

    #     if (smpld_marks is not None) and (oo.mdim > 1):
    #         tt0 = _tm.time()
    #         sN = smpld_marks.shape[0]   #  smpld_mks   Nx x mdim
    #         mv_smpld_marks  = smpld_marks
    #         p_smpld_marks   = &mv_smpld_marks[0, 0]
    #         for s in xrange(sN):
    #             icnt = 0
    #             ttt0 = _tm.time()
    #             #inds = _N.array((smpld_marks[s] - LLcrnr) / dm, dtype=_N.int)

    #             i0 = <long>((p_smpld_marks[s*mdim] - LLcrnr[0]) / p_dm[0])
    #             i1 = <long>((p_smpld_marks[s*mdim+1] - LLcrnr[1]) / p_dm[1])
    #             i2 = <long>((p_smpld_marks[s*mdim+2] - LLcrnr[2]) / p_dm[2])
    #             i3 = <long>((p_smpld_marks[s*mdim+3] - LLcrnr[3]) / p_dm[3])

    #             if p_O01[i0*g_M3+ i1*g_M2+ i2*g_M+ i3] == 0:
    #                 #p_O01[i0*g_M3+ i1*g_M2+ i2*g_M+ i3] = 1   #  we'll do this one later
    #                 i0_l = i0 - rad if i0 >= rad else 0
    #                 i0_h = i0 + rad+1 if i0 + rad < g_M else g_M
    #                 i1_l = i1 - rad if i1 >= rad else 0
    #                 i1_h = i1 + rad+1 if i1 + rad < g_M else g_M
    #                 i2_l = i2 - rad if i2 >= rad else 0
    #                 i2_h = i2 + rad+1 if i2 + rad < g_M else g_M
    #                 i3_l = i3 - rad if i3  >= rad else 0
    #                 i3_h = i3 + rad+1 if i3 + rad < g_M else g_M

    #                 if iskde == 0:
    #                     with nogil:
    #                         for ii0 in xrange(i0_l, i0_h):
    #                             for ii1 in xrange(i1_l, i1_h):
    #                                 for ii2 in xrange(i2_l, i2_h):
    #                                     for ii3 in xrange(i3_l, i3_h):
    #                                         ii = ii0*g_M3+ ii1*g_M2+ ii2*g_M+ ii3
    #                                         if p_O01[ii] == 0:
    #                                             p_O01[ii] = 1
    #                                             icnt += 1
    #                                             #  mrngs   # nTets x mdim x g_M
    #                                             p_mk[0] = p_mrngs[ii0]
    #                                             p_mk[1] = p_mrngs[g_M + ii1]
    #                                             p_mk[2] = p_mrngs[2*g_M + ii2]
    #                                             p_mk[3] = p_mrngs[3*g_M + ii3]
    #                                             _hb.CIFatFxdMks_nogil(p_mk, p_xp, p_l0dt_i2pidcovsnt, p_usnt, p_iCovsnt, p_fsnt, p_iq2snt, p_CIF_at_grid_mks, p_qdr_mk, p_qdr_sp, Mnt, ooNx, mdim, ddt)
    #                                             p_O[ii] = 0
    #                                             for tt in xrange(0, t1-t0-1):
    #                                                 p_O[ii] += p_CIF_at_grid_mks[p_disc_pos_t0t1[tt]]
    #             cum_icnt += icnt
    #             if icnt > 0:  #  in "for s in xrange(sN):"
    #                 print "spk %(s)d out of %(t)d,   cum mk spc smps %(c)d" % {"s" : s, "t" : sN, "c" : cum_icnt}

    #     tt1 = _tm.time()
    #     print "done   %.4f" % (tt1-tt0)

    #     return O


    # def calc_volrat(self, O, g_Mf, g_Tf, fg_Mf, fg_Tf, m1, m2, t, dtf, O_z, vlr_z):
    #     #  changes in rescaled-time direction is abrupt, while over marks may not be so abrupt.  Cut box in mark direction in 4 

    #     #  assumption O[m1+1, m2+1] = O[m1, m2] + dO_m1
    #     dO_m1 = O[m1+1, m2] - O[m1, m2]
    #     dO_m2 = O[m1, m2+1] - O[m1, m2]

    #     #  make a finer grid for O_z
    #     for im1f in xrange(g_Mf):
    #         for im2f in xrange(g_Mf):
    #             O_z[im1f, im2f] = O[m1, m2] + (im1f/(fg_Mf-1))*dO_m1 + (im2f/(fg_Mf-1))*dO_m2
    #     #O_z[g_Mf-1, g_Mf-1] = O[m1+1, m2+1]

    #     for im1f in xrange(g_Mf-1):
    #         for im2f in xrange(g_Mf-1):
    #             for itf in xrange(g_Tf-1):
    #                 tL = t + itf * dtf
    #                 tH = t + (itf+1) * dtf 

    #                 d1h = tH - O_z[im1f, im2f] 
    #                 d2h = tH - O_z[im1f+1, im2f] 
    #                 d3h = tH - O_z[im1f, im2f+1] 
    #                 d4h = tH - O_z[im1f+1, im2f+1]
    #                 d1l = O_z[im1f, im2f] - tL
    #                 d2l = O_z[im1f+1, im2f] - tL
    #                 d3l = O_z[im1f, im2f+1] - tL
    #                 d4l = O_z[im1f+1, im2f+1] - tL

    #                 if (((d1h > 0) or (d2h > 0) or \
    #                      (d3h > 0) or (d4h > 0)) and \
    #                     ((d1l > 0) or (d2l > 0) or \
    #                      (d3l > 0) or (d4l > 0))):
    #                     #  a border
    #                     if d1h > 0:
    #                         r1h = 1 if (d1h > dtf) else d1h / dtf
    #                     else:
    #                         r1h = 0.01  #  don't set to 0
    #                     if d2h > 0:
    #                         r2h = 1 if (d2h > dtf) else d2h / dtf
    #                     else:
    #                         r2h = 0.01 #  don't set to 0
    #                     if d3h > 0:
    #                         r3h = 1 if (d3h > dtf) else d3h / dtf
    #                     else:
    #                         r3h = 0.01  #  don't set to 0
    #                     if d4h > 0:
    #                         r4h = 1 if (d4h > dtf) else d4h / dtf
    #                     else:
    #                         r4h = 0.01  #  don't set to 0


    #                     vlr_z[im1f, im2f, itf] = r1h*r2h*r3h*r4h
    #                 else:  #  not a border
    #                     if ((d1h < 0) and (d2h < 0) and \
    #                         (d3h < 0) and (d4h < 0)):
    #                         vlr_z[im1f, im2f, itf] = 1
    #                     else:
    #                         vlr_z[im1f, im2f, itf] = 0

    #                 # if (((O_z[im1f, im2f] < tH) or (O_z[im1f+1, im2f] < tH) or \
    #                 #     (O_z[im1f, im2f+1] < tH) or (O_z[im1f+1, im2f+1] < tH)) and \
    #                 #     ((O_z[im1f, im2f] > tL) or (O_z[im1f+1, im2f] > tL) or \
    #                 #      (O_z[im1f, im2f+1] > tL) or (O_z[im1f+1, im2f+1] > tL))):
    #                 #     #  a border
    #                 #     vlr_z[im1f, im2f, itf] = 
    #                 # else:  #  not a border
    #                 #     if ((O_z[im1f, im2f] > tH) and (O_z[im1f+1, im2f] > tH) and \
    #                 #         (O_z[im1f, im2f+1] > tH) and (O_z[im1f+1, im2f+1] > tH)):
    #                 #         vlr_z[im1f, im2f, itf] = 1
    #                 #     else:
    #                 #         vlr_z[im1f, im2f, itf] = 0

    #     return _N.mean(vlr_z)

                        
                    
        
        
        
