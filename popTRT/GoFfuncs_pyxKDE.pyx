import numpy as _N
from EnDedirs import resFN, datFN
import time as _tm
import matplotlib.pyplot as _plt
import hc_bcast as _hb
cimport hc_bcast as _hb
from libc.stdio cimport printf
from libc.math cimport sqrt, exp, abs
import cython
cimport cython
import fastnum as _fn
cimport fastnum as _fn
import decodeutil as _du
#import CIF_on_grid as CIFongrd

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

    sts_per_tet = None
    _sts_per_tet = None
    svMkIntnsty = None   #  save just the mark intensities

    ##  X_   and _X
    def __init__(self, Nx=61, kde=False, mkfn=None, encfns=None, K=None, xLo=0, xHi=3, maze=mz_CRCL, spdMult=0.1, rotate=False):
        """
        """
        oo = self
        oo.Nx = Nx
        oo.maze = maze
        oo.kde = kde

        oo.spdMult = spdMult
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
        oo.xpr  = oo.xp.reshape((1, oo.Nx))
        #  bin space for occupation histogram.  same # intvs as space points
        oo.dxp   = oo.xp[1] - oo.xp[0]
        oo.xb    = _N.empty(oo.Nx+1)
        oo.xb[0:oo.Nx] = oo.xp - 0.5*oo.dxp
        oo.xb[oo.Nx] = oo.xp[-1]+ 0.5*oo.dxp
        ####

        #oo.lmdFLaT = oo.lmd.reshape(oo.Nx, oo.Nm**oo.mdim)
        oo.dt = 0.001

    def prepareDecKDE(self, t0, t1, telapse=0):
        #preparae decoding step for KDE
        oo = self

        sts = _N.where(oo.mkpos[t0:t1, 1] == 1)[0] + t0

        oo.tr_pos = _N.array(oo.mkpos[sts, 0])
        oo.tr_marks = _N.array(oo.mkpos[sts, 2:])


    #####################   GoF tools
    def getGridDims(self, method, prms, smpsPerSD, sds_to_use, obsvd_mks):
        #  the max-min range needed for each dimension calculated 
        #  by taking the extreme values of u +/- 5*sd for each cluster
        #  g_M then dividing that by smpsPerSD*sd of each cluster, and finding
        #  
        oo = self

        cdef int nt = 0
        l0   = _N.array(prms[0])

        us     = _N.array(prms[1])
        covs   = _N.array(prms[2])
        cdef int M  = covs.shape[0]

        los    = _N.empty((M, oo.mdim))
        his    = _N.empty((M, oo.mdim))
        cand_grd_szs = _N.empty((M, oo.mdim))

        for m in xrange(M):  #  slightly larger area to make sure increasing
            #  
            his[m]    = us[m] + 1.5*sds_to_use*_N.sqrt(_N.diagonal(covs[m]))  #  unit
            los[m]    = us[m] - 1.5*sds_to_use*_N.sqrt(_N.diagonal(covs[m]))  #  unit

            #print _N.sqrt(_N.diagonal(covs[m]))
            cand_grd_szs[m] = _N.sqrt(_N.diagonal(covs[m]))/smpsPerSD

        all_his = _N.max(his, axis=0)
        all_los = _N.min(los, axis=0)

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



        if (method == _du._KDE) or (method == _du._GT):
            grd_szs = _N.min(cand_grd_szs, axis=0)
        else:
            l0s = prms[0]
            covs= prms[2]
            q2s = prms[4]
            norms = l0s/((2*_N.pi**((oo.mdim+1)/2.))*_N.sqrt(_N.linalg.det(covs)*q2s))
            grd_szs = _N.empty(oo.mdim)

            for k in xrange(oo.mdim):
                inds_incr = _N.argsort(cand_grd_szs[:, k])
                print norms[inds_incr]
                print _N.linalg.det(covs)[inds_incr]
            
                grd_szs[k] = cand_grd_szs[inds_incr[0], k]


        amps  = all_his - all_los
        g_Ms = _N.array(amps / grd_szs, dtype=_N.int)

        #  quick hack, to allow me to calculate tetrode 11.
        for im in xrange(oo.mdim):
            if g_Ms[im] > 600:
                print "g_Ms[%(im)d] == %(g_Ms)d.  lower to 600" % {"im" : im, "g_Ms" : g_Ms[im]}
                g_Ms[im] = 600
            #g_Ms[im] = 600 if (g_Ms[im] > 600) else g_Ms[im]
        g_M_max = _N.max(g_Ms)
        
        print g_Ms
        print g_M_max
        mk_ranges = _N.empty((oo.mdim, g_M_max))

        #  returns # of bins for each dimension
        #  mk_ranges is mdim x max(g_Ms)
        
        for im in xrange(oo.mdim):
            mk_ranges[im, 0:g_Ms[im]] = _N.linspace(all_los[im], all_his[im], g_Ms[im], endpoint=True)

        return g_Ms, mk_ranges
            
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def rescale_spikes(self, prms, long t0, long t1, i_spc_occ_dt, kde=False):
        """
        uFE    which epoch fit to use for encoding model
        prms posterior params
        use params to decode marks from t0 to t1
        """
        oo = self
        ##  each 

        disc_pos = _N.array((oo.pos - oo.xLo) * (oo.Nx/(oo.xHi-oo.xLo)), dtype=_N.int)
        cdef long[::1] mv_disc_pos = disc_pos
        cdef long*  p_disc_pos     = &mv_disc_pos[0]
        i2pidcovs  = []
        i2pidcovsr = []

        l0s = _N.array(prms[0])
        mksp_us  = _N.array(prms[5])
        mksp_covs= _N.array(prms[6])

        us   = _N.array(prms[1])
        covs = _N.array(prms[2])
        fs   = _N.array(prms[3])
        q2s  = _N.array(prms[4])
        iCovs        = _N.linalg.inv(covs)
        iq2s          = _N.array(1./q2s)

        cdef long M  = covs.shape[0]
        cdef long t, inn

        iSgs = _N.linalg.inv(prms[6])
        l0dt = _N.array(prms[0])#*oo.dt)

        i2pidcovs = _N.array((_N.sqrt(2*_N.pi)**(oo.mdim+1))*_N.sqrt(_N.linalg.det(prms[6])))

        l0dt_i2pidcovs = l0dt/i2pidcovs

        cdef char use_kde = 1 if (kde == True) else 0
        mk = _N.empty(oo.mdim)
        cdef double[::1] mv_mk = mk
        cdef double* p_mk         = &mv_mk[0]
        cdef long im

        cdef double[::1] mv_i_spc_occ_dt = i_spc_occ_dt
        cdef double* p_i_spc_occ_dt         = &mv_i_spc_occ_dt[0]

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

        cdef double[::1] mv_l0dt_i2pidcovs = l0dt_i2pidcovs
        cdef double* p_l0dt_i2pidcovs   = &mv_l0dt_i2pidcovs[0]
        cdef double[:, :, ::1] mv_iCovs = iCovs
        cdef double* p_iCovs   = &mv_iCovs[0, 0, 0]
        cdef double iBm2

        CIF_at_grid_mks = _N.zeros(oo.Nx)
        cdef double[::1] mv_CIF_at_grid_mks = CIF_at_grid_mks
        cdef double*     p_CIF_at_grid_mks  = &mv_CIF_at_grid_mks[0]

        if use_kde == 1:
            iBm2 = mv_iCovs[0, 0, 0]
            print "%.5f" % iBm2

        M   = covs.shape[0]
        print "M  is %d" % M

        i_mksp_Sgs= _N.linalg.inv(mksp_covs)
        i2pid_mksp_covs = (1/_N.sqrt(2*_N.pi))**(oo.mdim+1)*(1./_N.sqrt(_N.linalg.det(mksp_covs)))
        twopidcovs = _N.array((_N.sqrt(2*_N.pi)**(oo.mdim+1))*_N.sqrt(_N.linalg.det(mksp_covs)))

        l0dt_i2pidcovs = l0s/twopidcovs#(l0s*oo.dt)/twopidcovs

        l0sr = _N.array(l0s)

        #fxdMks = _N.empty((oo.Nx, oo.mdim+1))  #  for each pos, a fixed mark
        #fxdMks[:, 0] = oo.xp
        #fxdMk = _N.empty(oo.mdim)  #  for each pos, a fixed mark

        qdr_mk    = _N.empty(M)

        for c in xrange(M):   # pre-compute this
            qdr_sp[c] = (oo.xp - fs[c])*(oo.xp - fs[c])*iq2s[c]

        sts = _N.where(oo.mkpos[t0+1:t1, 1] == 1)[0]
        cdef double[:, ::1] mv_mkpos = oo.mkpos
        cdef long Nspks = len(sts)

        rscldA = _N.empty((Nspks, mdim+1))  #  Nspks == M for KDE
        cdef double[:, ::1] mv_rscldA = rscldA
        cdef double* p_rscldA = &mv_rscldA[0, 0]


        cifs = _N.empty((Nspks, oo.Nx))
        cdef double[:, ::1] mv_cifs = cifs
        cdef double* p_cifs = &mv_cifs[0, 0]
        cdef long itt = -1

        theseClose = _N.arange(M)   # (KDE) for each spike, keep all close spks
        cdef long[::1] mv_theseClose = theseClose
        cdef long* p_theseClose      = &mv_theseClose[0]
        cdef long Nclose             = M

        ttt0 = _tm.time()
        if kde:
            iBm2 = iCovs[0, 0, 0]
            print "-----  %.4f" % iBm2








        with nogil:
            for t in xrange(t0+1, t1): # start at 1 because initial condition
                #if (oo.mkpos[t, 1] == 1):
                if (mv_mkpos[t, 1] == 1):
                    itt += 1
                    if itt % 200 == 0:
                        #ttt1 = _tm.time()
                        #print "%(itt)d    %(t).3f" % {"itt" : itt, "t" : (ttt1-ttt0)}
                        #print "%(itt)d" % {"itt" : itt}
                        #ttt0 = ttt1
                        printf("%ld\n", itt)

                    for im in xrange(mdim):
                        p_mk[im] = mv_mkpos[t, 2+im]#oo.mkpos[t, 2+im]
                        p_rscldA[pmdim*itt + im+1] = p_mk[im]

                    if use_kde:
                        _hb.CIFatFxdMks_kde_nogil(p_mk, p_l0dt_i2pidcovs, p_us, iBm2, p_CIF_at_grid_mks, p_qdr_mk, p_qdr_sp, p_i_spc_occ_dt, M, ooNx, mdim, ddt)
                    else:
                        _hb.CIFatFxdMks_nogil(p_mk, p_l0dt_i2pidcovs, p_us, p_iCovs, p_CIF_at_grid_mks, p_qdr_mk, p_qdr_sp, M, ooNx, mdim, ddt)

                    for inn in xrange(ooNx):
                        p_cifs[itt*ooNx + inn] = p_CIF_at_grid_mks[inn]
                    #  the rescaling of spike at time t depends on mark of that spk

                    p_rscldA[pmdim*itt] = _fn.sum_random_inds_nogil(p_CIF_at_grid_mks, p_disc_pos, t0, t)*ddt  #  actual rescaled time is 2nd element.  1st element used to draw boundary for 1D mark


        if use_kde:
            _N.savetxt("cifs_kde.dat", cifs, fmt="%.5e")
        else:
            _N.savetxt("cifs_gt.dat", cifs, fmt="%.5e")
        return rscldA


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def max_rescaled_T_at_mark_MoG(self, char use_kde, mrngs, long[::1] g_Ms, prms, long t0, long t1, double smpsPerSD, double sds_to_use, double[::1] mv_i_spc_occ_dt):
        """
        method to calculate boundary depends on the model
        prms posterior params
        use params to decode marks from t0 to t1

        mrngs     # nTets x mdim x M
        """
        oo = self
        ##  each 

        pos_hstgrm_t0t1, bns = _N.histogram(oo.pos[t0:t1], _N.linspace(oo.xLo, oo.xHi, oo.Nx+1, endpoint=True))

        cdef long[::1] mv_pos_hstgrm_t0t1 = pos_hstgrm_t0t1
        cdef long* p_pos_hstgrm_t0t1    = &mv_pos_hstgrm_t0t1[0]

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

        l0   = _N.array(prms[0])
        
        us   = _N.array(prms[1])
        covs = _N.array(prms[2])
        fs   = _N.array(prms[3])
        q2s  = _N.array(prms[4])

        cdef long M  = covs.shape[0]
        theseClose = _N.arange(M)   # (KDE) for each spike, keep all close spks
        cdef long[::1] mv_theseClose = theseClose
        cdef long* p_theseClose      = &mv_theseClose[0]

        iSgs = _N.linalg.inv(prms[6])
        l0dt = _N.array(prms[0])#*oo.dt)

        i2pidcovs = _N.array((_N.sqrt(2*_N.pi)**(oo.mdim+1))*_N.sqrt(_N.linalg.det(prms[6])))

        l0dt_i2pidcovs = l0dt/i2pidcovs

        iCovs        = _N.linalg.inv(covs)
        iq2s          = _N.array(1./q2s)

        cdef char* p_O01
        cdef char[::1] mv_O011
        cdef char[:, ::1] mv_O012
        cdef char[:, :, :, ::1] mv_O014
        cdef float* p_O
        cdef float[::1] mv_O1
        cdef float[:, ::1] mv_O2
        cdef float[:, :, :, ::1] mv_O4

        if oo.mdim == 1:
            O = _N.zeros(g_Ms[0], dtype=_N.float32)   #  where lambda is near 0, so is O
            mv_O1   = O
            p_O    = &mv_O1[0]
            O01 = _N.zeros([g_Ms[0]], dtype=_N.uint8)   #  where lambda is near 0, so is O
            mv_O011 = O01
            p_O01 = &mv_O011[0]
        elif oo.mdim == 2:
            O = _N.zeros([g_Ms[0], g_Ms[1]], dtype=_N.float32)
            mv_O2   = O
            p_O    = &mv_O2[0, 0]
            O01 = _N.zeros([g_Ms[0], g_Ms[1]], dtype=_N.uint8)   #  where lambda is near 0, so is O
            mv_O012 = O01
            p_O01 = &mv_O012[0, 0]
        elif oo.mdim == 4:
            O = _N.zeros([g_Ms[0], g_Ms[1], g_Ms[2], g_Ms[3]], dtype=_N.float32)
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

        cdef double* p_i_spc_occ_dt         = &mv_i_spc_occ_dt[0]

        cdef long ooNx = oo.Nx
        cdef long pmdim = oo.mdim + 1
        cdef long mdim = oo.mdim, cmdim
        
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

        cdef double[::1] mv_l0dt_i2pidcovs = l0dt_i2pidcovs
        cdef double* p_l0dt_i2pidcovs   = &mv_l0dt_i2pidcovs[0]
        cdef double l0dt_i2pidcov   = p_l0dt_i2pidcovs[0]
        cdef double[:, :, ::1] mv_iCovs = iCovs
        cdef double* p_iCovs   = &mv_iCovs[0, 0, 0]
        cdef double iBm, iBm2
        if use_kde == 1:
            iBm2 = mv_iCovs[0, 0, 0]
            Bm = 1./sqrt(iBm2)

        LLcrnr = mrngs[:, 0]   # lower left hand corner
        cdef double LLcrnr0
        cdef double LLcrnr1
        cdef double LLcrnr2
        cdef double LLcrnr3

        if mdim >= 1:
            LLcrnr0=mrngs[0, 0]
        if mdim >= 2:
            LLcrnr1=mrngs[1, 0]
        if mdim == 4:
            LLcrnr2=mrngs[2, 0]
            LLcrnr3=mrngs[3, 0]

        #  mrngs is mdim x g_Ms
        dm = _N.array(_N.diff(mrngs)[:, 0])   #  dm dimension is mdim
        cdef double[::1] mv_dm = dm   #  memory view
        cdef double *p_dm         = &mv_dm[0]
        idm = 1./dm
        cdef double[::1] mv_idm = idm   #  memory view
        cdef double *p_idm         = &mv_idm[0]


        CIF_at_grid_mks = _N.zeros(oo.Nx)
        cdef double[::1] mv_CIF_at_grid_mks = CIF_at_grid_mks
        cdef double*     p_CIF_at_grid_mks  = &mv_CIF_at_grid_mks[0]

        cdef long ii, iii    #  needs long because 128**4 is max addressable
        cdef long nx, m
        cdef double mrngs0, mrngs1, mrngs2, mrngs3
        cdef int icnt = 0, cum_icnt, ix
        cdef int skip0, skip1, skip2, skip3
        cdef double d_skip0, d_skip1, d_skip2, d_skip3

        cdef double u_g0, u_g1, u_g2, u_g3, tmp
        cdef int i0, i1, i2, i3, i, j, k, u0, u1, u2, u3, w0, w1, w2, w3
        cdef int lo_0, hi_0, lo_1, hi_1, lo_2, hi_2, lo_3, hi_3

        cdef double tt1, tt2
        cdef double m0, m1, m2, m3, iskip0, iskip1, iskip2, iskip3, iv
        cdef long I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8, I_9, I_10, I_11, I_12, I_13, I_14, I_15, id0, id1, id2, id3
        cdef double za, zb, zc, zd, ze, zf, zg, zh, zi, zj, zk, zl, zm, zn, zo, zp
        cdef int im10, im11, im12, im20, im21, im22, im30, im31, im32, im40, im41, im42
        cdef int Nclose, coi, coiK, cK, c, ic, icM   #KDE 
        cdef double close = covs[0, 0, 0]*4*sds_to_use*sds_to_use   # if kde, bandwidth^2.  both sides, so (2xsds_to_use)**2


        detcovs = _N.linalg.det(covs)
        sA      = detcovs.argsort()
        cdef long[::1] mv_sA = sA
        cdef long*     p_sA  = &mv_sA[0]
        
        cdef int skp_0_0, skp_0_1, skp_0_2, skp_0_3

        if mdim == 1:
            skp_0_0 = <int>(g_Ms[0]/10)    #  skip everywhere - sampling near CIF==0
        elif mdim == 2:
            skp_0_0 = <int>(g_Ms[0]/10)    #  skip everywhere - sampling near CIF==0
            skp_0_1 = <int>(g_Ms[1]/10)
        elif mdim == 4:
            skp_0_0 = <int>(g_Ms[0]/10)    #  skip everywhere - sampling near CIF==0
            skp_0_1 = <int>(g_Ms[1]/10)
            skp_0_2 = <int>(g_Ms[2]/10)
            skp_0_3 = <int>(g_Ms[3]/10)

        if use_kde == 1:  #  all clusters have same width (KDE)
            skip0 = 1
            skip1 = 1
            skip2 = 1
            skip3 = 1


        timePerSpike = _N.empty(M)
        cdef double[::1] mv_timePerSpike = timePerSpike
        cdef double* p_timePerSpike = &mv_timePerSpike[0]

        print "M  is %d" % M

        ##  if mdim == 1
        ##  if use_kde
        ##     if mdim == 4
        ##  if not use_kde
        for c in xrange(M):   # pre-compute this
            qdr_sp[c] = (oo.xp - fs[c])*(oo.xp - fs[c])*iq2s[c]

        if mdim == 1:  
            u0 = <int>((p_us[cK]   - LLcrnr0) / p_dm[0])

            if use_kde == 0:  #  all clusters have same width (KDE)
                d_skip0 = (sqrt(covs[c, 0, 0]) / smpsPerSD)/p_dm[0]
                skip0   = <int>_N.ceil(d_skip0 - 0.05)  #  allow for 1.01 to be 1
                skip0   = 1
            idl_grd_sz0 = p_dm[0]*skip0

            w0 = <int>((sqrt(covs[c, 0, 0]) / idl_grd_sz0)*sds_to_use)

            printf("skip0   %d\n" % skip0)
            with nogil:
                for i0 from 0 <= i0 < g_Ms[0]:
                    p_mk[0] = p_mrngs[i0]  #  mrngs[0, i0]  mrngs[0, 0:g_Ms[0]]
                    ii = i0
                    if (not ((i0 > g_Ms[0]) or (i0 < 0)) and \
                             (p_O01[ii] == 0)):
                        #if p_O01[ii] == 0:
                        p_O01[ii] = 1

                        icnt += 1
                        #  mrngs   # mdim x g_M
                        if use_kde:
                            _hb.CIFatFxdMks_kde_nogil(p_mk, p_l0dt_i2pidcovs, p_us, iBm2, p_CIF_at_grid_mks, p_qdr_mk, p_qdr_sp, p_i_spc_occ_dt, M, ooNx, mdim, ddt)
                        else:
                            _hb.CIFatFxdMks_nogil(p_mk, p_l0dt_i2pidcovs, p_us, p_iCovs, p_CIF_at_grid_mks, p_qdr_mk, p_qdr_sp, M, ooNx, mdim, ddt)
                        p_O[ii] = 0

                        for nn in xrange(ooNx):
                            p_O[ii] += p_pos_hstgrm_t0t1[nn]*p_CIF_at_grid_mks[nn]
                            #p_O[ii] += p_CIF_at_grid_mks[nn]
                        p_O[ii] *= ddt

                        ##  summing over entire path is VERY slow.  we get roughly 100x speed up when using histogram
                        #for tt in xrange(0, t1-t0-1):
                        #    p_O[ii] += p_CIF_at_grid_mks[p_disc_pos_t0t1[tt]]

        if (use_kde == 0) and (mdim > 1):   #  be explicit
            if mdim == 4:  #  NEAR PEAKS
                for ic in xrange(M):  
                    #  First, compute near peak of all clusters - no approximation
                    #  since near peak is very nonlinear so poor interpolation 
                    c   = p_sA[M-ic-1]  #  from narrowest to widest cluster
                    tt1 = _tm.time()
                    icnt = 0
                    cK = c*oo.mdim
                    printf("doing cluster peaks %d\n" % c)

                    u0 = <int>((p_us[cK]   - LLcrnr0) * p_idm[0])
                    u1 = <int>((p_us[cK+1] - LLcrnr1) * p_idm[1])
                    u2 = <int>((p_us[cK+2] - LLcrnr2) * p_idm[2])
                    u3 = <int>((p_us[cK+3] - LLcrnr3) * p_idm[3])

                    w0 = <int>((sqrt(covs[c, 0, 0]) * p_idm[0])) + 1
                    w1 = <int>((sqrt(covs[c, 1, 1]) * p_idm[1])) + 1
                    w2 = <int>((sqrt(covs[c, 2, 2]) * p_idm[2])) + 1
                    w3 = <int>((sqrt(covs[c, 3, 3]) * p_idm[3])) + 1

                    lo_0 = (u0 - w0) if (u0 - w0) >= 0 else 0
                    hi_0  = (u0 + w0+1) if (u0 + w0+1) <= g_Ms[0] else g_Ms[0]
                    lo_1 = (u1 - w1) if (u1 - w1) >= 0 else 0
                    hi_1  = (u1 + w1+1) if (u1 + w1+1) <= g_Ms[1] else g_Ms[1]
                    lo_2 = (u2 - w2) if (u2 - w2) >= 0 else 0
                    hi_2  = (u2 + w2+1) if (u2 + w2+1) <= g_Ms[2] else g_Ms[2]
                    lo_3 = (u3 - w3) if (u3 - w3) >= 0 else 0
                    hi_3  = (u3 + w3+1) if (u3 + w3+1) <= g_Ms[3] else g_Ms[3]

                    #with nogil:   #  sample near peak densely, no approx
                    for i0 from lo_0 <= i0 < hi_0:
                        for i1 from lo_1 <= i1 < hi_1:
                            for i2 from lo_2 <= i2 < hi_2:
                                for i3 from lo_3 <= i3 < hi_3:
                                    ii = i0*g_M1+ i1*g_M2+ i2*g_M3+ i3
                                    if p_O01[ii] == 0:
                                        p_O01[ii] = 1

                                        icnt += 1
                                        #  mrngs   # mdim x g_M
                                        p_mk[0] = p_mrngs[i0]  #  mrngs[0, i0]  mrngs[0, 0:g_Ms[0]]
                                        p_mk[1] = p_mrngs[g_M + i1]   #  mrngs is 4 vecs of dim g_M
                                        p_mk[2] = p_mrngs[2*g_M + i2]
                                        p_mk[3] = p_mrngs[3*g_M + i3]
                                        _hb.CIFatFxdMks_nogil(p_mk, p_l0dt_i2pidcovs, p_us, p_iCovs, p_CIF_at_grid_mks, p_qdr_mk, p_qdr_sp, M, ooNx, mdim, ddt)
                                        p_O[ii] = 0

                                        for nn in xrange(ooNx):
                                            p_O[ii] += p_pos_hstgrm_t0t1[nn]*p_CIF_at_grid_mks[nn]
                                        p_O[ii] *= ddt
                for ic in xrange(M):  #  now the rest of the mark space
                    c   = p_sA[M-ic-1]  #  from narrowest to widest cluster
                    tt1 = _tm.time()
                    icnt = 0
                    cK = c*mdim

                    u0 = <int>((p_us[cK]   - LLcrnr0) * p_idm[0])
                    u1 = <int>((p_us[cK+1] - LLcrnr1) * p_idm[1])
                    u2 = <int>((p_us[cK+2] - LLcrnr2) * p_idm[2])
                    u3 = <int>((p_us[cK+3] - LLcrnr3) * p_idm[3])

                    d_skip0 = (sqrt(covs[c, 0, 0]) / smpsPerSD) * p_idm[0]
                    skip0   = <int>_N.ceil(d_skip0 - 0.05)  #  allow for 1.01 to be 1
                    d_skip1 = (sqrt(covs[c, 1, 1]) / smpsPerSD) * p_idm[1]
                    skip1   = <int>_N.ceil(d_skip1 - 0.05)
                    d_skip2 = (sqrt(covs[c, 2, 2]) / smpsPerSD) * p_idm[2]
                    skip2   = <int>_N.ceil(d_skip2 - 0.05)
                    d_skip3 = (sqrt(covs[c, 3, 3]) / smpsPerSD) * p_idm[3]
                    skip3   = <int>_N.ceil(d_skip3 - 0.05)

                    # print "%(0)d %(1)d %(2)d %(3)d" % {"0" : skip0, "1" : skip1, "2" : skip2, "3" : skip3}
                    printf("us %d %d %d %d\n", skip0, skip1, skip2, skip3)

                    printf("us %d %d %d %d\n", skip0, skip1, skip2, skip3)
                    #  skips   3~5 -> 2     6~8 -> 3    9~11 -> 4
                    # skip0    = (skip0/2) + 1 if (skip0 > 2) else skip0
                    # skip1    = (skip1/2) + 1 if (skip1 > 2) else skip1
                    # skip2    = (skip2/2) + 1 if (skip2 > 2) else skip2
                    # skip3    = (skip3/2) + 1 if (skip3 > 2) else skip3

                    skip0    = 2 if skip0 > 2 else skip0
                    skip1    = 2 if skip1 > 2 else skip1
                    skip2    = 2 if skip2 > 2 else skip2
                    skip3    = 2 if skip3 > 2 else skip3
                    iskip0   = 1./skip0                    
                    iskip1   = 1./skip1
                    iskip2   = 1./skip2
                    iskip3   = 1./skip3
                    iv       = iskip0 * iskip1 * iskip2 * iskip3

                    idl_grd_sz0 = p_dm[0]*skip0
                    idl_grd_sz1 = p_dm[1]*skip1
                    idl_grd_sz2 = p_dm[2]*skip2
                    idl_grd_sz3 = p_dm[3]*skip3

                    w0 = <int>((sqrt(covs[c, 0, 0]) / idl_grd_sz0)*sds_to_use) + 3
                    w1 = <int>((sqrt(covs[c, 1, 1]) / idl_grd_sz1)*sds_to_use) + 3
                    w2 = <int>((sqrt(covs[c, 2, 2]) / idl_grd_sz2)*sds_to_use) + 3
                    w3 = <int>((sqrt(covs[c, 3, 3]) / idl_grd_sz3)*sds_to_use) + 3

                    lo_0 = (u0 - w0) if (u0 - w0) >= 0 else 0
                    hi_0  = (u0 + w0+1) if (u0 + w0+1) <= g_Ms[0] else g_Ms[0]
                    lo_1 = (u1 - w1) if (u1 - w1) >= 0 else 0
                    hi_1  = (u1 + w1+1) if (u1 + w1+1) <= g_Ms[1] else g_Ms[1]
                    lo_2 = (u2 - w2) if (u2 - w2) >= 0 else 0
                    hi_2  = (u2 + w2+1) if (u2 + w2+1) <= g_Ms[2] else g_Ms[2]
                    lo_3 = (u3 - w3) if (u3 - w3) >= 0 else 0
                    hi_3  = (u3 + w3+1) if (u3 + w3+1) <= g_Ms[3] else g_Ms[3]

                    #with nogil:   #  sample near peak densely, no approx
                    #  away from peak, sample more sparsely
                    for i0 from lo_0 <= i0 < hi_0 by skip0:
                        for i1 from lo_1 <= i1 < hi_1 by skip1:
                            for i2 from lo_2 <= i2 < hi_2 by skip2:
                                for i3 from lo_3 <= i3 < hi_3 by skip3:
                                    ii = i0*g_M1+ i1*g_M2+ i2*g_M3+ i3
                                    if p_O01[ii] == 0:
                                        p_O01[ii] = 1

                                        icnt += 1
                                        #  mrngs   # mdim x g_M
                                        p_mk[0] = p_mrngs[i0]  #  mrngs[0, i0]  mrngs[0, 0:g_Ms[0]]
                                        p_mk[1] = p_mrngs[g_M + i1]   #  mrngs is 4 vecs of dim g_M
                                        p_mk[2] = p_mrngs[2*g_M + i2]
                                        p_mk[3] = p_mrngs[3*g_M + i3]
                                        _hb.CIFatFxdMks_nogil(p_mk, p_l0dt_i2pidcovs, p_us, p_iCovs, p_CIF_at_grid_mks, p_qdr_mk, p_qdr_sp, M, ooNx, mdim, ddt)
                                        p_O[ii] = 0

                                        for nn in xrange(ooNx):
                                            p_O[ii] += p_pos_hstgrm_t0t1[nn]*p_CIF_at_grid_mks[nn]
                                        p_O[ii] *= ddt


                    #  do intrapolation    part 1
                    for i0 from lo_0 <= i0 < (hi_0-skip0) by skip0:
                        im10 = i0
                        im11 = i0 + skip0
                        for i1 from lo_1 <= i1 < (hi_1-skip1) by skip1:
                            im20 = i1
                            im21 = i1 + skip1
                            for i2 from lo_2 <= i2 < (hi_2-skip2) by skip2:
                                im30 = i2
                                im31 = i2 + skip2
                                for i3 from lo_3 <= i3 < (hi_3-skip3) by skip3:
                                    im40 = i3
                                    im41 = i3 + skip3

                                    I_0 = im10*g_M1 + im20*g_M2 + im30*g_M3 + im40

                                    I_1 = im11*g_M1 + im20*g_M2 + im30*g_M3 + im40
                                    I_2 = im10*g_M1 + im21*g_M2 + im30*g_M3 + im40
                                    I_3 = im10*g_M1 + im20*g_M2 + im31*g_M3 + im40
                                    I_4 = im10*g_M1 + im20*g_M2 + im30*g_M3 + im41

                                    I_5 = im11*g_M1 + im21*g_M2 + im30*g_M3 + im40
                                    I_6 = im11*g_M1 + im20*g_M2 + im31*g_M3 + im40
                                    I_7 = im11*g_M1 + im20*g_M2 + im30*g_M3 + im41
                                    I_8 = im10*g_M1 + im21*g_M2 + im31*g_M3 + im40
                                    I_9 = im10*g_M1 + im21*g_M2 + im30*g_M3 + im41
                                    I_10= im10*g_M1 + im20*g_M2 + im31*g_M3 + im41

                                    I_11= im11*g_M1 + im21*g_M2 + im31*g_M3 + im40
                                    I_12= im11*g_M1 + im21*g_M2 + im30*g_M3 + im41
                                    I_13= im11*g_M1 + im20*g_M2 + im31*g_M3 + im41
                                    I_14= im10*g_M1 + im21*g_M2 + im31*g_M3 + im41

                                    I_15= im11*g_M1 + im21*g_M2 + im31*g_M3 + im41


                                    za  = p_O[I_0]
                                    zb  = p_O[I_1]
                                    zc  = p_O[I_2]
                                    zd  = p_O[I_3]
                                    ze  = p_O[I_4]
                                    zf  = p_O[I_5]
                                    zg  = p_O[I_6]
                                    zh  = p_O[I_7]
                                    zi  = p_O[I_8]
                                    zj  = p_O[I_9]
                                    zk  = p_O[I_10]
                                    zl  = p_O[I_11]
                                    zm  = p_O[I_12]
                                    zn  = p_O[I_13]
                                    zo  = p_O[I_14]
                                    zp  = p_O[I_15]



                                    for id0 in xrange(skip0):
                                        im12 = id0 + i0
                                        for id1 in xrange(skip1):
                                            im22 = id1 + i1
                                            for id2 in xrange(skip2):
                                                im32 = id2 + i2
                                                for id3 in xrange(skip3):
                                                    im42 = id3 + i3
                                                    iii = im12*g_M1+ im22*g_M2+ im32*g_M3+ im42
                                                    if p_O01[iii] == 0:
                                                        zint  = iv*(za * (im11-im12)*(im21-im22)*(im31-im32)*(im41-im42) + \
                                                                    ##########################################
                                                                    zb * (im12-im10)*(im21-im22)*(im31-im32)*(im41-im42) + \
                                                                    zc * (im11-im12)*(im22-im20)*(im31-im32)*(im41-im42) + \
                                                                    zd * (im11-im12)*(im21-im22)*(im32-im30)*(im41-im42) + \
                                                                    ze * (im11-im12)*(im21-im22)*(im31-im32)*(im42-im40) + \
                                                                    ##########################################
                                                                    zf * (im12-im10)*(im22-im20)*(im31-im32)*(im41-im42) + \
                                                                    zg * (im12-im10)*(im21-im22)*(im32-im30)*(im41-im42) + \
                                                                    zh * (im12-im10)*(im21-im22)*(im31-im32)*(im42-im40) + \
                                                                    zi * (im11-im12)*(im22-im20)*(im32-im30)*(im41-im42) + \
                                                                    zj * (im11-im12)*(im22-im20)*(im31-im32)*(im42-im40) + \
                                                                    zk * (im11-im12)*(im21-im22)*(im32-im30)*(im42-im40) + \
                                                                    ##########################################
                                                                    zl * (im12-im10)*(im22-im20)*(im32-im30)*(im41-im42) + \
                                                                    zm * (im12-im10)*(im22-im20)*(im31-im32)*(im42-im40) + \
                                                                    zn * (im12-im10)*(im21-im22)*(im32-im30)*(im42-im40) + \
                                                                    zo * (im11-im12)*(im22-im20)*(im32-im30)*(im42-im40) + \
                                                ##########################################
                                                                    zp * (im12-im10)*(im22-im20)*(im32-im30)*(im42-im40))
                                                        p_O[iii] = zint
                                                        p_O01[iii] = 1

                    #  do intrapolation    part 2
                    for i0 from hi_0-skip0 <= i0 < hi_0 by skip0:
                        im10 = i0
                        im11 = i0 + skip0
                        for i1 from hi_1-skip1 <= i1 < hi_1 by skip1:
                            im20 = i1
                            im21 = i1 + skip1
                            for i2 from hi_2-skip2 <= i2 < hi_2 by skip2:
                                im30 = i2
                                im31 = i2 + skip2
                                for i3 from hi_3-skip3 <= i3 < hi_3 by skip3:
                                    im40 = i3
                                    im41 = i3 + skip3

                                    I_0 = im10*g_M1 + im20*g_M2 + im30*g_M3 + im40

                                    I_1 = im11*g_M1 + im20*g_M2 + im30*g_M3 + im40
                                    I_2 = im10*g_M1 + im21*g_M2 + im30*g_M3 + im40
                                    I_3 = im10*g_M1 + im20*g_M2 + im31*g_M3 + im40
                                    I_4 = im10*g_M1 + im20*g_M2 + im30*g_M3 + im41

                                    I_5 = im11*g_M1 + im21*g_M2 + im30*g_M3 + im40
                                    I_6 = im11*g_M1 + im20*g_M2 + im31*g_M3 + im40
                                    I_7 = im11*g_M1 + im20*g_M2 + im30*g_M3 + im41
                                    I_8 = im10*g_M1 + im21*g_M2 + im31*g_M3 + im40
                                    I_9 = im10*g_M1 + im21*g_M2 + im30*g_M3 + im41
                                    I_10= im10*g_M1 + im20*g_M2 + im31*g_M3 + im41

                                    I_11= im11*g_M1 + im21*g_M2 + im31*g_M3 + im40
                                    I_12= im11*g_M1 + im21*g_M2 + im30*g_M3 + im41
                                    I_13= im11*g_M1 + im20*g_M2 + im31*g_M3 + im41
                                    I_14= im10*g_M1 + im21*g_M2 + im31*g_M3 + im41

                                    I_15= im11*g_M1 + im21*g_M2 + im31*g_M3 + im41


                                    za  = p_O[I_0]
                                    zb  = p_O[I_1]
                                    zc  = p_O[I_2]
                                    zd  = p_O[I_3]
                                    ze  = p_O[I_4]
                                    zf  = p_O[I_5]
                                    zg  = p_O[I_6]
                                    zh  = p_O[I_7]
                                    zi  = p_O[I_8]
                                    zj  = p_O[I_9]
                                    zk  = p_O[I_10]
                                    zl  = p_O[I_11]
                                    zm  = p_O[I_12]
                                    zn  = p_O[I_13]
                                    zo  = p_O[I_14]
                                    zp  = p_O[I_15]



                                    for id0 in xrange(skip0):
                                        im12 = id0 + i0
                                        for id1 in xrange(skip1):
                                            im22 = id1 + i1
                                            for id2 in xrange(skip2):
                                                im32 = id2 + i2
                                                for id3 in xrange(skip3):
                                                    im42 = id3 + i3
                                                    iii = im12*g_M1+ im22*g_M2+ im32*g_M3+ im42
                                                    if p_O01[iii] == 0:
                                                        printf("filling 2nd\n")
                                                        zint  = iv*(za * (im11-im12)*(im21-im22)*(im31-im32)*(im41-im42) + \
                                                                    ##########################################
                                                                    zb * (im12-im10)*(im21-im22)*(im31-im32)*(im41-im42) + \
                                                                    zc * (im11-im12)*(im22-im20)*(im31-im32)*(im41-im42) + \
                                                                    zd * (im11-im12)*(im21-im22)*(im32-im30)*(im41-im42) + \
                                                                    ze * (im11-im12)*(im21-im22)*(im31-im32)*(im42-im40) + \
                                                                    ##########################################
                                                                    zf * (im12-im10)*(im22-im20)*(im31-im32)*(im41-im42) + \
                                                                    zg * (im12-im10)*(im21-im22)*(im32-im30)*(im41-im42) + \
                                                                    zh * (im12-im10)*(im21-im22)*(im31-im32)*(im42-im40) + \
                                                                    zi * (im11-im12)*(im22-im20)*(im32-im30)*(im41-im42) + \
                                                                    zj * (im11-im12)*(im22-im20)*(im31-im32)*(im42-im40) + \
                                                                    zk * (im11-im12)*(im21-im22)*(im32-im30)*(im42-im40) + \
                                                                    ##########################################
                                                                    zl * (im12-im10)*(im22-im20)*(im32-im30)*(im41-im42) + \
                                                                    zm * (im12-im10)*(im22-im20)*(im31-im32)*(im42-im40) + \
                                                                    zn * (im12-im10)*(im21-im22)*(im32-im30)*(im42-im40) + \
                                                                    zo * (im11-im12)*(im22-im20)*(im32-im30)*(im42-im40) + \
                                                ##########################################
                                                                    zp * (im12-im10)*(im22-im20)*(im32-im30)*(im42-im40))
                                                        p_O[iii] = zint
                                                        p_O01[iii] = 1

                tt2 = _tm.time()
                printf("**done   %.4f, icnt  %d\n", (tt2-tt1), icnt)

        if (use_kde == 1) and (mdim > 1):   #######KDEKDEKDE
            for icM in xrange(M):  #  now the rest of the mark space
                tt1 = _tm.time()
                icnt = 0
                cK = icM*mdim

                if mdim > 1:
                    #  since for kde we're make a list of closest spikes to this one
                    Nclose = 0
                    for coi in xrange(M):
                        coiK = coi*mdim

                        if mdim == 2:
                            if (((p_us[coiK] - p_us[cK])*(p_us[coiK] - p_us[cK])) + \
                                ((p_us[coiK+1] - p_us[cK+1])*(p_us[coiK+1] - p_us[cK+1])) < close):
                                p_theseClose[Nclose] = coi
                                Nclose += 1
                        elif mdim == 4:
                            if (((p_us[coiK] - p_us[cK])*(p_us[coiK] - p_us[cK])) + \
                                ((p_us[coiK+1] - p_us[cK+1])*(p_us[coiK+1] - p_us[cK+1])) + \
                                ((p_us[coiK+2] - p_us[cK+2])*(p_us[coiK+2] - p_us[cK+2])) + \
                                ((p_us[coiK+3] - p_us[cK+3])*(p_us[coiK+3] - p_us[cK+3])) < close):
                                p_theseClose[Nclose] = coi
                                Nclose += 1

                    printf("doing cluster %d     close %ld\n", icM, Nclose)

                if mdim == 4:
                    with nogil:
                        u0 = <int>((p_us[cK]   - LLcrnr0) * p_idm[0])
                        u1 = <int>((p_us[cK+1] - LLcrnr1) * p_idm[1])
                        u2 = <int>((p_us[cK+2] - LLcrnr2) * p_idm[2])
                        u3 = <int>((p_us[cK+3] - LLcrnr3) * p_idm[3])
                        w0 = <int>((Bm * p_idm[0])*sds_to_use)
                        w1 = <int>((Bm * p_idm[1])*sds_to_use)
                        w2 = <int>((Bm * p_idm[2])*sds_to_use)
                        w3 = <int>((Bm * p_idm[3])*sds_to_use)

                        lo_0 = (u0 - w0) if (u0 - w0) >= 0 else 0
                        hi_0  = (u0 + w0+1) if (u0 + w0+1) <= g_Ms[0] else g_Ms[0]
                        lo_1 = (u1 - w1) if (u1 - w1) >= 0 else 0
                        hi_1  = (u1 + w1+1) if (u1 + w1+1) <= g_Ms[1] else g_Ms[1]
                        lo_2 = (u2 - w2) if (u2 - w2) >= 0 else 0
                        hi_2  = (u2 + w2+1) if (u2 + w2+1) <= g_Ms[2] else g_Ms[2]
                        lo_3 = (u3 - w3) if (u3 - w3) >= 0 else 0
                        hi_3  = (u3 + w3+1) if (u3 + w3+1) <= g_Ms[3] else g_Ms[3]

                        #printf("%d %d   %d %d   %d %d   %d %d", lo_0, hi_0, lo_1, hi_1, lo_2, hi_2, lo_3, hi_3)


                        #  sample near peak densely, no approx
                        #  away from peak, sample more sparsely
                        for i0 from lo_0 <= i0 < hi_0:
                            u_g0 = p_mrngs[i0]  #  mrngs[0, i0]  mrngs[0, 0:g_Ms[0]]
                            for i1 from lo_1 <= i1 < hi_1:
                                u_g1 = p_mrngs[g_M + i1]   #  mrngs is 4 vecs of dim g_M
                                for i2 from lo_2 <= i2 < hi_2:
                                    u_g2 = p_mrngs[2*g_M + i2]
                                    for i3 from lo_3 <= i3 < hi_3:
                                        ii = i0*g_M1+ i1*g_M2+ i2*g_M3+ i3

                                        if p_O01[ii] == 0:
                                            u_g3 = p_mrngs[3*g_M + i3]
                                            for 0 <= ic < Nclose:
                                                c = p_theseClose[ic]
                                                cmdim = c*mdim
                                                p_qdr_mk[c] = iBm2 * ((u_g0-p_us[cmdim]) * (u_g0-p_us[cmdim]) + (u_g1-p_us[cmdim+1]) * (u_g1-p_us[cmdim+1]) + (u_g2-p_us[cmdim+2]) * (u_g2-p_us[cmdim+2]) + (u_g3-p_us[cmdim+3]) * (u_g3-p_us[cmdim+3]))

                                            for 0 <= ix < ooNx:  #  the mark contribution constant, modulating it by spatial contribution
                                                tmp = 0
                                                for 0 <= ic < Nclose:
                                                    c = p_theseClose[ic]
                                                    arg = p_qdr_sp[c*ooNx + ix] +p_qdr_mk[c]
                                                    if arg < 17:
                                                        tmp += exp(-0.5*arg)
                                                p_CIF_at_grid_mks[ix] = tmp * l0dt_i2pidcov * p_i_spc_occ_dt[ix]
                                                p_O[ii] += p_pos_hstgrm_t0t1[ix]*p_CIF_at_grid_mks[ix]
                                            p_O01[ii] = 1
                                            icnt += 1
                                            p_O[ii] *= ddt

                tt2 = _tm.time()
                printf("**done   %.4f, icnt  %d\n", (tt2-tt1), icnt)
                p_timePerSpike[icM] = (tt2-tt1)

        return O, timePerSpike
