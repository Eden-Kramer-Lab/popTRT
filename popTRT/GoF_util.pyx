import cython
import numpy as _N
cimport numpy as _N
from libc.stdio cimport printf
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_volrat(long g_T, long[::1] g_Ms, float[::1] O, double[::1] trngs, float[::1] volrat_mk):
    #  changes in rescaled-time direction is abrupt, while over marks may not be so abrupt.  Cut box in mark direction in 4 
    cdef double tL, tH
    cdef double d1h, d1l
    cdef long ti, inside, outside, border
    cdef long m1

    cdef long   *p_g_Ms  = &g_Ms[0]
    cdef float *p_O     = &O[0]
    cdef double *p_trngs     = &trngs[0]
    cdef float *p_volrat_mk = &volrat_mk[0]

    cdef long it, inboundary, i_here, i_til_end

    cdef long    g_Tm1 = g_T-1
    cdef long g_Mm1 = g_Ms[0]-1

    inside  = 0
    outside = 0
    border  = 0
    cdef double p01, p11, p12, p13, p14, p21, p22, p23, p24, p25, p26, p31, p32, p33, p34, p41

    
    cdef long _m1, _m2, _m3
    cdef long _m1p1, _m2p1, _m3p1, m4p1

    cdef int i_start_search 

    for m1 in xrange(p_g_Ms[0]-1):
        #print "m1  %d" % m1

        inboundary = 1

        it = -1

        while (it < g_T-2) and (inboundary == 1):
            it += 1

            tL = p_trngs[it]
            tH = p_trngs[it+1]

            d1h = tH - p_O[m1]
            d2h = tH - p_O[m1+1]

            d1l = p_O[m1] - tL
            d2l = p_O[m1+1] - tL
                        

            if (((d1h > 0) or (d2h > 0)) and
                ((d1l > 0) or (d2l > 0))):
                tmp = 0.5*(d1l + d2l) / (tH - tL)
                tmp = 0 if (tmp < 0) else tmp
                tmp = 1 if (tmp > 1) else tmp
                p_volrat_mk[m1] += tmp
            else:  #  not a border
                if ((d1h < 0) and (d2h < 0)):
                    p_volrat_mk[m1] += 1

    return inside, outside, border


def find_Occ(long[::1] g_Ms, int NT, double[::1] attimes, double[::1] occ, float[::1] O):
    #  at given rescaled time, number of mark voxels < boundary (rescaled time)
    cdef double maxt, att, attm1
    cdef int inboundary, i, j, k, l, it
    cdef double *p_attimes = &attimes[0]
    cdef double drt = p_attimes[1] - p_attimes[0]
    cdef double *p_occ    = &occ[0]
    cdef float *p_O      = &O[0]

    for i in xrange(g_Ms[0]):
        inboundary = 1
        it = -1
        while inboundary and (it < NT-1):
            it += 1
            att = p_attimes[it]

            if p_O[i] >= att:
                p_occ[it] += 1.
            elif (it > 0):
                attm1 = p_attimes[it-1]
                if (p_O[i] > attm1) and (p_O[i] < att):
                    p_occ[it] += (p_O[i] - attm1) / drt
            else:
                inboundary = 0
                    

def get_obs_exp_v_classic(long[::1] mv_g_Ms, double[::1] mv_expctd, short[::1] mv_chi2_boxes_mk, double low):
    cdef double c_e = 0
    cdef short c_o = 0
    cdef double c_v = 0
    cdef int i0, ii0
    cdef long* p_g_Ms = &mv_g_Ms[0]

    run_o = []
    run_e = []
    run_v = []

    for i0 in xrange(0, p_g_Ms[0]-2, 2):
        for ii0 in xrange(i0, i0+2):
            c_e += mv_expctd[ii0]
            c_o += mv_chi2_boxes_mk[ii0]
        c_v += 2

        if (c_o > low) and (c_e > low):
            run_o.append(c_o)
            run_e.append(c_e)
            run_v.append(c_v)
            c_o = 0
            c_e = 0
            c_v = 0

    return run_o, run_e, run_v



#  quads=[[30, 67], [35, 73]]  # intermediate
@cython.boundscheck(False)
@cython.wraparound(False)
def get_obs_exp_v(g_Ms, float[::1] mv_expctd, short[::1] mv_chi2_boxes_mk, double low, quadrnts=None):
    cdef double c_e = 0
    cdef short c_o = 0
    cdef double c_v = 0
    cdef long i0, i1, i2, i3, ii0, ii1, ii2, ii3
    cdef long[::1] mv_g_Ms = g_Ms
    cdef long* p_g_Ms = &mv_g_Ms[0]
    #cdef double* p_expctd = &mv_expctd[0, 0, 0, 0]
    #cdef int* p_chi2_boxes_mk = &mv_chi2_boxes_mk[0, 0, 0, 0]

    run_o = []
    run_e = []
    run_v = []


    if quadrnts is None:
        quads0 = _N.arange(0, 1);
        lims0  = _N.array([0, g_Ms[0]-1]);
    else:
        quads0 = _N.arange(0, len(quadrnts[0])+1)
        lims0  = _N.zeros(len(quadrnts[0])+2, dtype=_N.int)
        
        lims0[len(quadrnts[0])+1]  = g_Ms[0]-1;        
        for i in xrange(len(quadrnts[0])):  lims0[i+1] = quadrnts[0][i]

    run_os = []
    run_es = []
    run_vs = []

    lftovr_os = []   #  left overs
    lftovr_es = []   #  left overs
    for q0 in quads0:
        run_o = []
        run_e = []
        run_v = []
        for i0 in xrange(lims0[q0], lims0[q0+1]):
            c_e += mv_expctd[i0]
            c_o += mv_chi2_boxes_mk[i0]
            c_v += 1

            if (c_o > low) and (c_e > low):
                run_o.append(c_o)
                run_e.append(c_e)
                run_v.append(c_v)
                c_o = 0
                c_e = 0
                c_v = 0

        #  we calculate CIFs @ marks where model CIF takes large 
        #  value.  if model has predicts small CIF in  regions 
        #  where CIF should be large, c_e be too small and
        #  not trigger the above condition for inclusion into 
        #  run_e array.  
        lftovr_os.append(c_o)
        lftovr_es.append(c_e)
        run_os.append(run_o)
        run_es.append(run_e)
        run_vs.append(run_v)

    """

    for i0 in xrange(0, p_g_Ms[0]-2, 2):
        for i1 in xrange(0, p_g_Ms[1]-2, 2):
            for i2 in xrange(0, p_g_Ms[2]-2, 2):
                for i3 in xrange(0, p_g_Ms[3]-2, 2):
                    for ii0 in xrange(i0, i0+2):
                        for ii1 in xrange(i1, i1+2):
                            for ii2 in xrange(i2, i2+2):
                                for ii3 in xrange(i3, i3+2):
                                    c_e += mv_expctd[ii0, ii1, ii2, ii3]
                                    c_o += mv_chi2_boxes_mk[ii0, ii1, ii2, ii3]

                    c_v += 16
                    if (c_o > low) and (c_e > low):
                        run_o.append(c_o)
                        run_e.append(c_e)
                        run_v.append(c_v)
                        c_o = 0
                        c_e = 0
                        c_v = 0

    """


    return run_os, run_es, run_vs, lftovr_os, lftovr_es
