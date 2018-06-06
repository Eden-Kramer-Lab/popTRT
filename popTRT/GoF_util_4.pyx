cimport cython
import numpy as _N
cimport numpy as _N
from libc.stdio cimport printf
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_volrat4(long g_T, long[::1] g_Ms, float[:, :, :, ::1] O, double[::1] trngs, float[:, :, :, ::1] volrat_mk):
#def calc_volrat4(long g_T, long[::1] g_Ms, double[:, :, :, ::1] O, double[::1] trngs, double[:, :, :, ::1] volrat_mk):
    #  changes in rescaled-time direction is abrupt, while over marks may not be so abrupt.  Cut box in mark direction in 4 
    cdef double tL, tH
    cdef double d1h, d2h, d3h, d4h, d1l, d2l, d3l, d4l
    cdef long ti, inside, outside, border
    cdef long m1, m2, m3, m4
    #cdef double tmp
    cdef float tmp

    cdef long   *p_g_Ms  = &g_Ms[0]
    cdef float *p_O     = &O[0, 0, 0, 0]
    cdef double *p_trngs     = &trngs[0]
    cdef float *p_volrat_mk = &volrat_mk[0, 0, 0, 0]
    #cdef double *p_volrat_mk = &volrat_mk[0, 0, 0, 0]

    cdef long it, inboundary, i_here, i_til_end

    cdef long g_M3 = g_Ms[3]
    cdef long g_M2 = g_Ms[3]*g_Ms[2]
    cdef long g_M1 = g_Ms[3]*g_Ms[2]*g_Ms[1]


    cdef long    g_Tm1 = g_T-1
    cdef long g_Mm1 = g_Ms[3]-1
    cdef long g_Mm1_2 = (g_Ms[2]-1) * g_Mm1
    cdef long g_Mm1_3 = (g_Ms[1]-1) * g_Mm1_2

    inside  = 0
    outside = 0
    border  = 0
    cdef double p01, p11, p12, p13, p14, p21, p22, p23, p24, p25, p26, p31, p32, p33, p34, p41

    
    cdef long _m1, _m2, _m3
    cdef long _m1p1, _m2p1, _m3p1, m4p1

    cdef int i_start_search 

    for m1 in xrange(p_g_Ms[0]-1):
        #print "m1  %d" % m1
        _m1 = m1*g_M1
        _m1p1 = (m1+1)*g_M1
        for m2 in xrange(p_g_Ms[1]-1):
            _m2 = m2*g_M2
            _m2p1 = (m2+1)*g_M2
            for m3 in xrange(p_g_Ms[2]-1):
                _m3 = m3*g_M3
                _m3p1 = (m3+1)*g_M3
                for m4 in xrange(p_g_Ms[3]-1):
                    m4p1 = m4+1
                    inboundary = 1

                    it = -1
                    p01 = p_O[_m1 + _m2 + _m3+ m4]
                    
                    p11 = p_O[_m1p1 + _m2 + _m3+ m4]
                    p12 = p_O[_m1 + _m2p1 + _m3+ m4]
                    p13 = p_O[_m1 + _m2 + _m3p1+ m4]
                    p14 = p_O[_m1 + _m2 + _m3+ m4p1]

                    p21 = p_O[_m1p1 + _m2p1 + _m3+ m4]     # 1 2
                    p22 = p_O[_m1p1 + _m2 + _m3p1+ m4]     # 1 3
                    p23 = p_O[_m1p1 + _m2 + _m3+ m4p1]     # 1 4
                    p24 = p_O[_m1 + _m2p1 + _m3p1+ m4]     # 2 3
                    p25 = p_O[_m1 + _m2p1 + _m3+ m4p1]     # 2 4
                    p26 = p_O[_m1 + _m2 + _m3p1+ m4p1]     # 3 4

                    # 3  
                    p31 = p_O[_m1p1 + _m2p1 + _m3p1+ m4]   # 1 2 3
                    p32 = p_O[_m1p1 + _m2p1 + _m3+ m4p1]   # 1 2 4
                    p33 = p_O[_m1p1 + _m2 + _m3p1+ m4p1]   # 1 3 4
                    p34 = p_O[_m1 + _m2p1 + _m3p1+ m4p1]   # 2 3 4

                    # 4
                    p41 = p_O[_m1p1 + _m2p1 + _m3p1+ m4p1] # 1 2 3 4

                    while (it < g_T-2) and (inboundary == 1):
                        it += 1

                        tL = p_trngs[it]
                        tH = p_trngs[it+1]

                        d01h = tH - p01

                        #  1   
                        d11h = tH - p11
                        d12h = tH - p12
                        d13h = tH - p13
                        d14h = tH - p14

                        #  2   
                        d21h = tH - p21
                        d22h = tH - p22
                        d23h = tH - p23
                        d24h = tH - p24
                        d25h = tH - p25
                        d26h = tH - p26

                        # 3  
                        d31h = tH - p31
                        d32h = tH - p32
                        d33h = tH - p33
                        d34h = tH - p34

                        # 4
                        d41h = tH - p41

                        ###################################3
                        d01l = p01 - tL

                        #  1   
                        d11l = p11 - tL
                        d12l = p12 - tL
                        d13l = p13 - tL
                        d14l = p14 - tL

                        #  2   
                        d21l = p21 - tL
                        d22l = p22 - tL
                        d23l = p23 - tL
                        d24l = p24 - tL
                        d25l = p25 - tL
                        d26l = p26 - tL

                        # 3  
                        d31l = p31 - tL
                        d32l = p32 - tL
                        d33l = p33 - tL
                        d34l = p34 - tL

                        # 4
                        d41l = p41 - tL
                        
                        if (((d01h > 0) or \
                             (d11h > 0) or (d12h > 0) or (d13h > 0) or (d14h > 0) or \
                             (d21h > 0) or (d22h > 0) or (d23h > 0) or (d24h > 0) or (d25h > 0) or (d26h > 0) or \
                             (d31h > 0) or (d32h > 0) or (d33h > 0) or (d34h > 0) or \
                             (d41h > 0)) and
                            ((d01l > 0) or \
                             (d11l > 0) or (d12l > 0) or (d13l > 0) or (d14l > 0) or \
                             (d21l > 0) or (d22l > 0) or (d23l > 0) or (d24l > 0) or (d25l > 0) or (d26l > 0) or \
                             (d31l > 0) or (d32l > 0) or (d33l > 0) or (d34l > 0) or \
                             (d41l > 0))):
                            tmp = 0.0625 * ((d01l + d11l + d12l + d13l + d14l + d21l + d22l + d23l + d24l + d25l + d26l + d31l + d32l + d33l + d34l + d41l) / (tH - tL))
                            tmp = 0 if (tmp < 0) else tmp
                            tmp = 1 if (tmp > 1) else tmp

                            p_volrat_mk[m1*g_Mm1_3+ m2*g_Mm1_2 + m3*g_Mm1 + m4] += tmp
                        elif (d01h < 0) and \
                               (d11h<0) and (d12h<0) and (d13h<0) and (d14h<0) and \
                               (d21h<0) and (d22h<0) and (d23h<0) and (d24h<0) and (d25h<0) and (d26h<0) and \
                                 (d31h<0) and (d32h<0) and (d33h<0) and (d34h<0) and \
                                 (d41h < 0):
                                p_volrat_mk[m1*g_Mm1_3+ m2*g_Mm1_2 + m3*g_Mm1 + m4] += 1
                        else:
                            inboundary = 0

    return inside, outside, border
                    
@cython.boundscheck(False)
@cython.wraparound(False)
def find_Occ4(long[::1] g_Ms, int NT, double[::1] attimes, double[::1] occ, float[:, :, :, ::1] O):
    #  at given rescaled time, number of mark voxels < boundary (rescaled time)
    cdef double maxt, att
    cdef long inboundary, i, j, k, l, it
    cdef double *p_attimes = &attimes[0]
    cdef double *p_occ    = &occ[0]
    cdef float *p_O      = &O[0, 0, 0, 0]

    cdef long ig_M1, jg_M2, kg_M3, g_M4, g_M1, g_M3, g_M2

    g_M1 = g_Ms[1]*g_Ms[2]*g_Ms[3]
    g_M2 = g_Ms[2]*g_Ms[3]
    g_M3 = g_Ms[3]

    with nogil:
        for i in xrange(g_Ms[0]):
            ig_M1 = i*g_M1
            for j in xrange(g_Ms[1]):
                jg_M2 = j*g_M2
                for k in xrange(g_Ms[2]):
                    kg_M3 = k*g_M3
                    for l in xrange(g_Ms[3]):
                        inboundary = 1
                        it = -1
                        while inboundary and (it < NT-1):
                            it += 1
                            att = p_attimes[it]

                            if p_O[ig_M1 + jg_M2 + kg_M3 + l] >= att:
                                p_occ[it] += 1.
                            else:
                                inboundary = 0
                    

#  quads=[[30, 67], [35, 73]]  # intermediate
@cython.boundscheck(False)
@cython.wraparound(False)
def get_obs_exp_v(g_Ms, float[:, :, :, ::1] mv_expctd, char[:, :, :, ::1] mv_chi2_boxes_mk, double low, quadrnts=None):
#def get_obs_exp_v(long[::1] mv_g_Ms, double[:, :, :, ::1] mv_expctd, short[:, :, :, ::1] mv_chi2_boxes_mk, double low):
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
        quads0 = _N.arange(0, 1);  quads1 = _N.arange(0, 1)
        quads2 = _N.arange(0, 1);  quads3 = _N.arange(0, 1)
        lims0  = _N.array([0, g_Ms[0]-1]);   lims1  = _N.array([0, g_Ms[1]-1])
        lims2  = _N.array([0, g_Ms[2]-1]);   lims3  = _N.array([0, g_Ms[3]-1])
    else:
        quads0 = _N.arange(0, len(quadrnts[0])+1)
        quads1 = _N.arange(0, len(quadrnts[1])+1)
        quads2 = _N.arange(0, len(quadrnts[2])+1)
        quads3 = _N.arange(0, len(quadrnts[3])+1)
        lims0  = _N.zeros(len(quadrnts[0])+2, dtype=_N.int)
        lims1  = _N.zeros(len(quadrnts[1])+2, dtype=_N.int)
        lims2  = _N.zeros(len(quadrnts[2])+2, dtype=_N.int)
        lims3  = _N.zeros(len(quadrnts[3])+2, dtype=_N.int)

        
        lims0[len(quadrnts[0])+1]  = g_Ms[0]-1;        
        lims1[len(quadrnts[1])+1]  = g_Ms[1]-1
        lims2[len(quadrnts[2])+1]  = g_Ms[2]-1;        
        lims3[len(quadrnts[3])+1]  = g_Ms[3]-1
        for i in xrange(len(quadrnts[0])):  lims0[i+1] = quadrnts[0][i]
        for i in xrange(len(quadrnts[1])):  lims1[i+1] = quadrnts[1][i]
        for i in xrange(len(quadrnts[2])):  lims2[i+1] = quadrnts[2][i]
        for i in xrange(len(quadrnts[3])):  lims3[i+1] = quadrnts[3][i]

    """
    for q0 in quads0:
        for q1 in quads1:
            print "new quad  %(1)d %(2)d" % {"1" : q0, "2" : q1}
            for i0 in xrange(lims0[q0], lims0[q0+1]):
                for i1 in xrange(lims1[q1], lims1[q1+1]):
                    print "---  %(1)d %(2)d" % {"1" : i0, "2" : i1}
    """

    run_os = []
    run_es = []
    run_vs = []

    lftovr_os = []   #  left overs
    lftovr_es = []   #  left overs
    for q0 in quads0:
        for q1 in quads1:
            for q2 in quads2:
                for q3 in quads3:
                    run_o = []
                    run_e = []
                    run_v = []
                    for i0 in xrange(lims0[q0], lims0[q0+1]-2, 2):
                        for i1 in xrange(lims1[q1], lims1[q1+1]-2, 2):
                            for i2 in xrange(lims2[q2], lims2[q2+1]-2, 2):
                                for i3 in xrange(lims3[q3], lims3[q3+1]-2, 2):
                                    for ii0 in xrange(i0, i0+2):
                                        for ii1 in xrange(i1, i1+2):
                                            for ii2 in xrange(i2, i2+2):
                                                for ii3 in xrange(i3, i3+2):
                                                    c_e += mv_expctd[ii0, ii1, ii2, ii3]
                                                    c_o += mv_chi2_boxes_mk[ii0, ii1, ii2, ii3]
                                    c_v += 16

                                    # power analysis on chisquare test:
                                    # 6% rejection @ 5% level for expctd >= 6.  
                                    if c_e > low:
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



#  quads=[[30, 67], [35, 73]]  # intermediate
@cython.boundscheck(False)
@cython.wraparound(False)
def get_obs_exp_v_classic(g_Ms, float[:, :, :, ::1] mv_expctd, char[:, :, :, ::1] mv_chi2_boxes_mk, double low):
#def get_obs_exp_v(long[::1] mv_g_Ms, double[:, :, :, ::1] mv_expctd, short[:, :, :, ::1] mv_chi2_boxes_mk, double low):
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
                    # performed power analysis on chisquare test.
                    #  6% rejection a 5% level for expected > 6.  
                    if c_e > low:
                        run_o.append(c_o)
                        run_e.append(c_e)
                        run_v.append(c_v)
                        c_o = 0
                        c_e = 0
                        c_v = 0

    return run_o, run_e, run_v
