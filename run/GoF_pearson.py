##  decode script
import scipy.stats as _ss
import time as _tm
import utilities as _U

def group(run_os, run_es, these):
    """
    group together several quadrants and test them together
    """
    grpd_e = []
    grpd_o = []

    for nq in these:
        grpd_e.extend(run_es[nq])
        grpd_o.extend(run_os[nq])

    reslt   = _ss.chisquare(grpd_e, grpd_o)
    print reslt
    return grpd_o, grpd_e

def out_quad(these):
    """
    write out quadrant settings that produce regions of good/poor fit
    """
    fp = open("%s/quadrants" % outdirN, "w+")
    fp.write("#  quadrants that differentiate regions of good / poor fit\n")
    fp.write("#  g_Ms:   %s\n" % str(g_Ms))
    fp.write("quadrnts=%s\n" % str(these))
    fp.close()

#  use string provided in file "outdirN"

def run(fquadrnts=None):
    quadrnts   = None
    n_qdrnts   = 1
    if fquadrnts is not None:
        if K == 4:
            quadrnts  = [[int(g_Ms[0]*fquadrnts[0][0])], [int(g_Ms[1]*fquadrnts[1][0])], [int(g_Ms[2]*fquadrnts[2][0])], [int(g_Ms[3]*fquadrnts[3][0])]]
            n_qdrnts = (len(quadrnts[0])+1) * (len(quadrnts[1])+1) * (len(quadrnts[2])+1) * (len(quadrnts[3])+1)
        elif K == 1:
            quadrnts  = [[int(g_Ms[0]*fquadrnts[0][0])]]
            n_qdrnts = (len(quadrnts[0])+1)

    run_os, run_es, run_vs, lftovr_os, lftovr_es = _Gu.get_obs_exp_v(g_Ms, expctd, chi2_boxes_mk, 6., quadrnts=quadrnts)

    fp = open("%(df)s/exp_obs%(qd)s.txt" % {"df" : outdirN, "qd" : n_qdrnts}, "w+")
    fp.write("#  quadrants %s\n" % str(fquadrnts))
    fp.write("#  ks D, pv\n")

    fp.write("%(d).3e  %(pv).3e -1\n" % {"d" : ks_D, "pv" : ks_pv})

    fp.write("#  chi2, pv, dg freedom  # quadrnt#, \n")
    for nq in xrange(n_qdrnts):
        reslt   = _ss.chisquare(run_os[nq], run_es[nq])
        fp.write("%(chi2).3e  %(pv).3e  %(n)d  #  nq %(nq)d\n" % {"chi2" : reslt[0], "pv" : reslt[1], "n" : len(run_es[nq]), "nq" : nq})
    fp.close()

    lftovr = _N.empty((n_qdrnts, 2))
    lftovr[:, 0] = lftovr_os
    lftovr[:, 1] = lftovr_es

    _U.savetxtWCom("%(df)s/lftovrs%(qd)d.txt" % {"df" : outdirN, "qd" : n_qdrnts}, lftovr, fmt="%d %.3f", delimiter=" ", com=("# observed, expeceted, quadrants %s" % str(fquadrnts)))

#outdirN="dec-bond0906-8_2-1/decMoG_22_0/241_smpsPsd=3.0_sds2use=4.0"
#outdirN="dec-bond0906-8_2-1/decMoG_18_0/241_smpsPsd=3.0_sds2use=4.0"

lm = depickle("%s/vars.dmp" % outdirN)
K              = lm[0]
nspks          = lm[1]
g_Ms           = lm[2]
rscldA         = lm[3]
dV             = lm[4]
dt             = lm[5]
totalVol       = lm[6]
normlz         = lm[7]
mrngs          = lm[8]
ks_D           = lm[9]
ks_pv          = lm[10]

#f_volrat_mk    = _N.fromfile("%s/volrat.bin" % outdirN, dtype=_N.float32)

# if K == 1:
#     import GoF_util as _Gu
#     volrat_mk = f_volrat_mk
# elif K == 2:
#     import GoF_util_2 as _Gu    
#     volrat_mk = f_volrat_mk.reshape(g_Ms[0]-1, g_Ms[1] - 1)
# elif K == 4:
#     import GoF_util_4 as _Gu    
#     volrat_mk = f_volrat_mk.reshape(g_Ms[0]-1, g_Ms[1]-1, g_Ms[2]-1, g_Ms[3]-1)

if K >= 1:
    d_mk0 = mrngs[0, 1] - mrngs[0, 0]
if K >= 2:
    d_mk1 = mrngs[1, 1] - mrngs[1, 0]
if K >= 4:
    d_mk2 = mrngs[2, 1] - mrngs[2, 0]
    d_mk3 = mrngs[3, 1] - mrngs[3, 0]


if K == 1:
    chi2_boxes_mk       = _N.zeros((g_Ms[0]-1), dtype=_N.int16)   #  
elif K == 2:
    chi2_boxes_mk       = _N.zeros((g_Ms[0]-1, g_Ms[1]-1), dtype=_N.int8)   #  
elif K == 4:
    chi2_boxes_mk       = _N.zeros((g_Ms[0]-1, g_Ms[1]-1, g_Ms[2]-1, g_Ms[3]-1), dtype=_N.int8)   #  

out_of_bounds = []
for ns in xrange(nspks):
    t  = rscldA[ns, 0]
    mk = rscldA[ns, 1:]

    if K == 1:
        im0 = int((mk[0] - mrngs[0, 0])/d_mk0)
        chi2_boxes_mk[im0] += 1
    elif K == 2:
        im0 = int((mk[0] - mrngs[0, 0])/d_mk0)
        im1 = int((mk[1] - mrngs[1, 0])/d_mk1)
        chi2_boxes_mk[im0, im1] += 1
    elif K == 4:
        im0 = int((mk[0] - mrngs[0, 0])/d_mk0)
        im1 = int((mk[1] - mrngs[1, 0])/d_mk1)
        im2 = int((mk[2] - mrngs[2, 0])/d_mk2)
        im3 = int((mk[3] - mrngs[3, 0])/d_mk3)

        if (im0 < g_Ms[0] - 1) and (im1 < g_Ms[1] - 1) and (im2 < g_Ms[2] - 1) and (im3 < g_Ms[3] - 1):
            chi2_boxes_mk[im0, im1, im2, im3] += 1
        else:
            out_of_bounds.append(_N.array(mk))
            #  find where in mark-space, rescaled time spike lies.

#print "##################### chi2_boxes %.4f" % (tt6-tt5)

expctd = normlz*dV*dt * volrat_mk
volrat_mk = None  



#fquadrnts  = [[0.75]]
#fquadrnts = None

run()
fquadrnts = [[0.4], [0.4], [0.4], [0.4]]
run(fquadrnts=fquadrnts)
