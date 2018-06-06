##  decode script
import GoFfuncs_pyxKDE as _goff
#import GoFfuncs_pyxAd as _goff
import pickle as _pkl
import mcmcFigs as mF
import os
import decodeutil as _du
import manypgs as _mpgs
import time as _tm
import shutil
import tmrsclTest as tmrt
import scipy.stats as _ss
import gc
import EnDedirs as _edd
import utilities as _U


#_plt.ioff()    
_KDE_RT     = False   #  REALTIME:  data+behavior to be decoded used to train

fmt        = "png"
thnspks    = 4


anim       = "bond" # bn2
day        = "09"
ep         = "06"
"""
anim       = "bond" # bn2
day        = "03"
ep         = "02"

anim       = "frank" # bn2
day        = "06"
ep         = "02"
"""
rotate     = False
dt         = 0.001
#tets       = [0, 1, 2, 3, 5, 6, 8, 11, 12, 14, 17, 19]#[19]#, 1, 2, 3, 4]  # set
#tets       = [ 0,  1,  5,  6, 17, 20, 21]  #  0302 - smlChg
#tets        = [ 2,  3,  8, 11, 12, 14, 19]  # 0302  - lrgChg
#tets        = [ 19, ]  # 0302  - lrgChg

#tets       = [ 1,  2,  6,  9, 12, 14, 15, 17, 18, 19, 20]   # 0906 - smlChg
#tets       = [ 4,  8, 10, 11]   # 0906  - lrgChg
#tets        = [1, 2, 4, 6, 8, 9, 10, 11, 12, 14, 15, 17, 18, 19, 20] # 090206 all
tets        = [3]
#tets         = None

usemaze    = _goff.mz_W

#anim = "2us4"
#anim        = "2bs3"
#anim        = "8bn13"
#anim        = "2un9"
#anim       = "bn11"
#anim       = "8un3"
#anim        = "3bn8"
#anim        = "2us3"
#anim        = "un25"
#anim       = "bn15"
#anim       = "lt1"
#anim       = "2abn19"
#anim       = "4bn5"
#anim       = "bn7"
"""
day        = None
ep         = None
tets       = None 
"""

#usemaze    = _goff.mz_CRCL

#tets       = [0,3]  #  set

#itvfn    = "itv2_10" 
itvfn   = "itv8_2"
#itvfn   = "itv2_3"
#itvfn   = "itv1"

saveBin  = 1    #  decode @ 1ms, but we save it at lower resolution

lagfit     = 0   # normally set to 1.  But set larger if we want to use very old fit to decode

#EPCHS = 7
EPCHS = 1
K     = 4

Nx = 241

if K == 1:
    import GoF_util as _Gu
elif K == 2:
    import GoF_util_2 as _Gu    
elif K == 4:
    import GoF_util_4 as _Gu    

label     =  "1"   #  refers to

bUseNiC    = True   # nn-informative cluster
#datfn      = ["mdec%s" % outdir]
#method     = _du._KDE
method     = _du._MOG
#method     = _du._GT

if K > 1:
    smpsPerSD = 1.8 if method == _du._KDE else 3.0  #  1. too low, but this also probably depends on # of data points
else:
    smpsPerSD = 3. if method == _du._KDE else 8.0  #  1. too low, but this also probably depends on # of data points
#smpsPerSD = 2. if method == _du._KDE else 3.0  #  1. too low, but this also probably depends on # of data points
#smpsPerSD = 2. if method == _du._KDE else 3.0  #  1. too low, but this also probably depends on # of data points
sds_to_use = 4.5 if method == _du._KDE else 4.  #  1. too low, but this also probably depends on # of data points
#sds_to_use = 5 if method == _du._KDE else 4.  #  1. too low, but this also probably depends on # of data points

xLo   = -6;  xHi   = 6

bx=0.05   #  occ density smoothing
Bx=0.05
Bm=0.16
bx2=bx*bx
ibx2=1./bx2

_RAN_PRTRB = 0
_CONST_PRTRB = 1

bPrtrb = False    # perturb
prtrbMd  = _CONST_PRTRB   
prtrbF   = 0.6

sPrtrb = "" if not bPrtrb else "_prtrbd"

_datfns, resdns, outdir, tetstr = _du.makefilenames(anim, day, ep, tets, itvfn, label=label, ripple=False)
datfn = _datfns[0]

#  i also want in 
#  Nx=,bx=,Bx=,Bm=/smpsPerSD=,sds2use=  if KDE

sRMC = ""
rm_clstrs = None
if rm_clstrs is not None:
    sRMC = ",rm_%s" % str(rm_clstrs)
if method == _du._KDE:
    if _KDE_RT:
        outdirN = "%(d)s/decKDE_RT%(ts)s_%(lf)d_%(Nx)d_%(Bx).2f_%(Bm).2f_%(bx).2f/smpsPsd=%(spsd).1f_sds2use=%(sds).1f" % {"d" : outdir, "ts" :tetstr, "lf" : lagfit, "Bx" : Bx, "Bm" : Bm, "bx" : bx, "Nx" : Nx, "spsd" : smpsPerSD, "sds" : sds_to_use}
    else:
        outdirN = "%(d)s/decKDE%(ts)s_%(lf)d_%(Nx)d_%(Bx).2f_%(Bm).2f_%(bx).2f/smpsPsd=%(spsd).1f_sds2use=%(sds).1f" % {"d" : outdir, "ts" :tetstr, "lf" : lagfit, "Bx" : Bx, "Bm" : Bm, "bx" : bx, "Nx" : Nx, "spsd" : smpsPerSD, "sds" : sds_to_use}
elif method == _du._MOG:
    outdirN = "%(d)s/decMoG%(ts)s_%(lf)d/%(Nx)d_smpsPsd=%(spsd).1f_sds2use=%(sds).1f%(rmc)s" % {"d" : outdir, "ts" :tetstr, "lf" : lagfit, "Nx" : Nx, "spsd" : smpsPerSD, "sds" : sds_to_use, "rmc" : sRMC}
else:
    outdirN = "%(d)s/decGT%(ts)s_%(lf)d/%(Nx)d_smpsPsd=%(spsd).1f_sds2use=%(sds).1f%(rmc)s" % {"d" : outdir, "ts" :tetstr, "lf" : lagfit, "Nx" : Nx, "spsd" : smpsPerSD, "sds" : sds_to_use, "rmc" : sRMC}

#if not os.access(outdir, os.F_OK):
_edd.recurseMkdir(outdirN)
    #os.mkdir(outdir)
#if not os.access(outdirN, os.F_OK):
#    os.mkdir(outdirN)
shutil.copyfile("GoF.py", "%s/GoF.py" % outdirN)



#0, 4, 6, 9, 14

dd    = os.getenv("__EnDeDataDir__")
_dat   = _N.loadtxt("%(dd)s/%(dfn)s.dat" % {"dd" : dd, "dfn" : datfn})  # DATA/anim_mdec_dyep_tet.dat
dat   = _dat[:, 0:2+K]

intvs = _N.array(_N.loadtxt("%(dd)s/%(iv)s.dat" % {"dd" : dd, "iv" : itvfn})*dat.shape[0], dtype=_N.int)
#intvs[-1] = 292741    # hack for frank0606

if method == _du._GT:
    _all_gt_prms  = []
    with open("%(dd)s/%(dfn)s_prms.pkl" % {"dd" : dd, "dfn" : datfn}, "rb") as f:
        gtprms = _pkl.load(f)
        gt_Dt    = gtprms["intv"]   #  how often gtprms sampled
    kms = gtprms["km"]    
    Mgt  = max([item for sublist in kms for item in sublist]) + 1  #  if max index is m, # of clusters is m+1

for dn in resdns:   #  1 result dir for each tetrode
    #dfn = "%(tet)s-%(itv)s" % {"tet" : nt, "itv" : itv_runN}

    if method == _du._MOG:
        prms_4_each_epoch_all_cl = []
        for e in xrange(0, EPCHS-lagfit):
            with open("%(d)s/posteriors_%(e)d.dmp" % {"d" : dn, "e" : e}, "rb") as f:
                lm = _pkl.load(f)
                f.close()

            M            = lm["M"]

            inuse_clstr = _N.where(lm["freeClstr"] == False)[0]
            M            = len(inuse_clstr)
            Mwowonz      = M
            spmk_us      = _N.zeros((Mwowonz, K+1))          
            spmk_covs    = _N.zeros((Mwowonz, K+1, K+1))
            spmk_l0s     = _N.zeros((Mwowonz))


            mkprms       = lm["mk_prmPstMd"]
            #mkprms       = [_N.median(lm["smp_mk_prms"][0][:, 2000:], axis=1).T, _N.median(lm["smp_mk_prms"][1][:, :, 2000:], axis=2).T]
            l0s          = lm["sp_prmPstMd"][::3][inuse_clstr]
            fs           = lm["sp_prmPstMd"][1::3][inuse_clstr]
            q2s          = lm["sp_prmPstMd"][2::3][inuse_clstr]
            nanq2s       = _N.where(_N.isnan(q2s))   #  old version set q2s to nan
            if len(nanq2s) > 0:
                q2s[nanq2s]  = 10000.

            us           = mkprms[0][inuse_clstr]
            covs         = mkprms[1][inuse_clstr]
            # for m in xrange(M):
            #     if covs[m, K-1, K-1] > 10:
            #         covs[m, K-1, K-1] = 0.01   ##  really large variance last element.  Why?
            # if Mwowonz > M:
            #     us[M]    = lm["nz_u"]
            #     covs[M]    = lm["nz_Sg"]

            ix = _N.where(l0s <= 0)
            l0s[ix] = 0.001
            if rm_clstrs is not None:    #  effect of removed clusters
                l0s[rm_clstrs] = 0.00001

            if not bPrtrb:
                rnds = _N.ones(M)
                q2s  *= rnds

            if bPrtrb:
                rnds = (0.3+1.7*_N.random.rand(M))
            spmk_l0s[:] = l0s*rnds
            spmk_us[:, 0] = fs
            spmk_us[:, 1:] = us
            if bPrtrb:
                rnds = (0.3+1.7*_N.random.rand(M))
            spmk_covs[:, 0, 0] = q2s  #  us, covs are pmdim
            spmk_covs[:, 1:, 1:] = covs
            prms_4_each_epoch_all_cl.append([spmk_l0s, us, covs, fs, q2s, spmk_us, spmk_covs])
    if method == _du._GT:
        gt_us      = _N.zeros((Mgt, K+1))          
        gt_covs    = _N.zeros((Mgt, K+1, K+1))
        gt_l0s     = _N.zeros((Mgt))

        if not bPrtrb:
            rnds = _N.ones(Mgt)

        l = 0
        for epc in xrange(EPCHS-lagfit):
            t0 = intvs[epc] / gt_Dt  #  decode times
            t1 = intvs[epc+1] / gt_Dt

            if bPrtrb:
                if prtrbMd == _RAN_PRTRB:
                    rnds = _N.ones(Mgt)*(0.3+1.7*_N.random.rand(Mgt))
                else:
                    rnds = _N.ones(Mgt)*prtrbF
                    #rnds = _N.array([0.6, 1.6])
            #gt_l0s[epc:, 0] = _N.mean(gtprms["l0"][:, t0:t1], axis=1)*rnds
            #gt_l0s = _N.mean(gtprms["l0"][:, t0:t1], axis=1)*_N.array([1.5, 0.5])
            gt_l0s = _N.mean(gtprms["l0"][:, t0:t1], axis=1)*_N.array([1.1, 0.9])
            #gt_l0s = _N.mean(gtprms["l0"][:, t0:t1], axis=1)*_N.array([1.6, 0.4
#])
            #gt_l0s = _N.mean(gtprms["l0"][:, t0:t1], axis=1)
            ###########
            j_fs   = _N.mean(gtprms["f"][:, t0:t1], axis=1)
            j_q2   = _N.mean(gtprms["sq2"][:, t0:t1], axis=1)
            #  per neuron, not place field
            j_us_pn   = _N.mean(gtprms["u"][:, t0:t1], axis=1)  
            j_covs_pn = gtprms["covs"]
            j_us   = _N.empty((Mgt, K))
            j_covs   = _N.empty((Mgt, K, K))


            ###########
            for cl in xrange(len(kms)):
                pfis = kms[cl]   #  place fields of each cell
                for ipf in pfis:
                    # gt_us[epc, ipf, 0]  = j_fs[ipf, ]
                    # gt_us[epc, ipf, 1:] = j_us[cl]

                    # gt_covs[epc, ipf, 0, 0] = j_q2[ipf, ]
                    # gt_covs[epc, ipf, 1:, 1:] = j_covs[cl]
                    gt_us[ipf, 0]  = j_fs[ipf, ]
                    gt_us[ipf, 1:] = j_us_pn[cl]

                    gt_covs[ipf, 0, 0] = j_q2[ipf, ]
                    gt_covs[ipf, 1:, 1:] = j_covs_pn[cl]
                    j_us[ipf]  = j_us_pn[cl]
                    j_covs[ipf] = j_covs_pn[cl]

            _all_gt_prms.append([gt_l0s, j_us, j_covs, j_fs, j_q2, gt_us, gt_covs])

    if method == _du._KDE:
        _all_kde_prms  = []

        for epc in xrange(EPCHS-lagfit):
            t0 = intvs[epc]
            t1 = intvs[epc+1]

            sts   = _N.where(_dat[t0:t1, 1] == 1)[0]
            Mkde  = len(sts)
            kde_us      = _N.zeros((Mkde, K+1))          
            kde_covs    = _N.zeros((Mkde, K+1, K+1))
            for mk in xrange(Mkde):
                _N.fill_diagonal(kde_covs[mk, 1:, 1:], Bm*Bm)
                kde_covs[mk, 0, 0] = Bx*Bx

            #kde_l0s     = _N.ones((Mkde)) * (1./((t1-t0)*dt))  #  one should contribute 1spk/total time
            kde_l0s     = _N.ones((Mkde))  #  one should contribute 1spk/total time
            ###########
            j_fs   = _N.array(_dat[sts, 0])
            j_q2   = _N.ones(Mkde)*Bx*Bx
            j_us   = _N.array(_dat[sts, 2:])
            kde_us[:, 0] = j_fs
            kde_us[:, 1:] = j_us

            j_covs = _N.zeros((Mkde, K, K))
            for mk in xrange(Mkde):
                _N.fill_diagonal(j_covs[mk], Bm*Bm)

            # ###########
            # for cl in xrange(len(kms)):
            #     pfis = kms[cl]   #  place fields of each cell
            #     for ipf in pfis:
            #         kde_us[ipf, 0]  = j_fs[ipf, ]
            #         kde_us[ipf, 1:] = j_us[cl]

            #         kde_covs[ipf, 0, 0] = j_q2[ipf, ]
            #         kde_covs[ipf, 1:, 1:] = j_covs[cl]

            _all_kde_prms.append([kde_l0s, j_us, j_covs, j_fs, j_q2, kde_us, kde_covs])

mkd   = _goff.GoFfuncs(kde=(method == _du._KDE), K=K, mkfn=datfn, xLo=xLo, xHi=xHi, maze=usemaze, spdMult=0.1, Nx=Nx, rotate=rotate)

#  we should be adding another axis for tetrode
#silenceLklhds = _N.empty((mkd.Nx, EPCHS-lagfit))


for epch in xrange(0, 1):
    t0 = intvs[epch]   #  decode times
    t1 = intvs[epch+1]
    tt0 = _tm.time()

    sts   = _N.where(mkd.mkpos[0:t1, 1] == 1)[0]
    obsvd_mks = mkd.mkpos[sts, 2:]

    print "##################### rescale spikes"
    if (method == _du._MOG): ################  MOG
        maxfr = prms_4_each_epoch_all_cl[epch-lagfit][0]/ _N.sqrt(2*_N.pi*prms_4_each_epoch_all_cl[epch-lagfit][4])
        hf_cl = _N.where(maxfr > 0.01)[0]   # high firing clusters
        prms = [prms_4_each_epoch_all_cl[epch-lagfit][0][hf_cl], prms_4_each_epoch_all_cl[epch-lagfit][1][hf_cl], prms_4_each_epoch_all_cl[epch-lagfit][2][hf_cl], prms_4_each_epoch_all_cl[epch-lagfit][3][hf_cl], prms_4_each_epoch_all_cl[epch-lagfit][4][hf_cl], prms_4_each_epoch_all_cl[epch-lagfit][5][hf_cl], prms_4_each_epoch_all_cl[epch-lagfit][6][hf_cl]]
    if (method == _du._GT): #################  GT
        prms = [_all_gt_prms[epch-lagfit][0], _all_gt_prms[epch-lagfit][1], _all_gt_prms[epch-lagfit][2], _all_gt_prms[epch-lagfit][3], _all_gt_prms[epch-lagfit][4], _all_gt_prms[epch-lagfit][5], _all_gt_prms[epch-lagfit][6]]
    if (method == _du._KDE): ################  KDE
        prms = [_all_kde_prms[epch-lagfit][0], _all_kde_prms[epch-lagfit][1], _all_kde_prms[epch-lagfit][2], _all_kde_prms[epch-lagfit][3], _all_kde_prms[epch-lagfit][4], _all_kde_prms[epch-lagfit][5], _all_kde_prms[epch-lagfit][6]]

    sp_occ = None

    pos  = _dat[t0:t1, 0]
    posr = pos.reshape(pos.shape[0], 1)

    #  
    dm         = mkd.xp[1] - mkd.xp[0]
    spc_occ    = (1/_N.sqrt(2*_N.pi*bx2))*_N.sum(_N.exp(-0.5*ibx2*(mkd.xpr - posr)**2), axis=0)
    i_spc_occ_dt  = 1./(spc_occ*dt)

    if (method == _du._MOG) or (method == _du._GT):
        i_spc_occ_dt = _N.zeros(Nx)
        rscldA = mkd.rescale_spikes(prms, t0, t1, i_spc_occ_dt, kde=False)   # epch-1 means use fit from previous epoch
        #  rscldA[:, 0] is not monotonic
        g_Ms, mrngs = mkd.getGridDims(method, prms, smpsPerSD, sds_to_use, obsvd_mks)
    elif method == _du._KDE:
        if _KDE_RT:
            mkd.prepareDecKDE(0, t0)
        else:
            mkd.prepareDecKDE(0, t1)

        g_Ms, mrngs = mkd.getGridDims(method, prms, smpsPerSD, sds_to_use, obsvd_mks)

        rscldA = mkd.rescale_spikes(prms, t0, t1, i_spc_occ_dt, kde=True)   # epch-1 means use fit from previous epoch


    # b2Fine = True
    # while b2Fine:
    #     g_Ms, mrngs = mkd.getGridDims(method, prms, smpsPerSD, sds_to_use, obsvd_mks)
    #     if _N.product(g_Ms) > 330**4:  # 1e9
    #         smpsPerSD *= 0.95
    #         print "making smpsPerSD smaller"
    #     else:
    #         b2Fine = False


    #  #rscldA[:, 1]   is rescaled spike times, ALWAYS should be less than 
    # tt1 = _tm.time()


    # # print "##################### done rescale spikes"
    # # print "##################### finding boundary"

    # #########   fiND BOUNDARY   O
    # #if method == _du._GT:
    use_kde = 1 if (method == _du._KDE) else 0

    # print "-----------   going into max_rescaled"
    O, tps = mkd.max_rescaled_T_at_mark_MoG(use_kde, mrngs, g_Ms, prms, t0, t1, smpsPerSD, sds_to_use, i_spc_occ_dt)






    # O = mkd.max_rescaled_T_at_mark_MoG(use_kde, mrngs, g_Ms, prms, t0, t1, smpsPerSD, sds_to_use, i_spc_occ_dt)

    #  pickle
    
    #  fill all points of O that are 0 with very small value
    # i1s, i2s, i3s, i4s = _N.where(O > 0)
    # minV = _N.min(O[i1s, i2s, i3s, i4s])
    # i1s, i2s, i3s, i4s = _N.where(O == 0)
    # O[i1s, i2s, i3s, i4s] = minV*0.0001
    tt2 = _tm.time()
    print "##################### FOUND BOUNDARY   O"


     # theoretically _N.max(O) >= _N.max(rscldA[:, 0]), but due to grid
     # of marks, it may be that _N.max(O) < _N.max(rscldA[:, 0])

     # Now, I have O(m1, m2, m3, m4)
     # First, build volume.
     # obsCnt(m1, m2, m3, m4, O/ dT)  == 0
     # vol[m1, m2, m3, m4, O/ dT]  == 0
     # for each rscldA[:, 1]
     #     find index in volume  obsCnt[m1, m2, m3, m4, O/dt]++
     # for each m1,m2,m3,m4,O/dT:
     # occVolume 
     # if O > 0, 
     #  vol = True

    print "####  1"
    maxT = _N.max(rscldA[:, 0])
    print "####  2"
    if mkd.mdim == 1:
        fig = _plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(1, 1, 1)
        maxY = _N.max(rscldA[:, 1])
        minY = _N.min(rscldA[:, 1])
        A    = maxY - minY
        
        # x    = rscldA[:, 0]
        # y    = rscldA[:, 2]
        # sinds = [i[0] for i in sorted(enumerate(y), key=lambda x:x[1])]
        # _plt.plot(x[sinds], y[sinds], lw=3, color="red")
        # _plt.scatter(rscldA[:, 1], rscldA[:, 2], color="#0077EE", s=16)
        # mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
        # mF.setTicksAndLims(xlabel=None, ylabel=None, xticks=None, yticks=None,  xticksD=None, yticksD=None, xlim=[-1, maxT*1.03], ylim=[minY-0.01*A, maxY+0.01*A], tickFS=23, labelFS=26)
        # #mF.setTicksAndLims(xlabel="Rescaled time", ylabel="mark", xticks=[0, 200, 400, 600], yticks=None, xticksD=None, yticksD=None, xlim=[-1, 600], ylim=[minY-0.01*A, maxY+0.01*A], tickFS=23, labelFS=26)
        # #mF.setTicksAndLims(xlabel="Rescaled time", ylabel="marks", xticks=[0, 200, 400, 600], yticks=None, xticksD=None, yticksD=None, xlim=[-1, 600], ylim=[9, 14], tickFS=23, labelFS=26)
        # fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

    print "####  3"
    if mkd.mdim == 4:
        print "####  3a"
        fig = _plt.figure(figsize=(12, 12))
        for p in xrange(4):
            print "####  3  %d" % p
            ax = fig.add_subplot(2, 2, p+1)
            maxY = _N.max(rscldA[:, 1+p])
            minY = _N.min(rscldA[:, 1+p])
            A    = maxY - minY
            _plt.scatter(rscldA[:, 0], rscldA[:, 1+p], color="black", s=10)
            mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
            mF.setTicksAndLims(xlabel="rescaled time", ylabel=("mark channel %d" % (p+1)), xticks=[0, 3000, 6000], yticks=None, xticksD=None, yticksD=None, xlim=[-1, maxT*1.03], ylim=[minY-0.01*A, maxY+0.01*A], tickFS=25, labelFS=27)
    if tets is not None:
        _plt.savefig("%(df)s/rescltets%(ep)d%(ts)s%(ptrb)s" % {"df" : outdirN, "ep" : epch, "ts" : tetstr, "ptrb" : sPrtrb}, transparent=True)
        _plt.savefig("%(df)s/rescltets%(ep)d%(ts)s%(ptrb)s.eps" % {"df" : outdirN, "ep" : epch, "ts" : tetstr, "ptrb" : sPrtrb}, transparent=True)
    else:
        _plt.savefig("%(df)s/rescltets%(ep)d%(ptrb)s" % {"df" : outdirN, "ep" : epch, "ptrb" : sPrtrb}, transparent=True)
        _plt.savefig("%(df)s/rescltets%(ep)d%(ptrb)s.eps" % {"df" : outdirN, "ep" : epch, "ptrb" : sPrtrb}, transparent=True)
        

#     print "####  4"
#     tt3 = _tm.time()        
#     #  asking for the occ at time==0 doesn't make sense, so 
#     dt     = 0.001   #  in rescaled time
#     rscldT = _N.linspace(dt, _N.max(rscldA[:, 0])*1.01, 1000)
#     print "####  5"
#     #rscldT += _N.diff(rscldT)[0]      
#     rt_dt  = (_N.max(rscldA[:, 0])*1.01 - dt) / 2000000
#     rscldT_final = _N.arange(dt, _N.max(rscldA[:, 0])+0.1, rt_dt)

#     print "####  6"
#     NT_final     = rscldT_final.shape[0]
#     NT     = rscldT.shape[0]
#     vol    = 1   #  
#     for im in xrange(mkd.mdim):
#         vol *= _N.diff(mrngs[im])[0]

    
#     occ    = _N.zeros(NT)    #  occupation (normalized rate of inhomo poisson)
#     if mkd.mdim == 1:
#         _Gu.find_Occ(g_Ms, NT, rscldT, occ, O)
#     if mkd.mdim == 2:
#         _Gu.find_Occ2(g_Ms, NT, rscldT, occ, O)
#     elif mkd.mdim == 4:
#         _Gu.find_Occ4(g_Ms, NT, rscldT, occ, O)
#     tt4 = _tm.time()

#     #      occ[0] = _N.product(g_Ms)   :  all O cells are > rscldT = 0
#     print "##################### find_Ok   done %.4f" % (tt4-tt3)

#     t1 = _tm.time()

#     #  shape of rescldLam same as rescldT
#     rescldLam = occ * vol  #  rescldLam(0) = infinity
#     cps = _N.where(_N.diff(rescldLam) != 0)[0]
#     rescldLam_final = _N.interp(rscldT_final, rscldT[cps], rescldLam[cps])

#     sts    = _N.zeros(NT_final, dtype=_N.int)
#     #  rscldA[:, 0] is no monotonic
#     sts[_N.array(rscldA[:, 0] / rt_dt, dtype=_N.int)] = 1
#     #  sts is marginalized spike time.  don't care where spike originated.
    
#     # delete this
#     #dat = _N.empty((sts.shape[0], 2))
#     #dat[:, 0] = rescldLam_final
#     #dat[:, 1] = sts
    
#     frs  = rescldLam_final.reshape((1, rescldLam_final.shape[0]))
#     s01s = sts.reshape((1, sts.shape[0]))

#     print "b4 2nd timerescale"
#     #x, zss, bs, bsp, bsn = tmrt.timeRescaleTest(frs, s01s, dt, 1, 10, nohist=False, loP=0.00001)


#     x, zss, bs, bsp, bsn, rts = tmrt.timeRescaleTest(frs, s01s, rt_dt, 1, 5, nohist=True)                                                            
#     ks_D, ks_pv = _ss.ks_2samp(x, zss)

#     tt5 = _tm.time()
#     print "##################### timerescale done %.4f" % (tt5-tt4)

# if True:
#     fig = _plt.figure(figsize=(5, 5))
#     ax  = fig.add_subplot(1, 1, 1)
#     _plt.axis("equal")
#     _plt.plot(x, bs, ls=":", lw=2, color="black")
#     _plt.plot(zss, x, color="blue", lw=2)
#     _plt.plot(x, bsp, color="red", lw=2, ls="--")
#     _plt.plot(x, bsn, color="red", lw=2, ls="--")
#     _plt.xlim(0, 1.)
#     _plt.ylim(0, 1)
#     _plt.xlabel("Empiricial CDF", fontsize=23)
#     _plt.ylabel("Model CDF", fontsize=23)
#     _plt.xticks([0, 3000, 6000], fontsize=20)
#     _plt.yticks(fontsize=20)
#     fig.subplots_adjust(left=0.19, bottom=0.19, right=0.95, top=0.95)
#     _plt.savefig("%(df)s/KS%(ep)d%(ts)s%(prt)s" % {"df" : outdirN, "ep" : epch, "ts" : tetstr, "prt" : sPrtrb}, transparent=True)
#     _plt.savefig("%(df)s/KS%(ep)d%(ts)s%(prt)s.eps" % {"df" : outdirN, "ep" : epch, "ts" : tetstr, "prt" : sPrtrb}, transparent=True)


#     #_N.savetxt("%(df)s/sts_rescldlam%(ep)d%(ts)s%(prt)s.dat" % {"df" : outdirN, "ep" : epch, "ts" : tetstr, "prt" : sPrtrb}, dat, fmt="%.2f %d")

#      #dm1 x dm2 x 
#      #Chi2 test


#     # # #  cts_in_mkpos_spc    g_M x g_M x g_T
#     # # #  volrat              g_M x g_M x g_T

#     if K > 1:
#         g_T        = 90
#     else:
#         g_T        = 1000
#     trngs      = _N.linspace(0, _N.max(O)*1.01, g_T)

#     #  cut up border cells to calculate how much of their volume is inside boundary
#     #  first, calculate volrat
#     inside_t = 0
#     outside_t = 0
#     border_t = 0

#     print "---------------------------------------"

#     #  we're almost always outside region.  optimize this by for a given m1, m2, at point where we reach outside, all subsequent times will be outside as well.
#     brdrs = []

#     #dmp = open("O.dmp" % {"df" : outdirN}, "wb")
#     #_pkl.dump([O, trngs], dmp, -1)
#     #dmp.close()


#     t4 = _tm.time()
#     if mkd.mdim == 1:
#         volrat_mk       = _N.zeros((g_Ms[0]-1), dtype=_N.float32)
#         _Gu.calc_volrat(g_T, g_Ms, O, trngs, volrat_mk)
#     elif mkd.mdim == 2:
#         volrat_mk       = _N.zeros((g_Ms[0]-1, g_Ms[1]-1), dtype=_N.float32)
#         inside, outside, border, partials = _Gu.calc_volrat2(g_T, g_Ms, O, trngs, volrat_mk)
#     elif mkd.mdim == 4:
#         volrat_mk       = _N.zeros((g_Ms[0]-1, g_Ms[1]-1, g_Ms[2]-1, g_Ms[3]-1), dtype=_N.float32)
#         _Gu.calc_volrat4(g_T, g_Ms, O, trngs, volrat_mk)

#     #  save in 

#     #O = None


#     # #dV   = 
#     dt          = (trngs[1] - trngs[0])
#     if mkd.mdim == 1:
#         dV          = _N.diff(mrngs[0])[0]
#     elif mkd.mdim == 2:
#         dV          = _N.diff(mrngs[0])[0] * _N.diff(mrngs[1])[0]
#     elif mkd.mdim == 4:
#         dV          = _N.diff(mrngs[0])[0] * _N.diff(mrngs[1])[0] * \
#                       _N.diff(mrngs[2])[0] * _N.diff(mrngs[3])[0]
#     totalVol = dV*dt * _N.sum(volrat_mk)
#     nspks      = rscldA.shape[0]
#     normlz   = (nspks / totalVol) 


#     dmp = open("%s/vars.dmp" % outdirN, "wb")
#     _pkl.dump([K, nspks, g_Ms, rscldA, dV, dt, totalVol, normlz, mrngs, ks_D, ks_pv], dmp, -1)
#     dmp.close()

#     fp = open("%s/outdirN" % outdirN, "w+")
#     fp.write("#  outdirN variable to be used in GoF_pearson\n")
#     fp.write("outdirN=\"%s\"\n" % outdirN)
#     fp.close()

#     # fp = open("%s/volrat.bin" % outdirN, "wb")
#     # volrat_mk.tofile(fp)
#     # fp.close()
