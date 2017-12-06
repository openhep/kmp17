
# Routines for algebraic and numeric testing
#  of 1-loop GUT of coupling constants for a given particle set

# Also routines searching for a particle sets resulting in
#  a successful GUT

# Finally, routines for numerical check of GUT at 1-loop and 2-loops

import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy import pi

from constants import *
from rge import *


def t(mu):
    """Evolution variable t in terms of mu [Gev]."""
    return math.log(mu/mZ)/(2*pi)

def mu(t):
    """Inverse of t(mu)"""
    return mZ*math.exp(2*pi*t)

def rk(muk, mGUT=5.0e15):
    """Threshold weight coeff correcting B_12 and B_23."""
    tt = t(mGUT)
    return (tt - t(muk))/tt

def muk(rk, mGUT=5.0e15):
    """Inverse of rk(muk)."""
    return mGUT*np.exp(rk*np.log(mZ/mGUT))


########################
# ## Traditional B-test
########################

# At 1L to have unification we need
# $$\frac{b_2 - b_3}{b_1 - b_2} = 0.718 \pm 0.007 $$


def Bunif1L(BSMfermions=[], BSMscalars=[]):
    b1, b2, b3 = rge1LM(BSMfermions, BSMscalars)
    mGUT = mZ*np.exp(184.87/(b1-b2))
    print('MGUT = {:.2g} GeV'.format(mGUT))
    print('Bunif = {:.3f} (should be 0.718)'.format((b2-b3)/(b1-b2)))


########################
# ## 'Linearized' A-test --> FIXME: to be renamed to C-test
########################

def Atest(reps, Br=0.718):
    "Return B12 and A-test value."
    if not isinstance(reps, list):
        reps = [reps]
    b1, b2, b3 = [rge1BSM(k, reps) for k in [1,2,3]]
    return np.array([b1-b2, (b2-b3)-Br*(b1-b2)], dtype=np.dtype(float))





#########################
# ## GUT search algorithm
#########################

# BPR neutrino mass model is used below as a default at some places

BPR = [Weyl(1,2,1, 3), Weyl(1,2,-1, 3), ComplexScalar(1,1,2), RealScalar(1,3,0)]

def rlow(rep, mGUT):
    """Catalogue of low bounds on rk for rep.
    
    """
    if str(rep) == 'RealScalar(8,1,0,1)':
        # LHC direct bound search is 3.1e3 but
        # it is model dependent!
        #low = 3.1e3  # GeV
        low = 500.
        ##low = mZ # for test
    elif str(rep) == 'ComplexScalar(15,1,-2/3,1)':
        # Perturbativity by Finnish group
        low = 5.0e6
    else:
        low = 500.
    
    return rk(low, mGUT=mGUT)


# 1. Simple analytic algorithm derived using Lagrange multipliers



def ABsolve(freereps=[], fixreps=BPR, mGUTtarget=5.e15, verbose=False):
    """Solve A-test with given mGUT and minimal constrained |x|
    
       freereps = BSM reps with variable scale
       fixreps  = BSM reps fixed at ew/LHC scale
       mGUTtarget = required GUT scale

    """
    hiscale = mGUTtarget # search up to this scale
    rhigh = rk(hiscale, mGUT=mGUTtarget)  # high r limit
    #
    Br = 0.718   # B-test ratio
    BZ = 184.8 # MGUT expression exponent numerator
    b1, b2, b3 = rge1LM()  # SM RGE coeffs
    a = (b1-b2)*Br - (b2-b3) # needed BSM contribution to A-test
    b = BZ/math.log(mGUTtarget/mZ)-(b1-b2)  # needed BSM contribution to B_12
    # Remove contributions coming from fixed reps:
    Mfix = 500  # GeV
    B12fix, Afix = Atest(fixreps) * rk(Mfix)
    a -= Afix  
    b -= B12fix 
    # Remaining values of a,b should be provided by free reps:
    BA = np.array([Atest(rep) for rep in freereps]).transpose()
    A = BA[1,:]
    B = BA[0,:]
    # --- Switch to x = rb - r ---
    # 1. determine bounds
    xhi = []
    rb = []
    for rep in freereps:
        # low bounds come from catalogue function rlow()
        low = rlow(rep, mGUTtarget)
        xhi.append(low  - rhigh)  # high bound on x
        rb.append(low)          # corresponding low bound on r
    if verbose: print(rb)
    xhi = np.array(xhi)
    rb = np.array(rb)
    # 2. Switch from r to x is obtained by a->atilde, b->btilde
    # atilde = A.rb - a
    # btilde = B.rb - b
    btat = np.dot(BA,rb) - np.array([b,a])
    if len(freereps) == 0:
        print("No freereps specified. Nothing to search.")
    elif len(freereps) == 1:
        # Single free param. Require crossing only.
        x = btat[1]/BA[1,:]
    elif len(freereps) == 2:
        # Two free params. Require crossing and required scale
        # so we have two eqs in two unknowns and x = BA^(-1).(bt,at)^T
        x = np.dot(np.linalg.inv(BA), btat)
    elif len(freereps) >= 3:
        # underdermined system. Optimize for minimum |x|
        # --- Calculate optimal x ---
        BAT = BA.transpose()
        x = np.dot(BAT, np.dot(np.linalg.inv(np.dot(BA, BAT)), btat))
    if verbose: print('x = {}'.format(x))
    # Going back x --> r
    r = rb - x
    B12 = (b1-b2) + B12fix + np.dot(B,r)
    mGUT = mZ*np.exp(BZ/B12)
    if verbose: 
    #if mGUT > 2.e15 and abs(np.dot(A,r)-a) < 0.1 and (r > rhigh).all()  and (r < 1.05).all() :
        print("\n--- {} --- ".format([(rep.D3, rep.D2, rep.Y) for rep in freereps]))
        print("rk = {}".format(r))
        print("mk = {}".format(muk(r, mGUT=mGUT)))
        print("mGUT = {:.2e} (<- {:.1e})".format(mGUT, mGUTtarget))
        print("Adiff = {:.1e}, Bdiff = {:.1e}".format(np.dot(A,r)-a, np.dot(B,r)-b))
        if len(A) >= 2:
            print("|x| = {:.2e}; (Bx)^2 = {:.6e}".format(np.sqrt(np.dot(x,x)), np.dot(B,x)**2))
    return mGUT, r


def ABsearch(freereps=[], fixreps=BPR, mGUTmin=1.e15, verbose=False):
    """Search for solutions of A-test with given mGUT and lowest allowed rk.
       via scanning mGUT values from minimal to highest one 
       for which solution exists
    
       freereps = BSM reps with variable scale
       fixreps  = BSM reps fixed at ew/LHC scale
       mGUTmin  = minimal GUT scale, where search starts

    """
    if len(freereps) < 3:
        print("This search doesn't make sense for less than 3 free states!")
        return None
    
    r = np.zeros(len(freereps))
    mGUTtarget = mGUTmin
    mGUT = mGUTmin
    rb = [rlow(rep, mGUTtarget) for rep in freereps]
    # We do a poor's man adaptive procedure decreasing step
    # by hand:
    steps = [0.1e15, 0.02e15, 0.005e15, 0.001e15]
    for step in steps:
        while (r < rb).all() and (r >= 0).all():
            mGUT, r = ABsolve(freereps=freereps, fixreps=fixreps, mGUTtarget=mGUTtarget, verbose=verbose)
            assert abs(mGUT-mGUTtarget)<3.e13
            mGUTtarget += step
            rb = [rlow(rep, mGUTtarget) for rep in freereps]
        # step back and decrease step size
        mGUTtarget -= 2*step
        mGUT, r = ABsolve(freereps=freereps, fixreps=fixreps, mGUTtarget=mGUTtarget, verbose=verbose)
    if mGUT and mGUT > 2.e15 and (r < rb).all() and (r > 0.4).all():
        print("\n--- {} --- ".format([(rep.D3, rep.D2, rep.Y) for rep in freereps]))
        print("rk = {}".format(r))
        print("mk = {}".format(muk(r, mGUT=mGUT)))
        print("mGUTmax = {:.3e} (<- {:.1e})".format(mGUT, mGUTmin))



# Numerical version of above algorithm, using scipy.optimize.minimize with
#  SLSQ algorithm that also can deal with strict bounds


def NBsolve(freereps=[], fixreps=BPR, mGUTtarget=5.e15, verbose=False):
    """Solve A-test with given mGUT and minimal |x|
    
       freereps = BSM reps with variable scale
       fixreps  = BSM reps fixed at ew/LHC scale
       mGUTtarget = required GUT scale

    """
    hiscale = mGUTtarget  # search up to this scale
    rhigh = rk(hiscale, mGUT=mGUTtarget)  # high r limit
    #
    Br = 0.718   # B-test ratio
    BZ = 184.8 # MGUT expression exponent numerator
    b1, b2, b3 = rge1LM()  # SM RGE coeffs
    a = (b1-b2)*Br - (b2-b3) # needed BSM contribution to A-test
    b = BZ/math.log(mGUTtarget/mZ)-(b1-b2)  # needed BSM contribution to B_12
    # Remove contributions coming from fixed reps:
    Mfix = 500  # GeV
    B12fix, Afix = Atest(fixreps) * rk(Mfix)
    a -= Afix  
    b -= B12fix 
    # Remaining values of a,b should be provided by free reps:
    BA = np.array([Atest(rep) for rep in freereps]).transpose()
    A = BA[1,:]
    B = BA[0,:]
    # --- Switch to x = rb - r ---
    # 1. determine bounds
    bounds = []
    xhi = []
    rb = []
    for rep in freereps:
        # low bounds come from catalogue function rlow()
        low = rlow(rep, mGUTtarget)
        xhi.append(low  - rhigh)  # high bound on x
        bounds.append((0, low  - rhigh))  # bounds on x
        rb.append(low)          # corresponding low bound on r
    if verbose: print(rb)
    xhi = np.array(xhi)
    rb = np.array(rb)
    # 2. Switch from r to x is obtained by a->atilde, b->btilde
    # atilde = A.rb - a
    # btilde = B.rb - b
    btat = np.dot(BA,rb) - np.array([b,a])
    
    def func(x, sign=1.0):
        """Function to be minimized, maximal discovery potential. Can be called with sign=-1 for maximization."""
        return sign*(np.dot(x,x)**2)/2.

    def Dfunc(x, sign=1.0):
        """Derivation of func. Can be called with sign=-1 for maximization."""
        return sign*x
    
    def cross_scale_fun(x):
        """Constraint function. Crossing and GUT scale"""
        return (np.dot(BA,x)-btat)
      

    constraints = {'type':'eq', 'fun':cross_scale_fun}
    
    x0 = [0. for rep in freereps]   # initial point

    
    if len(freereps) == 0:
        print("No freereps specified. Nothing to search.")
    elif len(freereps) == 1:
        # Single free param. Require crossing only.
        x = btat[1]/BA[1,:]
    elif len(freereps) == 2:
        # Two free params. Require crossing and required scale
        # so we have two eqs in two unknowns and x = BA^(-1).(bt,at)^T
        try:
            x = np.dot(np.linalg.inv(BA), btat)
        except np.linalg.LinAlgError:
            return None, rb
    elif len(freereps) >= 3:
        # underdermined system. Optimize for minimum |x|
        # --- Calculate optimal x ---
        res = minimize(func, x0, args=(1.0,), jac=Dfunc, method='SLSQP', bounds=bounds, constraints=constraints)
        if res['success']:
            x = res['x']
        else:
            if verbose: print('opt not successful')
            return None, rb
    if verbose: print(x, cross_scale_fun(x))
    # Going back x --> r
    r = rb - x
    B12 = (b1-b2) + B12fix + np.dot(B,r)
    mGUT = mZ*np.exp(BZ/B12)
    if verbose: 
    #if mGUT > 2.e15 and abs(np.dot(A,r)-a) < 0.1 and (r > rhigh).all()  and (r < 1.05).all() :
        print("\n--- {} --- ".format([(rep.D3, rep.D2, rep.Y) for rep in freereps]))
        print("rk = {}".format(r))
        print("mk = {}".format(muk(r, mGUT=mGUT)))
        print("mGUT = {:.2e} (<- {:.1e})".format(mGUT, mGUTtarget))
        print("Adiff = {:.1e}, Bdiff = {:.1e}".format(np.dot(A,r)-a, np.dot(B,r)-b))
        if len(A) >= 2:
            print("|x| = {:.2e}; (Bx)^2 = {:.6e}".format(np.sqrt(np.dot(x,x)), np.dot(B,x)**2))
    if mGUT > 0.9e15 and abs(np.dot(A,r)-a) < 0.1:
        return mGUT, r
    else:
        if verbose: print('C-test of S-test fail.')
        return None, rb


def NBsearch(freereps=[], fixreps=BPR, mGUTmin=1.e15, verbose=False):
    """Search for solutions of A-test with given mGUT and lowest allowed rk.
       via scanning mGUT values from minimal to highest one 
       for which solution exists
    
       freereps = BSM reps with variable scale
       fixreps  = BSM reps fixed at ew/LHC scale
       mGUTmin  = minimal GUT scale, where search starts

    """

    r = np.zeros(len(freereps))
    mGUTtarget = mGUTmin
    mGUT = mGUTmin
    rb = [rlow(rep, mGUTtarget) for rep in freereps]
    # We do a poor's man adaptive procedure decreasing step
    # by hand:
    steps = [0.1e15, 0.02e15, 0.005e15, 0.001e15]
    for step in steps:
        while (r < rb).all() and (r >= 0).all():
            mGUT, r = NBsolve(freereps=freereps, fixreps=fixreps, mGUTtarget=mGUTtarget, verbose=verbose)
            #print('NBsolve: mGUT = {:.2e}'.format(mGUT))
            if not mGUT or abs(mGUT-mGUTtarget)>3.e13:
                # print('opt failed, diff = {:.3e}, {:.3e}, {:.3e}'.format(
                # mGUT, mGUTtarget, mGUT-mGUTtarget))
                break  # opt failed, or mGUT not met -> step back
            mGUTtarget += step
            rb = [rlow(rep, mGUTtarget) for rep in freereps]
            #print(mGUTtarget, rb)
        # step back and decrease step size
        mGUTtarget -= 2*step
        #print(mGUT, (r < rb).all(), '-->', mGUTtarget)
        mGUT, r = NBsolve(freereps=freereps, fixreps=fixreps, mGUTtarget=mGUTtarget, verbose=verbose)
    if mGUT and mGUT > 2.e15 and (r < rb).all() and (r >= 0.4).all():
        print("\n--- {} --- ".format([(rep.D3, rep.D2, rep.Y) for rep in freereps]))
        print("rk = {}".format(r))
        print("mk = {}".format(muk(r, mGUT=mGUT)))
        print("mGUTmax = {:.3e} (<- {:.1e})".format(mGUT, mGUTmin))



###################################
# ## Numeric GUT with thresholds
###################################


def precalcRGE(thresholds=[]):
    """Precalculates RGE coeffs.
    
    Element of thresholds list: (mu, [irrep1, irrep2, ...])
    with irreps entering evolution at scale mu [GeV].
    mu's should increase along the list.
    Returns: array of t-thresholds and corresponding
             arrays of RGE coeffs -a_i and
             matrices b_ij / (4pi).
    """
    ts = [t(mZ)]
    cRGE = [(-rge1LM(), -rge2LM()/(4*pi))]
    active_BSM_fermions = []
    active_BSM_scalars = []
    thresholds.sort(key=lambda it: it[0])
    for mu, reps in thresholds:
        if type(reps) != list: reps=[reps]
        for rep in reps:
            if rep.__class__.__name__[-4:] == 'alar':
                active_BSM_scalars.append(rep)
            else:
                active_BSM_fermions.append(rep)
        ts.append(t(mu))
        cRGE.append((-rge1LM(BSMfermions=active_BSM_fermions, BSMscalars=active_BSM_scalars), -rge2LM(BSMfermions=active_BSM_fermions, BSMscalars=active_BSM_scalars)/(4*pi)))
    return np.array(ts), cRGE


def meet(ts, ys):
    """Return total distance of alphas at best unification, and corresponding scale."""
    mn = 100
    tmn = 0
    for ti, (y1, y2, y3) in zip(ts,ys):
        dst = abs(y1-y2)+abs(y1-y3)+abs(y2-y3)
        if dst < mn:
            mn = dst
            tmn = ti
    yavg = (y1 + y2 + y3) / 3.
    mX = mu(tmn)
    vev = mX*np.sqrt(2*yavg/(25*np.pi))
    #print("Dist = {:.2f}, M_GUT = 10^({:.1f}) GeV".format(mn, math.log10(mu(tmn))))
    print("Dist = {:.2f}, M_GUT = {:.2g} GeV, , vGUT = {:.2g} GeV".format(mn, mX, vev))


def rgefig(ts, ys, xlim=False, ylim=False):    
    logms = np.array([math.log10(mu(t)) for t in ts])
    fig, ax = plt.subplots(figsize=[4,3])
    ax.plot(logms, ys[:,0], 'k:', label=r'$\alpha_{1}^{-1}$')
    ax.plot(logms, ys[:,1], 'b-', label=r'$\alpha_{2}^{-1}$')
    ax.plot(logms, ys[:,2], 'r--', label=r'$\alpha_{3}^{-1}$')
    props = dict(color="green", linestyle="-", linewidth=1)
    ax.axhline(y=0, **props)
    ax.set_ylabel(r'$\alpha_{i}^{-1}$', fontsize='14')
    ax.set_xlabel(r'$\log_{10}(\mu/{\rm GeV})$', fontsize='14')
    ax.legend(loc='upper right').draw_frame(0)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    plt.show()



