
# ### Definition of 1- and 2-loop RGE for a general theory

import numpy as np

from sympy import Rational, sqrt, Symbol, S, init_printing, latex

from constants import *
from group_theory import *

# Starting point for all evolutions
y0 = np.array([1/alpha1_0, 1/alpha2_0, 1/alpha3_0])


# Definition of SM

ng = Symbol('n_g')  # number of generations
nh = Symbol('n_h')  # number of Higgs doublets
# we count only left-handed particles
SMfermions = [
    Weyl(1, 2, -1, ng),  # left lepton doublet
    Weyl(1, 1,  2, ng),  # anti-right lepton singlet
    Weyl(3, 2, Rational(1,3), ng),  # left quark doublet
    Weyl(3, 1, Rational(-4,3), ng), # anti-right up quark singlet
    Weyl(3, 1, Rational(2,3), ng) # anti-right down quark singlet
    ]
SMscalars = [
    ComplexScalar(1, 2, 1, nh)     # Higgs doublet
    ]


# We now implement RGE, 
#  compare with [Di Luzio et al.](http://www.arXiv.org/abs/1504.00359)
#   Eq. (35)-(37)


def rge1L(N, BSMfermions=[], BSMscalars=[], num=False):
    """One loop RGE coefficient."""
    
    a = Rational(-11,3)*CA(N)
    for rep in SMfermions + BSMfermions:
        a += Rational(4,3) * rep.wS2(N)
    for rep in SMscalars + BSMscalars:
        a += Rational(1,3) * rep.wS2(N)
    if num:
        return float(a.subs({ng:3, nh:1}))
    else:
        return a



def rge1LM(BSMfermions=[], BSMscalars=[]):
    """All three one-loop RGE coefficients."""
    return np.array([rge1L(N, BSMfermions=BSMfermions, BSMscalars=BSMscalars, num=True) for N in [1,2,3]])



def rge2L(N, M, BSMfermions=[], BSMscalars=[], num=False):
    """Two-loop RGE coeffs. Same syntax as for one-loop."""
    b = 0
    if M != N:
        for rep in SMfermions + SMscalars + BSMfermions + BSMscalars:
            b += 4 * rep.wS2(N) * rep.C2(M)
    if M == N:
        b += Rational(-34,3)*CA(N)**2
        for rep in SMfermions + BSMfermions:
            b += rep.wS2(N)*(4*rep.C2(N)+Rational(20,3)*CA(N))
        for rep in SMscalars + BSMscalars:
            b += rep.wS2(N)*(4*rep.C2(N)+Rational(2,3)*CA(N))
    if num:
        return float(b.subs({ng:3, nh:1}))
    else:
        return b


def rge2LM(BSMfermions=[], BSMscalars=[]):
    """Whole matrix of two-loop RGE coefficients."""
    return np.array([[rge2L(N, M, BSMfermions=BSMfermions, BSMscalars=BSMscalars, num=True) for M in [1,2,3]] for N in [1,2,3]])



def rge1BSM(N, reps):
    """Contribution to one loop RGE coefficient."""
    a = 0
    for rep in reps:
        if isinstance(rep, Weyl) or isinstance(rep, Dirac):
            a += Rational(4,3) * rep.wS2(N)
        elif isinstance(rep, ComplexScalar) or isinstance(rep, RealScalar):
            a += Rational(1,3) * rep.wS2(N)
        else:
            raise ValueError
    return a


# 1L SM RGE coeffs
SM1L = [rge1L(k).subs({ng:3, nh:1}) for k in [1,2,3]]


