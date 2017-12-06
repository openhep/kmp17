
# ### Group theory stuff and definition of IRREPS

from sympy import Rational

global GUTU1
GUTU1 = True    # Is U(1) coupling normalized for GUT unification?

def CA(N):
    """Quadratic Casimir in adjoint rep C2(A)."""
    if N == 1: 
        return 0  # no self-interaction in U(1)
    else:
        return N
    
def DA(N):
    """Dimension of adjoint rep D(A)."""
    if N == 1: 
        return 1
    else:
        return N**2 - 1


class irrep(object):
    """IRREP of SM group
    
    specified by triplet (D3, D2, Y, multiplicity) of dimensions
    w.r.t. SU(3), SU(2) and hypercharge Y, within
    Q = T3 + Y/2 normalization.
    
    multiplicity -- number of such reps, e.g. n_g
    
    """
    
    def __init__(self, D3, D2, Y, multiplicity=1):
        self.D3 = D3
        self.D2 = D2
        self.Y = Y
        self.multiplicity = multiplicity
        self.D = self.D3 * self.D2  # total dimension
        
    def __str__(self):
        return "{}({},{},{},{})".format(self.__class__.__name__, self.D3, self.D2, self.Y, self.multiplicity)
    
    def C2(self, N):
        """Quadratic Casimir.
        
        By default, for U(1) GUT normalization sqrt(3/5) is included.
        """
        global GUTU1
        if N == 1:
            if GUTU1 == True:
                GUTcsq = Rational(3,5)
            else:
                GUTcsq = 1
            C2 = Rational(1,4)*self.Y**2 * GUTcsq
        elif N == 2:
            C2 = Rational(1,4)*(self.D2**2-1)
        elif N == 3:             
            # NOTE: There is ambiguity here. 15: (4,0) is also possible
            # (2,1) is irrep occuring in 70 of SU(5)
            dim2pq = {1: (0,0), 3: (1,0), 6: (2,0), 8: (1,1), 10: (3,0), 15: (2,1)}
            p, q = dim2pq[self.D3]
            C2 = Rational(1,3)*(p**2 + q**3 +3*p + 3*q + p*q)
        return C2 
            
        
    def S2(self, N):
        """Quadratic Dynkin index"""
        return self.multiplicity * self.D * self.C2(N) / DA(N)


def SMDynkin(reps):
    """Total Dynkin index for SM gauge groups, adding all reps."""
    S1, S2, S3 = (0, 0, 0)
    for rep in reps:
        S1 += rep.wS2(1)
        S2 += rep.wS2(2)
        S3 += rep.wS2(3)
    return S1, S2, S3


class Weyl(irrep):
    
    def wS2(self, N):
        """Quadratic Dynkin weighted with n.d.o.f."""
        return Rational(1,2)*self.S2(N)
    

class Dirac(irrep):
    
    def wS2(self, N):
        """Quadratic Dynkin weighted with n.d.o.f."""
        return self.S2(N)

    
class RealScalar(irrep):
    
    def wS2(self, N):
        """Quadratic Dynkin weighted with n.d.o.f."""
        return Rational(1,2)*self.S2(N)
    

class ComplexScalar(irrep):
    
    def wS2(self, N):
        """Quadratic Dynkin weighted with n.d.o.f."""
        return self.S2(N)



# ### SU(5) GUT irreps decomposition from Slansky

SU5_5 = {
    'H5' : (1,2,1),
    'S1' : (3,1,-Rational(2,3))
}

SU5_10 = {
    'h' : (1,1,2),
    'U1' : (3,1,-Rational(4,3)),
    'R2t' : (3,2,Rational(1,3))
}

SU5_15 = {
    'i1' : (1,3,2),
    'R2t' : (3,2,Rational(1,3)),
    'i3' : (6,1,-Rational(4,3))
}
SU5_24 = {
    'c1' : (1,1,0),
    'DEL' : (1,3,0),
    'V2' : (3,2,-Rational(5,3)),
    'V2c' : (3,2,Rational(5,3)),
    'c5' : (8,1,0)
}
SU5_35 = {
    'd1' : (1,4,-3),
    'd2' : (3,3,-Rational(4,3)),
    'd3' : (6,2,Rational(1,3)),
    'd4' : (10,1,2),
}
SU5_40 = {
    'e1' : (1,2,-3),
    'e2' : (3,2,Rational(1,3)),
    'e3' : (3,1,-Rational(4,3)),
    'U3' : (3,3,-Rational(4,3)),
    'e5' : (8,1,2),
    'e6' : (6,2,Rational(1,3))

}
SU5_45 = {
    'H45' : (1,2,1),
    'S1' : (3,1,-Rational(2,3)),
    'S3' : (3,3,-Rational(2,3)),
    'S1t' : (3,1,Rational(8,3)),
    'R2' : (3,2,-Rational(7,3)),
    'f6' : (6,1,-Rational(2,3)),
    
    'f7' : (8,2,1)
}
SU5_50 = {
    'g1' : (1,1,-4),
    'S1' : (3,1,-Rational(2,3)),
    'R2' : (3,2,-Rational(7,3)),
    'g4' : (6,3,-Rational(2,3)),
    'g5' : (6,1,Rational(8,3)),
    'g7' : (8,2,1)
}
SU5_70 = {
    'H70' : (1,2,1),
    'h2' : (1,4,1),
    'S1' : (3,1,-Rational(2,3)),
    'S3' : (3,3,-Rational(2,3)),
    'h5' : (3,3,Rational(8,3)),
    'h6' : (6,2,-Rational(7,3)),
    'h7' : (8,2,1),
    'h8' : (15,1,-Rational(2,3))
}

