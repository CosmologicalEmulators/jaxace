"""Regenerate the saved original-quadrature neutrino references.

Independent of ``jaxace._neutrinos``: it re-implements the original 128-node
Fermi-Dirac quadrature in plain NumPy, so the expected values never come from
the lookup table under test.  Run: python generate_reference.py
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
TCMB, TREF, NREF = 2.7255, .71611, 3.044
KB_EV = 1.3806504e-23/1.602176487e-19
RINST = (4/11)**(1/3)
N_PER_SPECIES = (TREF/RINST)**4
N_UR_REF = NREF-3*N_PER_SPECIES

x, w = np.polynomial.legendre.leggauss(128)
Q = 32*(x+1)
W = 32*w*Q**2/(1+np.exp(Q))

def F(y):
    return np.sum(W*np.sqrt(Q**2+y**2))

def Fy(y):
    return np.sum(W*y/np.sqrt(Q**2+y**2))

def Fyy(y):
    return np.sum(W*Q**2/(Q**2+y**2)**1.5)

def density(a, ogamma, masses, neff, prescription):
    if prescription == "temperature":
        s = neff/NREF
        T, nur = TREF*s**.25, N_UR_REF*s
    else:
        T, nur = TREF, neff-3*N_PER_SPECIES
    y = a*np.asarray(masses)/(KB_EV*T*TCMB)
    return ogamma/a**4*(15/np.pi**4*T**4*sum(F(v) for v in y)+nur*7/8*RINST**4)

# zero, tiny, small-mass transition, knot-like, large, grid boundary, beyond
YS = [0., 1e-300, 1e-12, 1e-8, 1e-5, 1e-3, .01, .1, .5, 1., 2.5, 7., 20.,
      63.9, 64., 100., 1e3, 1e4, 3e4, 59999., 60000., 60001., 1e5, 1e6, 1e8, 1e12]
with open(HERE/"quadrature_reference.txt", "w") as f:
    f.write("# y F Fy Fyy  (original 128-node quadrature, float64)\n")
    for y in YS:
        f.write(" ".join(repr(float(v)) for v in (y, F(y), Fy(y), Fyy(y)))+"\n")

CASES = [
    (1., (0., 0., 0.), 3.044, "temperature"),
    (1., (.02, .02, .02), 3.044, "temperature"),
    (.5, (.0, .009, .05), 3.044, "temperature"),
    (.1, (.1, .1, .1), 3.5, "temperature"),
    (1e-3, (.3, .1, 0.), 2.8, "temperature"),
    (.02, (.001, .01, .3), 3.044, "radiation"),
    (1., (.06, .0, .0), 3.4, "radiation"),
    (.3, (.2, .2, .2), 3.2, "radiation"),
    (1e-5, (.05, .05, .05), 3.044, "radiation"),
]
OGAMMA = 2.4728e-5/.67**2*.67**2
with open(HERE/"density_reference.txt", "w") as f:
    f.write("# a Ogamma m1 m2 m3 Neff prescription density\n")
    for a, m, n, p in CASES:
        f.write(" ".join([repr(a), repr(OGAMMA), *map(repr, m), repr(n), p,
                          repr(float(density(a, OGAMMA, m, n, p)))])+"\n")
