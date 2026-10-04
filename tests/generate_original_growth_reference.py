"""Independent original-(D,Dprime) oracle: NumPy/SciPy only, no JAX/flux.

Restart DOP853 at every frozen Akima knot crossing. Each segment uses its
interior polynomial even at RK endpoint stages, avoiding ambiguous F'' sides.
Complex-step mass sensitivities use the same real-mass segmentation: moving
internal boundaries contribute no jump term because the primal RHS is C0.
Never overwrite fixtures. Run from the repo root with the existing environment.
"""
from pathlib import Path
import hashlib
import numpy as np
import scipy
from scipy.integrate import solve_ivp

DATA = Path(__file__).parent / "data"
TABLE = DATA / "scalar_growth_polynomials.txt"
COEFF = np.loadtxt(TABLE)
AI = 1/139
KT = 8.617342e-5*.71611*2.7255
GAMMA4 = ((4/11)**(1/3)*(3.044/3)**.25)**4


def polynomial(y, piece=None):
    if piece is None:
        piece = np.clip(np.searchsorted(COEFF[:,0], np.real(y), side="right")-1,
                        0, len(COEFF)-2)
    t, u, b, c, d = COEFF[piece]
    w = y-t
    return u+w*(b+w*(c+w*d)), b+w*(2*c+3*w*d)


def original_growth(mass, species, z, ocb=.3, h=.67, w0=-1., wa=0., curvature=0., rtol=1e-13):
    """Return raw D and f; only the historical scalar thermal model is used."""
    photon = 2.469e-5/h**2
    pref = 15/np.pi**4*GAMMA4*photon
    rho0 = pref*polynomial(mass/KT)[0]
    vacuum = 1-photon-ocb-curvature-rho0
    massless = polynomial(0.)[0]
    query = np.log(1/(1+np.asarray(z)))
    knot_a = COEFF[:,0]*KT/np.real(mass)
    stops = np.unique(np.r_[np.log(AI), np.log(knot_a[(knot_a>AI)&(knot_a<1.01)]),
                            query, np.log(1.01)])
    state = np.array([AI,AI], dtype=np.result_type(mass,float))
    answers = {}
    for left,right in zip(stops[:-1],stops[1:]):
        mid_y = np.real(mass)*np.exp((left+right)/2)/KT
        piece = np.clip(np.searchsorted(COEFF[:,0],mid_y,side="right")-1,0,len(COEFF)-2)
        def rhs(t,u):
            a = np.exp(t); y = mass*a/KT
            F,Fp = polynomial(y,piece)
            rho = pref/a**4*F
            de = vacuum*a**(-3*(1+w0+wa))*np.exp(3*wa*(a-1))
            e2 = photon/a**4+ocb/a**3+curvature/a**2+de+rho
            de2 = (-4*photon/a**4-3*ocb/a**3-2*curvature/a**2
                   +(-3*(1+w0+wa)+3*wa*a)*de+pref/a**4*(y*Fp-4*F))
            source = ocb/a**3+(rho-pref/a**4*massless if species=="m" else 0)
            return np.array([u[1],-(2+.5*de2/e2)*u[1]+1.5*source/e2*u[0]])
        sol = solve_ivp(rhs,(left,right),state,method="DOP853",rtol=rtol,atol=rtol/100)
        if not sol.success:
            raise RuntimeError(sol.message)
        state = sol.y[:,-1]
        answers[right] = state.copy()
    values = np.array([answers[t] for t in query])
    return np.column_stack([values[:,0],values[:,1]/values[:,0]])


def generate():
    output = DATA / "scalar_growth_original_reference.txt"
    if output.exists():
        raise FileExistsError(output)
    rows = []
    zs = np.array([0.,.5,1.,2.,3.,5.])
    cases = [(m,.3,.67,-1.,0.,0.) for m in (.06,.3,.75)] + [(.1,.5,.6,-1.5,.2,0.)]
    for m,ocb,h,w0,wa,k in cases:
        for species in ("cb","m"):
            values = original_growth(m,species,zs,ocb,h,w0,wa,k)
            tighter = original_growth(m,species,zs,ocb,h,w0,wa,k,rtol=3e-14)
            complex_values = original_growth(m+1e-20j,species,zs,ocb,h,w0,wa,k)
            gradients = complex_values.imag/1e-20
            gradients2 = original_growth(m+2e-20j,species,zs,ocb,h,w0,wa,k,rtol=3e-14).imag/2e-20
            np.testing.assert_allclose(values,tighter,rtol=3e-11,atol=1e-12)
            np.testing.assert_allclose(gradients,gradients2,rtol=1e-7,atol=2e-10)
            for i,z in enumerate(zs):
                rows.append([int(species=="m"),m,ocb,h,w0,wa,k,z,*values[i],*gradients[i]])
            print(species,m,ocb,"checked D/f and complex-step convergence",flush=True)
    header = (f"SciPy {scipy.__version__} DOP853 original D/Dprime; knot-restarted; rtol=1e-13 atol=1e-15\n"
              f"frozen polynomial SHA256 {hashlib.sha256(TABLE.read_bytes()).hexdigest()}\n"
              "complex-step 1e-20; checked against rtol=3e-14 and step2e-20; not physical scalar-model certification\n"
              "species_m mass ocb h w0 wa curvature z D_raw f derivative_mass_D derivative_mass_f")
    np.savetxt(output,rows,header=header)


if __name__ == "__main__":
    generate()
