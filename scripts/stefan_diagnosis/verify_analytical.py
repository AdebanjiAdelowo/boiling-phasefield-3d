"""Independent verification of the Stefan reference solution used by the repo."""
import numpy as np
from scipy.special import erf
from scipy.integrate import solve_ivp, quad
from stefan_lab import *

print("== (a) dimensional / algebraic checks ==")
print(f"alpha=k/(rho Cp)={ALPHA:.4e} m^2/s ; St=Cp*dT/h={ST} ; xi={XI:.6f} (paper: 0.3064)")
print("transcendental residual xi*exp(xi^2)*erf(xi)-St/sqrt(pi) =", XI*np.exp(XI**2)*erf(XI)-ST/np.sqrt(np.pi))

print("\n== (b) Stefan condition rho*h*ddelta/dt = -k*T_x(delta^-), by finite differences of the closed forms ==")
for t in [24.7, 100.0, 250.0]:
    h = 1e-6*t
    ddel = (delta_ana(t+h)-delta_ana(t-h))/(2*h)
    d = delta_ana(t); e = 1e-9
    Tx = (T_ana(np.array([d-e]),t)[0]-T_ana(np.array([d-2*e]),t)[0])/e   # one-sided from vapour side
    lhs, rhs = RHO*H*ddel, -K*Tx
    print(f"t={t:6.1f}  rho h dδ/dt={lhs:.6e}  -k T_x(δ-)={rhs:.6e}  rel.diff={abs(lhs-rhs)/lhs:.2e}  T(δ)={T_ana(np.array([d-1e-12]),t)[0]:.2e}  T(0)={T_ana(np.array([0.]),t)[0]:.3f}")

print("\n== (c) exact continuous energy budget  dE/dt = Q_wall - rho h dδ/dt  (E = rho Cp ∫_0^δ (T-Ts) dx) ==")
def E(t):
    d = delta_ana(t); return RHO*CP*quad(lambda x: T_ana(np.array([x]),t)[0]-TS, 0, d, epsabs=1e-13, epsrel=1e-13)[0]
for t in [24.7, 100.0, 250.0]:
    h = 1e-4*t
    dE = (E(t+h)-E(t-h))/(2*h)
    Qw = -K*(T_ana(np.array([1e-9]),t)[0]-T_ana(np.array([0.]),t)[0])/1e-9
    ddel = (delta_ana(t+h)-delta_ana(t-h))/(2*h)
    print(f"t={t:6.1f}  dE/dt={dE:.6e}  Qwall-rho h dδ/dt={Qw-RHO*H*ddel:.6e}  Qwall={Qw:.5e}  latent={RHO*H*ddel:.5e}")

print("\n== (d) independent moving-boundary PDE solve (front-fixing y=x/δ, implicit-free RK45 MOL) ==")
# unknowns: T_j at y_j (j=1..n-1), δ.  T_t = α T_yy/δ² + y (δ'/δ) T_y ;  δ' = -(k/(ρ h)) T_y(1)/δ
n = 400; y = np.linspace(0,1,n+1); dy = y[1]-y[0]
t0 = 24.7; d0 = delta_ana(t0)
Tini = T_ana(y*d0, t0); Tini[0]=TW; Tini[-1]=TS
def rhs(t, s):
    T = np.concatenate(([TW], s[:-1], [TS])); d = s[-1]
    Ty1 = (3*T[-1]-4*T[-2]+T[-3])/(2*dy)             # 2nd-order one-sided at y=1
    ddot = -(K/(RHO*H))*Ty1/d
    Tyy = (T[2:]-2*T[1:-1]+T[:-2])/dy**2
    Ty = (T[2:]-T[:-2])/(2*dy)
    dT = ALPHA*Tyy/d**2 + y[1:-1]*(ddot/d)*Ty
    return np.concatenate((dT,[ddot]))
sol = solve_ivp(rhs,(t0,250.0),np.concatenate((Tini[1:-1],[d0])),method='BDF',t_eval=[25,50,100,150,200,250],rtol=1e-10,atol=1e-12)
for t,d in zip(sol.t, sol.y[-1]):
    print(f"t={t:6.1f}  δ_PDE={d:.8f}  δ_similarity={delta_ana(t):.8f}  rel.diff={abs(d-delta_ana(t))/delta_ana(t):.2e}")
