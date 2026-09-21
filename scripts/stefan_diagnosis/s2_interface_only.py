"""S2: interface side ONLY.  T is PRESCRIBED = exact analytic profile (sharp kink at delta_ana(t)).
Tests: vaporisation closure + Allen-Cahn source/sharpening.  No energy equation is solved."""
import numpy as np
from stefan_lab import *

def closure(phi, Tn, dx, eps, kind, dprobe=None):
    """returns mdot_vol [kg/m3/s] (rho_v=1 so also dphi/dt source)"""
    dphi = d1(phi, dx)
    if kind == "central_sign":            # repo closure (grid-local, central, sign normal)
        nx = dphi/(np.abs(dphi)+1e-14)
        return (K*d1(Tn,dx)*nx/H)*phi*(1-phi)/eps
    if kind == "backward_sign":           # grid-local, one-sided (vapour-side) backward difference
        gb = (Tn-np.roll(Tn,1))/dx
        nx = dphi/(np.abs(dphi)+1e-14)
        return (K*gb*nx/H)*phi*(1-phi)/eps
    if kind == "probe":                   # 1-D probe closure: one scalar flux at phi=0.5, applied over the layer
        xf = interface_first(phi, np.arange(len(phi))*dx)
        h = dprobe
        xs = np.arange(len(phi))*dx
        Tf = TS                            # Dirichlet value at the interface
        T1 = np.interp(xf-h, xs, Tn); T2 = np.interp(xf-2*h, xs, Tn)
        g = (3*Tf - 4*T1 + T2)/(2*h)       # dT/dx at xf, 2nd-order one-sided using T_f=Ts  (negative)
        msurf = -K*g/H                     # k|T_x|/h_lv
        return msurf*phi*(1-phi)/eps
    raise ValueError(kind)

def run_s2(kind, N=200, dt=5e-4, gamma=0.1, eps_over_dx=1.5, t_end=250.0, dprobe_cells=3.0, report=(25,26,30,60,100,250)):
    L=0.2; dx=L/N; eps=eps_over_dx*dx; x=np.arange(N)*dx; t0=24.7
    phi = 0.5*(1-np.tanh((x-delta_ana(t0))/(2*eps)))
    c = Cfg(N=N, dt=dt, gamma=gamma, eps_over_dx=eps_over_dx)
    n=int(round((t_end-t0)/dt)); res=[]; 
    for k in range(1,n+1):
        t=t0+(k-1)*dt
        Tn = T_ana(x,t)
        md = closure(phi, Tn, dx, eps, kind, dprobe_cells*dx)
        phi = np.clip(phi+dt*rhs_phi(phi, md, c),0,1)
        tt=t0+k*dt
        if any(abs(tt-r)<dt/2 for r in report):
            res.append((tt, interface_first(phi,x), delta_ana(tt), phi.sum()*dx))
    return res

if __name__=="__main__":
    print(f"{'closure':<16}{'gamma':>7} | t: err% at t=25,26,30,60,100,250   (delta_num-delta0 vs analytic advance ratio at 250)")
    for kind in ["central_sign","backward_sign","probe"]:
        for g in [0.1, 1.5e-3]:
            r = run_s2(kind, gamma=g)
            errs = [abs(a-b)/b*100 for (_,a,b,_) in r]
            d0=delta_ana(24.7); ratio=(r[-1][1]-d0)/(r[-1][2]-d0)
            print(f"{kind:<16}{g:7.4f} | "+"  ".join(f"{e:6.2f}" for e in errs)+f"   advance_ratio={ratio:.3f}")
