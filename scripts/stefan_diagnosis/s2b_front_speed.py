"""S2b: pure Allen-Cahn front-propagation test with a CONSTANT prescribed source (no T, no closure).
Front should move at v = mdot_surf/rho_v.  Measures the realised speed vs gamma, eps/dx, dx.
This is the 1-D analogue of the validated bubble benchmark, but at the Stefan interface speed."""
import numpy as np
from stefan_lab import *

def speed(v, gamma, eps_over_dx=1.5, N=200, dt=5e-4, T=60.0, L=0.2):
    dx=L/N; eps=eps_over_dx*dx; x=np.arange(N)*dx
    c=Cfg(N=N,dt=dt,gamma=gamma,eps_over_dx=eps_over_dx)
    d0=0.0153; phi=0.5*(1-np.tanh((x-d0)/(2*eps)))
    n=int(round(T/dt)); pos=[]
    for k in range(1,n+1):
        md=v*RHO*phi*(1-phi)/eps
        phi=np.clip(phi+dt*rhs_phi(phi,md,c),0,1)
        if k%int(round(1.0/dt))==0: pos.append(interface_first(phi,x))
    pos=np.array(pos); t=np.arange(1,len(pos)+1.0)
    # speed from the second half to skip the initial profile adjustment
    m=len(pos)//2; sp=np.polyfit(t[m:],pos[m:],1)[0]
    return sp, pos[-1]-d0, phi.min(), phi.max()

if __name__=="__main__":
    v=3.0e-4   # ~ interface speed at t0 (xi*sqrt(alpha/t0)=3.08e-4 m/s)
    print(f"target speed v = {v:.2e} m/s ; dx=1mm ; dt=5e-4 s ; run 60 s\n")
    print(f"{'gamma':>9} {'γ/v':>8} | " + " | ".join(f"eps/dx={e:<4}: v_num/v" for e in [1.0,1.5,2.0,3.0]))
    for g in [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1]:
        row=[]
        for e in [1.0,1.5,2.0,3.0]:
            sp,adv,pmin,pmax=speed(v,g,e); row.append(f"{sp/v:15.3f}")
        print(f"{g:9.1e} {g/v:8.1f} | "+" | ".join(row))
