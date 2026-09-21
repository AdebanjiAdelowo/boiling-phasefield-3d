"""S1: energy equation ONLY. phi is PRESCRIBED from the analytic front delta_ana(t).
No Allen-Cahn dynamics, no vaporisation closure.  Isolates the T-equation discretisation + wall BC.
Error is measured against the verified analytic T(x,t) at nodes inside the vapour."""
import numpy as np, sys
from stefan_lab import *

def phi_of(x, t, eps):
    return 0.5*(1-np.tanh((x-delta_ana(t))/(2*eps)))

def run_s1(N=200, dt=5e-4, variant="wide_wrapTs", t_end=250.0, eps_over_dx=1.5, report=(25,26,30,60,100,250)):
    L=0.2; dx=L/N; eps=eps_over_dx*dx; x=np.arange(N)*dx; t0=24.7
    T=T_ana(x,t0).astype(float); T[0]=TW
    n=int(round((t_end-t0)/dt)); out={}
    for k in range(1,n+1):
        t=t0+(k-1)*dt; phi=phi_of(x,t,eps); phin=phi_of(x,t+dt,eps)
        a=ALPHA*phi
        if variant.startswith("wide"):
            Tn=T+dt*d1(a*d1(T,dx),dx)                       # repo operator
        elif variant.startswith("compact"):
            af=0.5*(a+np.roll(a,-1))                        # face alpha_{i+1/2}
            Tp=np.roll(T,-1); Tm=np.roll(T,1)
            Tn=T+dt*(af*(Tp-T)-np.roll(af,1)*(T-Tm))/dx**2
        Tn=np.where(phin<0.5,TS,Tn)
        if variant.endswith("wrapTs"): pass                 # repo: node 0 only, wrapped neighbour left at Ts
        elif variant.endswith("ghostTw"): Tn[-1]=TW         # periodic neighbour of node 0 also at Tw
        Tn[0]=TW
        T=Tn
        tt=t0+k*dt
        for r in report:
            if abs(tt-r)<dt/2: out[r]=T.copy()
    return x,out

if __name__=="__main__":
    print(f"{'variant':<18}{'t':>6} {'max|T-Tana| (x<δ-3dx)':>24} {'sawtooth amp max|T_i-(T_{i-1}+T_{i+1})/2|':>44} {'wall flux num':>14} {'wall flux ana':>14}")
    for variant in ["wide_wrapTs","wide_ghostTw","compact_wrapTs","compact_ghostTw"]:
        x,out=run_s1(variant=variant)
        for t,T in out.items():
            d=delta_ana(t); m=(x<d-3e-3)&(x>0.0015)
            err=np.max(np.abs(T[m]-T_ana(x[m],t)))
            saw=np.max(np.abs(T[2:20]-0.5*(T[1:19]+T[3:21])))
            qn=-K*(-3*T[0]+4*T[1]-T[2])/(2*1e-3)
            xx=np.array([0.0,1e-9]); Tx=(T_ana(xx,t)[1]-T_ana(xx,t)[0])/1e-9; qa=-K*Tx
            print(f"{variant:<18}{t:6.0f} {err:24.4f} {saw:44.4f} {qn:14.4f} {qa:14.4f}")
