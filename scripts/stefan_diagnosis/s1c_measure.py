"""S1c: exact prescribed phi (tanh at delta_ana), compact T stencil, constant alpha (Roccon R.10 with alpha_l=alpha_v)
vs alpha_v*phi.  Thermal treatment at the front:  'clamp' (repo, staircase at first node with phi<0.5)  vs
'ghost' (sub-grid Dirichlet Ts at the exact phi=0.5 location, linear-interp ghost value).
Flux measures at the front (ratio to exact k|T_x|(delta-)/(rho h) = delta'):
  m1 probe with T_f=Ts (2nd order, 3dx,6dx)   m2 gradient extrapolation from central differences at xf-h, xf-2h (h=2dx)
  m3 last-cell flux  k (T_j - Ts)/(x_D - x_j)  [energy-consistent flux into the Dirichlet point]"""
import numpy as np, multiprocessing as mp
from stefan_lab import *

def run(N, alpha_mode, front, eps_over_dx=1.5, L=0.1, t_end=250.0, report=(25,30,60,100,250)):
    dx=L/N; eps=eps_over_dx*dx; x=np.arange(N)*dx; t0=24.7
    dt=min(5e-4,0.2*dx*dx/ALPHA)
    T=np.where(x<delta_ana(t0),T_ana(x,t0),TS).astype(float); T[0]=TW
    n=int(round((t_end-t0)/dt)); out={}
    for k in range(1,n+1):
        t=t0+(k-1)*dt; tn=t+dt; xf=delta_ana(tn)          # exact interface at new time
        phi=0.5*(1-np.tanh((x-delta_ana(t))/(2*eps)))
        a=ALPHA*phi if alpha_mode=="phi" else ALPHA*np.ones(N)
        af=0.5*(a[:-1]+a[1:]); Tn=T.copy()
        if front=="clamp":
            F=af*(T[1:]-T[:-1])/dx; Tn[1:-1]+=dt*(F[1:]-F[:-1])/dx
            phin=0.5*(1-np.tanh((x-xf)/(2*eps))); Tn=np.where(phin<0.5,TS,Tn)
        else:  # ghost: nodes j<=jf active; ghost at j+1 makes linear interp equal Ts at xf
            jf=int(np.floor(xf/dx)); th=(xf-x[jf])/dx        # xf in [x_jf, x_jf+dx)
            Tg=T.copy(); Tg[jf+1:]=TS
            # T_ghost s.t. T(xf)=Ts on line through x_jf, x_jf+1:  T_j+th*(Tg-T_j)=Ts -> Tg=(Ts-(1-th)T_j)/th
            th_=max(th,1e-3); Tgv=(TS-(1-th_)*T[jf])/th_ if th_>=0.05 else None
            Tg2=T.copy()
            if Tgv is None:  # interface essentially at node jf: Dirichlet at node
                Tg2[jf]=TS
            else:
                Tg2[jf+1]=Tgv
            Tg2[jf+2:]=TS
            F=af*(Tg2[1:]-Tg2[:-1])/dx; Tn=T+0.0; Tn[1:-1]+=dt*(F[1:]-F[:-1])/dx
            Tn[jf+1:]=TS
            if Tgv is None: Tn[jf]=TS
        Tn[0]=TW; T=Tn; tt=t0+k*dt
        for r in report:
            if abs(tt-r)<dt/2:
                d=delta_ana(tt); v_ana=(delta_ana(tt+0.01)-delta_ana(tt-0.01))/0.02
                h=3*dx; g=(3*TS-4*np.interp(d-h,x,T)+np.interp(d-2*h,x,T))/(2*h); m1=-K*g/(RHO*H)/v_ana
                h=2*dx; gc=lambda xp:(np.interp(xp+dx,x,T)-np.interp(xp-dx,x,T))/(2*dx)
                g0=2*gc(d-h)-gc(d-2*h); m2=-K*g0/(RHO*H)/v_ana
                j=int(np.floor(d/dx)); m3=K*(T[j]-TS)/(x[j+1]-x[j])/(RHO*H)/v_ana
                out[r]=(m1,m2,m3)
    return out
def job(a): return a, run(*a)
if __name__=="__main__":
    jobs=[(200,am,fr) for am in("phi","one") for fr in("clamp","ghost")]
    jobs+=[(N,"one",fr) for N in (100,400,800) for fr in("clamp","ghost")]
    with mp.Pool(11) as p: res=p.map(job,jobs)
    print("ratio to exact; columns per time: m1(probe,Tf=Ts) m2(extrap-central) m3(last-cell)")
    print(f"{'N':>4} {'alpha':>5} {'front':>6} | "+" | ".join(f"t={t:<3}  m1    m2    m3 " for t in (25,30,60,100,250)))
    for (N,am,fr),o in sorted(res,key=lambda z:(z[0][1],z[0][2],z[0][0])):
        print(f"{N:4d} {am:>5} {fr:>6} | "+" | ".join(f"{o[t][0]:6.3f}{o[t][1]:6.3f}{o[t][2]:6.3f}" for t in (25,30,60,100,250)))
