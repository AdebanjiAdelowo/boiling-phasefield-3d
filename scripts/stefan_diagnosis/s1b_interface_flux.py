"""S1b: thermal problem with PRESCRIBED exact phi (tanh at delta_ana(t)); compact conservative T stencil, real wall BC.
Question: what interface heat flux does the numerical T give the vaporisation closure, vs the exact k|T_x|(delta-)?
Flux measures: probe (2nd-order, 3dx/6dx behind phi=0.5 with T_f=Ts), central-at-front, and wall flux."""
import numpy as np, multiprocessing as mp
from stefan_lab import *

def run(N, eps_over_dx, alpha_mode, clamp, L=0.1, dt=None, t_end=250.0, report=(25,30,60,100,250)):
    dx=L/N; eps=eps_over_dx*dx; x=np.arange(N)*dx; t0=24.7
    if dt is None: dt=min(5e-4, 0.2*dx*dx/ALPHA)
    T=np.where(x<delta_ana(t0),T_ana(x,t0),TS).astype(float); T[0]=TW
    n=int(round((t_end-t0)/dt)); out={}
    phi_f=lambda t: 0.5*(1-np.tanh((x-delta_ana(t))/(2*eps)))
    for k in range(1,n+1):
        t=t0+(k-1)*dt; phi=phi_f(t); phin=phi_f(t+dt)
        a=ALPHA*phi if alpha_mode=="phi" else ALPHA*np.ones_like(phi)
        af=0.5*(a[:-1]+a[1:])                 # faces i+1/2, i=0..N-2
        F=af*(T[1:]-T[:-1])/dx                # alpha dT/dx at faces
        Tn=T.copy(); Tn[1:-1]+=dt*(F[1:]-F[:-1])/dx
        if clamp: Tn=np.where(phin<0.5,TS,Tn)
        Tn[0]=TW; T=Tn
        tt=t0+k*dt
        for r in report:
            if abs(tt-r)<dt/2:
                d=delta_ana(tt); v_ana=RHO*H*0+ (delta_ana(tt+0.01)-delta_ana(tt-0.01))/0.02
                xf=d   # exact interface (phi=0.5) is at delta_ana by construction
                h=3*dx
                T1=np.interp(xf-h,x,T); T2=np.interp(xf-2*h,x,T)
                g=(3*TS-4*T1+T2)/(2*h); v_probe=-K*g/(RHO*H)
                out[r]=dict(v_probe=v_probe/v_ana, v_wall=None)
                Tw2=(-3*T[0]+4*T[1]-T[2])/(2*dx); q_wall=-K*Tw2
                q_wall_ana=-K*(T_ana(np.array([1e-9]),tt)[0]-TW)/1e-9
                out[r]["q_wall"]=q_wall/q_wall_ana
    return out

def job(args):
    N,e,am,cl=args; return args, run(N,e,am,cl)

if __name__=="__main__":
    jobs=[]
    for am,cl in [("phi",True),("one",True),("phi",False)]:
        for e in [1.5,3.0]: jobs.append((200,e,am,cl))
    for N in [100,200,400,800]: jobs.append((N,1.5,"phi",True))
    with mp.Pool(11) as pool: res=pool.map(job,jobs)
    print("ratios num/exact.  v_probe = probe-flux vaporisation speed / exact δ'(t);  q_wall = wall flux / exact")
    print(f"{'N':>4} {'ε/dx':>5} {'alpha':>6} {'clamp':>6} | "+" | ".join(f"t={t:<4} v_probe  q_wall" for t in (25,30,60,100,250)))
    for (N,e,am,cl),o in res:
        print(f"{N:4d} {e:5.1f} {am:>6} {str(cl):>6} | "+" | ".join(f"{o[t]['v_probe']:14.3f} {o[t]['q_wall']:7.3f}" if t in o else " "*22 for t in (25,30,60,100,250)))
