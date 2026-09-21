"""S2c: pure AC front, constant source speed v=1.5e-4 m/s (~ Stefan speed at t~100), 150 s.
Track speed of phi=0.5 relative to v, 10-90 width [cells] (ideal 4.39*eps/dx), and source-integral ratio  ∫ mdot_vol/rho dx / v."""
import numpy as np, multiprocessing as mp
from lab2 import *
def width(phi,x,dx):
    i=np.where(phi<0.9)[0][0]; j=np.where(phi<0.1)[0][0]
    xs=x[i-2:j+3]; ps=phi[i-2:j+3]
    return (np.interp(0.1,ps[::-1],xs[::-1])-np.interp(0.9,ps[::-1],xs[::-1]))/dx
def go(a):
    N,e,g_over_v,acop,src=a; v=1.5e-4; L=0.1; dt=5e-4
    c=C(N=N,L=L,dt=dt,eps_over_dx=e,acop=acop,bc="wall"); dx=c.dx; x=np.arange(N)*dx; eps=c.eps
    phi=0.5*(1-np.tanh((x-0.02)/(2*eps))); gam=g_over_v*v; out=[]
    n=int(150/dt)
    for k in range(1,n+1):
        if src=="phi1mphi": md=v*RHO*phi*(1-phi)/eps
        else:  # exact |grad phi| normalisation (central diff)
            md=v*RHO*np.abs(D1(phi,c))
        phi=np.clip(phi+dt*rhs_phi(phi,md,c,gam),0,1)
        if k%int(25/dt)==0:
            out.append((k*dt,interface_first(phi,x),width(phi,x,dx)/(4.39*e), np.sum(md)*dx/RHO/v))
    return a,out
if __name__=="__main__":
    jobs=[(400,1.5,g,ac,s) for ac in("nested","compact") for g in(0.5,1.0,2.0) for s in("phi1mphi","gradphi")]
    with mp.Pool(11) as p: res=p.map(go,jobs)
    print("N=400 (dx=0.25mm), eps=1.5dx, v=1.5e-4.  entries: width/ideal , source-integral/v  at t=25,50,...,150 (relative to run start)")
    for (N,e,g,ac,s),o in res:
        d0=0.02; sp=(o[-1][1]-o[0][1])/(o[-1][0]-o[0][0])/1.5e-4
        print(f"AC={ac:<7} γ/v={g:<3} src={s:<9} speed/v(25-150s)={sp:6.3f} | "+" ".join(f"{w:5.2f}/{q:5.2f}" for (_,_,w,q) in o))
