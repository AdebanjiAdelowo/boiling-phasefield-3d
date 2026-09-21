import multiprocessing as mp, pickle, numpy as np, sys
from lab2 import *
BASE=dict(bc="wall", tstencil="compact", alpha_mode="one", closure="probe2", gamma="adaptive")
def work(a):
    name,kw=a; c=C(**kw); r=run(c); return name,r
if __name__=="__main__":
    cases={}
    # domain-size check: same dx=1mm
    cases["N=200 L=0.2 dt=5e-4"]=dict(BASE,N=200,L=0.2,dt=5e-4)
    cases["N=100 L=0.1 dt=5e-4"]=dict(BASE,N=100,L=0.1,dt=5e-4)
    for N in (50,100,200,400,800):
        L=0.1; dx=L/N; dt=min(5e-4,0.2*dx*dx/ALPHA)
        cases[f"refine N={N} dx={dx*1e3:.3f}mm dt={dt:.2e}"]=dict(BASE,N=N,L=L,dt=dt,out_every=max(1,int(round(0.25/dt))))
    with mp.Pool(11) as p: res=p.map(work,list(cases.items()))
    pickle.dump(dict(res),open("coupled_refine.pkl","wb"))
    print(f"{'case':<44}"+" ".join(f"{('t='+str(t)):>7}" for t in (25,26,30,60,100,250))+"  (err %)")
    for n,r in res: summary(n,r)
