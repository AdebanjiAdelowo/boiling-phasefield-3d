import multiprocessing as mp, pickle, numpy as np
from lab2 import *
BASE=dict(bc="wall", tstencil="compact", alpha_mode="one", closure="probe2", gamma="adaptive", src="gradphi", acop="compact")
def work(a): n,kw=a; return n,run(C(**kw))
if __name__=="__main__":
    cases={}
    for N in (50,100,200,400,800):
        L=0.1; dx=L/N; dt=min(5e-4,0.2*dx*dx/ALPHA)
        cases[f"N={N} dx={dx*1e3:.3f}mm dt={dt:.1e}"]=dict(BASE,N=N,L=L,dt=dt,out_every=max(1,int(round(0.25/dt))))
    with mp.Pool(11) as p: res=p.map(work,list(cases.items()))
    pickle.dump(dict(res),open("coupled2.pkl","wb"))
    print(f"{'case':<34}"+" ".join(f"{('t='+str(t)):>7}" for t in (25,26,30,60,100,150,250))+"  err%")
    for n,r in res:
        e=errs_at(r,(25,26,30,60,100,150,250)); R=r['rec'][-1]
        print(f"{n:<34}"+" ".join(f"{v:7.3f}" for v in e)+f" | ncross={R['ncross']} clipΣ={R['clip']:+.1e}")
