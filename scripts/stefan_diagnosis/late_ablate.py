import multiprocessing as mp, pickle, numpy as np
from lab2 import *
BASE=dict(bc="wall", tstencil="compact", alpha_mode="one", closure="probe2", gamma="adaptive", N=400, L=0.1, dt=5e-4, out_every=500)
CASES={
 "N=400 base (adaptive c=1, nested AC)": {},
 "  γ fixed 3e-4": dict(gamma=3e-4),
 "  γ fixed 1e-3": dict(gamma=1e-3),
 "  γ adaptive c=2": dict(gamma_c=2.0),
 "  γ adaptive c=4": dict(gamma_c=4.0),
 "  AC flux operator compact": dict(acop="compact"),
 "  AC compact + γ adaptive c=2": dict(acop="compact",gamma_c=2.0),
 "  AC compact + γ fixed 3e-4": dict(acop="compact",gamma=3e-4),
}
def work(a):
    n,kw=a; return n,run(C(**{**BASE,**kw}))
if __name__=="__main__":
    with mp.Pool(11) as p: res=p.map(work,list(CASES.items()))
    pickle.dump(dict(res),open("late_ablate.pkl","wb"))
    print(f"{'case':<40}"+" ".join(f"{('t='+str(t)):>7}" for t in (30,50,60,100,150,250))+"  err%")
    for n,r in res:
        e=errs_at(r,(30,50,60,100,150,250)); R=r['rec'][-1]
        print(f"{n:<40}"+" ".join(f"{v:7.2f}" for v in e)+f" | ncross_end={R['ncross']} clipΣ={R['clip']:+.1e}")
