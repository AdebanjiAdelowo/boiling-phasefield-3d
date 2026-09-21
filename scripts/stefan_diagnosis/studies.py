import multiprocessing as mp, pickle, numpy as np
from lab2 import *
B=dict(bc="wall", tstencil="compact", alpha_mode="one", closure="probe2", gamma="adaptive", src="gradphi", acop="compact", L=0.1)
def dtf(N): return min(5e-4,0.2*(0.1/N)**2/ALPHA)
def mk(**kw):
    d=dict(B); d.update(kw); N=d.get("N",400); d.setdefault("dt",dtf(N)); d["out_every"]=max(1,int(round(0.25/d["dt"]))); return d
CASES={}
# --- A. leave-one-out from converged scheme at N=200 (dx=0.5mm)
CASES["A0 converged scheme N=200"]=mk(N=200)
CASES["A1 src -> phi(1-phi)/eps"]=mk(N=200,src="phi1mphi")
CASES["A2 closure -> central_sign (repo)"]=mk(N=200,closure="central_sign")
CASES["A3 closure -> probe(T_f=Ts)"]=mk(N=200,closure="probe")
CASES["A4 alpha -> alpha_v*phi (repo)"]=mk(N=200,alpha_mode="phi")
CASES["A5 T stencil -> wide (repo)"]=mk(N=200,tstencil="wide")
CASES["A6 AC flux -> nested (repo)"]=mk(N=200,acop="nested")
CASES["A7 gamma -> 0.1 (HEAD)"]=mk(N=200,gamma=0.1)
CASES["A8 gamma -> 1.5e-3"]=mk(N=200,gamma=1.5e-3)
CASES["A9 bc -> periodic wrap"]=mk(N=200,bc="periodic")
# --- B. temporal (N=200, dx=0.5mm)
for dt in (2.5e-3,1.25e-3,6.25e-4,3.125e-4,1.5625e-4,7.8125e-5):
    CASES[f"B dt={dt:.4e} N=200"]=mk(N=200,dt=dt)
# --- C. interface thickness
for e in (1.0,1.25,1.5,2.0,3.0,4.0):
    CASES[f"C fixed dx=0.25mm eps/dx={e}"]=mk(N=400,eps_over_dx=e)
for N in (100,200,400,800):   # fixed eps = 1.5 mm
    e=1.5e-3/(0.1/N); CASES[f"C fixed eps=1.5mm  N={N} eps/dx={e:.1f}"]=mk(N=N,eps_over_dx=e)
# --- D. gamma scale and probe distance sensitivity (N=200)
for c in (0.25,0.5,2.0,4.0,8.0): CASES[f"D gamma_c={c}"]=mk(N=200,gamma_c=c)
for g in (1e-4,3e-4,1e-3,1e-2,1e-1): CASES[f"D gamma fixed={g}"]=mk(N=200,gamma=g)
for d in (1.0,1.5,3.0,4.0): CASES[f"D probe2 h={d}dx"]=mk(N=200,dprobe2=d)
def work(a): n,kw=a; return n,run(C(**kw))
if __name__=="__main__":
    items=sorted(CASES.items(), key=lambda kv:-(1/kv[1]["dt"])*kv[1]["N"]**0.3 if "dt" in kv[1] else 0)
    with mp.Pool(11) as p: res=p.map(work,items,chunksize=1)
    R=dict(res); pickle.dump(R,open("studies.pkl","wb"))
    print("done", len(R))
