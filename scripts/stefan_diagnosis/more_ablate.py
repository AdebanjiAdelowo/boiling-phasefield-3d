import multiprocessing as mp, pickle
from lab2 import *
from studies import mk
CASES={
 "N=400 converged":mk(N=400),
 "N=400 src->phi(1-phi)/eps":mk(N=400,src="phi1mphi"),
 "N=400 AC->nested":mk(N=400,acop="nested"),
 "N=800 converged":mk(N=800),
 "N=800 src->phi(1-phi)/eps":mk(N=800,src="phi1mphi"),
 "N=800 AC->nested":mk(N=800,acop="nested"),
}
def work(a): n,kw=a; return n,run(C(**kw))
if __name__=="__main__":
    with mp.Pool(6) as p: res=p.map(work,list(CASES.items()),chunksize=1)
    pickle.dump(dict(res),open("more_ablate.pkl","wb")); print("done")
