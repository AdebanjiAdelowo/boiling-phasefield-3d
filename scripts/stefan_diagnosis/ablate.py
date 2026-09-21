import multiprocessing as mp, pickle, numpy as np
from lab2 import *

CORR = dict(bc="wall", tstencil="compact", closure="probe", gamma="adaptive")
CASES = {
 # --- forward: one change from HEAD ---
 "F0 HEAD (repo)": {},
 "F1 HEAD + gamma=1.5e-3 (pre-fix)": dict(gamma=1.5e-3),
 "F2 HEAD + gamma=adaptive": dict(gamma="adaptive"),
 "F3 HEAD + wrapped nbr of node0 = Tw": dict(wall_ghost="Tw"),
 "F4 HEAD + compact T stencil": dict(tstencil="compact"),
 "F5 HEAD + probe closure": dict(closure="probe"),
 "F6 HEAD + backward closure": dict(closure="backward_sign"),
 "F7 HEAD + non-periodic (reflect) bc": dict(bc="wall"),
 # --- corrected scheme and leave-one-out ---
 "C* corrected (wall+compact+probe+adaptive)": dict(CORR),
 "C* - compact  (wide T, wall bc)": {**CORR, "tstencil": "wide"},
 "C* - nonperiodic bc (periodic, compact T)": {**CORR, "bc": "periodic"},
 "C* - probe (-> central_sign)": {**CORR, "closure": "central_sign"},
 "C* - probe (-> backward_sign)": {**CORR, "closure": "backward_sign"},
 "C* - adaptive gamma (-> 0.1 HEAD)": {**CORR, "gamma": 0.1},
 "C* - adaptive gamma (-> 1.5e-3)": {**CORR, "gamma": 1.5e-3},
 "C* + compact AC flux operator": {**CORR, "acop": "compact"},
}
def work(item):
    name, kw = item
    r = run(C(**kw), snapshots=[26.0, 30.0, 100.0, 250.0])
    return name, r
if __name__ == "__main__":
    with mp.Pool(11) as pool:
        res = pool.map(work, list(CASES.items()))
    pickle.dump({n: r for n, r in res}, open("ablate.pkl", "wb"))
    print(f"{'case':<44}" + " ".join(f"{('t='+str(t)):>7}" for t in (25,26,30,60,100,250)) + "   (interface error %)")
    for n, r in res: summary(n, r)
