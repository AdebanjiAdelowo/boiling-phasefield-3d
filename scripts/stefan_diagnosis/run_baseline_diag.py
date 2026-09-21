import numpy as np, pickle
from stefan_lab import *
from scipy.integrate import quad
snap_t=[24.7+0.5,25.0,26.0,27.0,30.0,40.0,60.0,100.0,150.0,250.0]
c=Cfg(out_every=250)
r=run(c,snapshots=snap_t)
pickle.dump(r,open("baseline_head.pkl","wb"))
rec=r["rec"]; x=r["x"]
def Eana(t):
    d=delta_ana(t); return RHO*CP*quad(lambda s: T_ana(np.array([s]),t)[0]-TS,0,d,epsabs=1e-12)[0]
E0a=Eana(c.t0); d0=delta_ana(c.t0)
print(f"{'t':>7} {'δnum':>8} {'δana':>8} {'err%':>6} | {'ΔM':>8} {'Δδana':>8} {'ρh·ΔM':>9} {'ρh·Δδ':>9} | {'ΣEclamp':>9} {'ΣEwall':>9} {'ΔE':>8} {'ΔE_ana':>8} | {'clipΣ':>9} {'ncr':>3} {'φmin':>7} {'φmax':>6} {'Tmax':>6} {'Tmin':>7}")
for R in rec:
    t=R['t']
    if any(abs(t-s)<0.13 for s in [24.7,25,26,27,30,40,60,80,100,120,150,200,250]):
        print(f"{t:7.2f} {R['d']:8.5f} {R['da']:8.5f} {abs(R['d']-R['da'])/R['da']*100:6.2f} | {R['M']-r['M0']:8.5f} {R['da']-d0:8.5f} {RHO*H*(R['M']-r['M0']):9.2f} {RHO*H*(R['da']-d0):9.2f} | {R['eclamp']:9.2f} {R['ewall']:9.2f} {R['E']-r['E0']:8.2f} {Eana(t)-E0a:8.2f} | {R['clip']:9.2e} {R['ncross']:3d} {R['phimin']:7.4f} {R['phimax']:6.3f} {R['Tmax']:6.2f} {R['Tmin']:7.3f}")
