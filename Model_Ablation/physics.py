import numpy as np, pandas as pd
from ablation import load, WL, HBI, FREE, peak
a,s,t=load("data"); subs=sorted(set(s))
p=np.zeros(31); p[9]=1; p[7]=-.5; p[11]=-.5   # Peak functional (470/490/510)
# Oracle in-sample per-subject slope
bs={}
for sid in subs:
    m=s==sid; h=a[m,HBI]; A=a[m]; hc=h-h.mean(); bs[sid]=hc@(A-A.mean(0))/(hc@hc)
R=pd.read_csv("results/BiGRU8_resid06.csv")
rc=[f"r{w}" for w in WL]; bc=[f"b1_{w}" for w in WL]
def rough(v): v=np.asarray(v); return np.linalg.norm(np.diff(v,2))/np.linalg.norm(v)
rows=[]
for sid in subs:
    g=R[R.subject==sid]; r=g[rc].to_numpy(); b1=g[bc].to_numpy()[0]
    rm=r.mean(0); d=bs[sid]-b1
    cos=(rm[FREE]@d[FREE])/np.linalg.norm(rm[FREE])/np.linalg.norm(d[FREE])
    rows.append(dict(subject=sid, cos_r_d=cos, peak_r=p@rm, peak_d=p@d, rough_b1=rough(b1), rough_bs=rough(bs[sid]),
                     rough_tot=np.mean([rough(b1+x) for x in r]), rough_r=np.mean([rough(x) for x in r]),
                     rnorm=np.linalg.norm(rm)/np.linalg.norm(b1), dnorm=np.linalg.norm(d)/np.linalg.norm(b1)))
D=pd.DataFrame(rows); pd.set_option("display.width",200)
print(D.round(4).to_string())
print("cos>0:",(D.cos_r_d>0).sum(),"mean cos",D.cos_r_d.mean())
print("corr peak_r vs peak_d:",np.corrcoef(D.peak_r,D.peak_d)[0,1], "sign agree", (np.sign(D.peak_r)==np.sign(D.peak_d)).sum())
print("rough means:",D[["rough_b1","rough_bs","rough_tot","rough_r"]].mean().round(4).to_dict())
# curvature: pooled within-subject quadratic A = c0_s + c1 h + c2 h^2
X1=[];X2=[];Y=[];G=[]
for sid in subs:
    m=s==sid; h=a[m,HBI]; A=a[m]
    X1.append(h-h.mean()); X2.append(h**2-(h**2).mean()); Y.append(A-A.mean(0))
X=np.column_stack([np.concatenate(X1),np.concatenate(X2)]); Yv=np.vstack(Y)
coef,res,_,_=np.linalg.lstsq(X,Yv,rcond=None)
# subject-level c2: per subject quadratic fit, then mean/SE across subjects
c2s=[]
for sid in subs:
    m=s==sid; h=a[m,HBI]; A=a[m]
    V=np.column_stack([np.ones_like(h),h,h**2]); c=np.linalg.lstsq(V,A,rcond=None)[0]; c2s.append(c[2])
c2s=np.array(c2s); c2m=c2s.mean(0); c2se=c2s.std(0,ddof=1)/np.sqrt(10); kpos=(c2s>0).sum(0)
print("wl c2_mean c2_se n_pos")
for i,w in enumerate(WL):
    if w%30==10 or w in (490,570,700,400): print(w, f"{c2m[i]:+.3f} {c2se[i]:.3f} {kpos[i]}")
# BiGRU residual vs input h at target 0.6: dr/dh per wavelength, pooled within subject
drdh=[]
for sid in subs:
    g=R[R.subject==sid]; tr=g.trial.to_numpy()
    hmap={int(tt):hh for tt,hh in zip(t[s==sid],a[s==sid,HBI])}; h=np.array([hmap[int(x)] for x in tr])
    r=g[rc].to_numpy(); hc=h-h.mean(); drdh.append(hc@(r-r.mean(0))/(hc@hc))
drdh=np.array(drdh); dm=drdh.mean(0)
# Theory: effective slope b + c2 (h + h*)  => d r / d h ~ c2  (at fixed h*)
ok=FREE
print("corr across wavelengths dr/dh vs c2 (pooled coef):", np.corrcoef(dm[ok], coef[1][ok])[0,1])
print("corr dr/dh vs c2 subject-mean:", np.corrcoef(dm[ok], c2m[ok])[0,1])
pd.DataFrame({"wl":WL,"b1_all":fit_b1 if False else np.nan, "c2_pooled":coef[1],"c2_mean":c2m,"c2_se":c2se,"c2_npos":kpos,"drdh_mean":dm}).to_csv("curv.csv",index=False)
D.to_csv("phys_subject.csv",index=False)
# B1 slope over all subjects (shape)
ac=a.copy(); hc=a[:,HBI].copy()
for sid in subs:
    m=s==sid; ac[m]-=ac[m].mean(0); hc[m]-=hc[m].mean()
b1all=hc@ac/(hc@hc)
print("b1 all:", dict(zip(WL[::3], b1all[::3].round(3))))
np.save("b1all.npy",b1all); np.save("bs.npy",np.array([bs[x] for x in subs]))
