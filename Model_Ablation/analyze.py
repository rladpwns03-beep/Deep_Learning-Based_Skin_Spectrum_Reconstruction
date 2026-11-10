import pandas as pd, numpy as np, json, glob
names=["BiGRU8","GRU8","BiLSTM8","LSTM8","BiRNN8","RNN8","MLP16","GRU12","BiLSTM7","BiRNN16"]
rows=[]
for n in names:
    d=pd.read_csv(f"results/{n}_pairs.csv"); meta=json.load(open(f"results/{n}_meta.json"))
    ps=d.groupby(["subject","method"])[["r_rmse_pp","a_rmse","a_rmse_top","peak_mae"]].mean().unstack("method")
    t=pd.read_csv(f"results/{n}_target.csv").groupby(["subject","method"]).peak_sd.mean().unstack("method")
    res={"model":n,"params":meta["n_params"]}
    for m in ["a_rmse","a_rmse_top","peak_mae","r_rmse_pp"]:
        diff=ps[(m,n)]-ps[(m,"B1")]
        res[m]=ps[(m,n)].mean(); res[m+"_d"]=diff.mean(); res[m+"_se"]=diff.std(ddof=1)/np.sqrt(len(diff)); res[m+"_k"]=int((diff<0).sum())
    diff=t[n]-t["B1"]; res["psd"]=t[n].mean(); res["psd_d"]=diff.mean(); res["psd_se"]=diff.std(ddof=1)/np.sqrt(10); res["psd_k"]=int((diff<0).sum())
    ep=d[d.method==n].epoch; res["ep_med"]=ep.median(); res["ep0"]=int((ep==0).sum())
    # seed sd of subject-mean a_rmse
    sm=d[d.method==n].groupby("seed").a_rmse.mean(); res["seed_sd"]=sm.std(ddof=1)
    rows.append(res)
R=pd.DataFrame(rows); pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
print(R[["model","params","a_rmse","a_rmse_d","a_rmse_se","a_rmse_k","a_rmse_top_d","a_rmse_top_se","a_rmse_top_k"]].to_string(float_format=lambda x:f"{x:.6f}"))
print(R[["model","peak_mae_d","peak_mae_se","peak_mae_k","psd","psd_d","psd_se","psd_k","ep_med","ep0","seed_sd"]].to_string(float_format=lambda x:f"{x:.3e}"))
b1=ps[("a_rmse","B1")].mean(); print("B1 a_rmse", b1)
R.to_csv("summary.csv",index=False)
# paired comparisons vs BiGRU8
base=pd.read_csv("results/BiGRU8_pairs.csv"); bb=base[base.method=="BiGRU8"].groupby("subject")[["a_rmse","a_rmse_top","peak_mae"]].mean()
for n in names[1:]:
    d=pd.read_csv(f"results/{n}_pairs.csv"); x=d[d.method==n].groupby("subject")[["a_rmse","a_rmse_top","peak_mae"]].mean()
    out=[]
    for m in ["a_rmse","a_rmse_top","peak_mae"]:
        df=x[m]-bb[m]; out.append(f"{m}: {df.mean():+.2e} ({df.std(ddof=1)/np.sqrt(10):.1e}) BiGRU better {(df>0).sum()}/10")
    print(n,"minus BiGRU8 |", " | ".join(out))
