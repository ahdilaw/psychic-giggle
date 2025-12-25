# type: ignore
# _krt_transfer.py  — compact tilt-transfer with resilient device specs + torch kernels
import json, os, re, math
import pandas as pd, numpy as np

RAW_CSV = "agg/raw_metrics.csv"
DEV_SPECS_JSON = "agg/devices.json"
K_TORCH = "agg/__ai_torch_results_.json"
MODELS_JSON = "models.json"

SOURCE = ("rtx_4090","gpu","torch","")
TARGET = ("rtx_6000_ada","gpu","torch","")
BASELINE_MODEL_ID = "R50"
PROFILE_ORDER = ["S", "P","D"]
EPS = 1e-12

def nkey(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())

def pick_profile(devdict):
    # prefer S, else P, else D, else any with numbers
    for k in PROFILE_ORDER:
        d = devdict.get(k)
        if d and float(d.get("Pd_TFLOPs",0) or 0)>0 and float(d.get("BWd_GBs",0) or 0)>0:
            return d, k
    # try any profile that has numbers
    for k,d in devdict.items():
        if d and float(d.get("Pd_TFLOPs",0) or 0)>0 and float(d.get("BWd_GBs",0) or 0)>0:
            return d, k
    return None, None

def resolve_device_with_fallback(specs, user_key):
    """Return (GF/s, GB/s, resolved_key, profile_used). Fallback across CPU_* siblings/nearest gens."""
    # 1) direct or fuzzy key
    cand = {nkey(k):k for k in specs}
    uk = nkey(user_key)
    key = cand.get(uk)
    if not key:
        for k in specs:
            if uk in nkey(k) or nkey(k) in uk:
                key = k; break
    tried = []
    def prof_for(k):
        tried.append(k)
        prof, prof_name = pick_profile(specs[k])
        if prof:
            Pd = float(prof["Pd_TFLOPs"])*1000.0
            BW = float(prof["BWd_GBs"])
            return Pd, BW, k, prof_name
        return None
    # 2) try direct
    if key and prof_for(key):
        return prof_for(key)
    # 3) CPU sibling swap (GP<->MEM) same gen
    m = re.match(r"(CPU)_(GP|MEM)_v(\d+)$", key or "", flags=re.I)
    if m:
        fam, typ, ver = m.group(1).upper(), m.group(2).upper(), m.group(3)
        sib = f"{fam}_{'MEM' if typ=='GP' else 'GP'}_v{ver}"
        if sib in specs and prof_for(sib): return prof_for(sib)
    # 4) find closest CPU_* version with specs
    if re.match(r"CPU_", key or "", flags=re.I) or re.match(r"CPU_", user_key, flags=re.I):
        # collect CPU_* keys with numeric specs
        avail = []
        for k in specs:
            if not k.upper().startswith("CPU_"): continue
            res = prof_for(k)
            if res: avail.append((k,res))
        if avail:
            # prefer same family (GP/MEM) then closest version number
            def ver_of(k):
                m = re.search(r"_v(\d+)$", k, flags=re.I); return int(m.group(1)) if m else 0
            want_ver = ver_of(key or user_key)
            want_typ = "GP" if "GP" in (key or user_key).upper() else ("MEM" if "MEM" in (key or user_key).upper() else "")
            avail.sort(key=lambda t: (0 if (want_typ and want_typ in t[0].upper()) else 1, abs(ver_of(t[0]) - want_ver)))
            return avail[0][1]
    # 5) last resort: any device with numbers
    for k in specs:
        res = prof_for(k)
        if res: return res
    return None, None, None, None

def load_torch_kernels(path):
    if not os.path.exists(path): return {}
    per = json.load(open(path)).get("per_model_results", {})
    out={}
    for k,cls in per.items():
        out[k]={"classes":cls,
                "F_G": sum(float(c.get("F_c",0.0)) for c in cls.values())/1e9}
    return out

def best_kernel_key(mid, kmap):
    if not kmap: return None
    if mid in kmap: return mid
    midu = mid.upper(); midn = nkey(mid)
    for k in kmap:
        if k.upper()==midu: return k
    for k in kmap:
        if k.upper().startswith(midu) or nkey(k).startswith(midn):
            return k
    # smallest normalized length diff
    return min(kmap.keys(), key=lambda k: abs(len(nkey(k))-len(midn)), default=None)

def T_from_k(kmap, kkey, P_G, BW):
    if not kkey or not (P_G and BW): return None
    t=0.0
    for c in kmap[kkey]["classes"].values():
        Fg=float(c.get("F_c",0.0))/1e9; Ugb=float(c.get("U_c",0.0))/1e9
        t += max(Fg/(P_G+EPS), Ugb/(BW+EPS))
    return t

def safe_div(a,b): 
    return (a/b) if (a not in (None,0) and b not in (None,0) and np.isfinite(a) and np.isfinite(b)) else None

def rf_from_models_json(mid, base_mid):
    if not os.path.exists(MODELS_JSON): return None
    mj=json.load(open(MODELS_JSON))
    f=lambda m: (mj.get(m) or {}).get("flops_g")
    Fm,Fb=f(mid),f(base_mid)
    return None if (Fm is None or Fb in (None,0)) else Fm/Fb

# ==== load ====
df = pd.read_csv(RAW_CSV)
df["framework"]=df["framework"].str.lower()
df["accelerator"]=df["accelerator"].fillna("").astype(str)

dev_specs = json.load(open(DEV_SPECS_JSON))
kmap = load_torch_kernels(K_TORCH)

# groups
def subgrp(d,t,fw,acc):
    m=(df["device"]==d)&(df["tier"]==t)&(df["framework"]==fw)&(df["accelerator"]==acc)
    return df.loc[m].copy()

src=subgrp(*SOURCE); tgt=subgrp(*TARGET)
if src.empty or tgt.empty:
    raise SystemExit(f"Missing rows. source={SOURCE} target={TARGET}")
if BASELINE_MODEL_ID not in src["model_id"].values or BASELINE_MODEL_ID not in tgt["model_id"].values:
    raise SystemExit(f"Baseline {BASELINE_MODEL_ID} missing in source/target.")

# resolve devices (with fallback)
P_X,BW_X,DKX,PROFX = resolve_device_with_fallback(dev_specs, SOURCE[0])
P_Y,BW_Y,DKY,PROFY = resolve_device_with_fallback(dev_specs, TARGET[0])

print(f"TILT SOURCE→TARGET | source={SOURCE} target={TARGET} baseline={BASELINE_MODEL_ID}")
print(f"Resolved devices: X={DKX}({PROFX})  Y={DKY}({PROFY})")
if not (P_X and BW_X and P_Y and BW_Y):
    print("Note: could not find numeric Pd/BW for one or both devices after fallback — predictions may be N/A.")

# baseline tilt anchors
kkey_B = best_kernel_key(BASELINE_MODEL_ID, kmap)
T_B_X = T_from_k(kmap, kkey_B, P_X, BW_X)
T_B_Y = T_from_k(kmap, kkey_B, P_Y, BW_Y)

lat_B_src = float(src.loc[src["model_id"]==BASELINE_MODEL_ID,"s_latency_ms"].iloc[0])
lat_B_tgt = float(tgt.loc[tgt["model_id"]==BASELINE_MODEL_ID,"s_latency_ms"].iloc[0])
mem_B_tgt = float(tgt.loc[tgt["model_id"]==BASELINE_MODEL_ID,"memory_mib"].iloc[0])

rows=[]
for mid in sorted(set(src["model_id"]).intersection(tgt["model_id"])):
    rs = src.loc[src["model_id"]==mid].iloc[0]
    rt = tgt.loc[tgt["model_id"]==mid].iloc[0]

    rX_emp = safe_div(float(rs["s_latency_ms"]), lat_B_src)
    rY_emp = safe_div(float(rt["s_latency_ms"]), lat_B_tgt)

    kkey = best_kernel_key(mid, kmap)
    TX = T_from_k(kmap, kkey, P_X, BW_X)
    TY = T_from_k(kmap, kkey, P_Y, BW_Y)
    rX_pred = safe_div(TX, T_B_X)
    rY_pred = safe_div(TY, T_B_Y)
    theta_pred = safe_div(rY_pred, rX_pred)

    r_src_anchor = rX_emp if rX_emp is not None else rX_pred
    Rr_hat = None if (r_src_anchor is None or theta_pred is None) else r_src_anchor*theta_pred

    Rm = safe_div(float(rt["memory_mib"]), mem_B_tgt)

    # Rf from kernels if both present, else fallback to models.json
    Fm = (kmap.get(kkey) or {}).get("F_G")
    Fb = (kmap.get(kkey_B) or {}).get("F_G")
    Rf = (None if (Fm is None or Fb in (None,0)) else Fm/Fb) or rf_from_models_json(mid, BASELINE_MODEL_ID)

    FRM_hat = None if (Rr_hat in (None,0) or Rf in (None,0) or Rm in (None,0)) else (Rr_hat*Rf*Rm)**(1/3)
    FRM_act = None if (rY_emp in (None,0) or Rf in (None,0) or Rm in (None,0)) else (rY_emp*Rf*Rm)**(1/3)
    err_frm = None if (FRM_act in (None,0) or FRM_hat in (None,)) else (FRM_hat/FRM_act-1)*100.0

    rows.append({
        "model_id": mid, "kernel_key": kkey or "N/A",
        "T_X_pred(s)": TX, "T_Y_pred(s)": TY,
        f"r_emp({SOURCE[0]})": rX_emp, f"r_emp({TARGET[0]})": rY_emp,
        f"r_pred({SOURCE[0]})": rX_pred, f"r_pred({TARGET[0]})": rY_pred,
        "Rr_hat": Rr_hat, "Rf": Rf, "Rm": Rm,
        "FRM_hat": FRM_hat, "FRM_actual": FRM_act, "%err_FRM": err_frm
    })

out = pd.DataFrame(rows).sort_values("model_id").reset_index(drop=True)

def fmt(x):
    if x is None or (isinstance(x,float) and not np.isfinite(x)): return "N/A"
    return f"{x:.6f}" if isinstance(x,float) else x

print(out.applymap(fmt))
