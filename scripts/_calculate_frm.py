# type: ignore
import pandas as pd, json, os

INPUT_CSV = "agg/raw_metrics.csv"
MODELS_JSON = "agg/models.json"
OUT_DIR = "agg/frm"

w_mem = w_lat = w_flp = w_eng = w_acc = 1.0

with open(MODELS_JSON) as f:
    models_info = json.load(f)

df = pd.read_csv(INPUT_CSV)
groups = df.groupby(["device","framework","accelerator"], dropna=False)

for baseline_id in models_info.keys():
    rows = []
    for (dev,fw,acc), g in groups:
        g = g.reset_index(drop=True)
        base = g[g["model_id"].str.lower()==baseline_id.lower()]
        if base.empty: continue
        b = base.iloc[0]

        b_mem, b_lat, b_acc = b["memory_mib"], b["s_latency_ms"], b["acc_top1"]
        b_flp = models_info.get(b["model_id"],{}).get("flops_g",1.0)
        has_energy = "s_energy_j" in g.columns and g["s_energy_j"].notna().any()
        if has_energy: b_eng = b["s_energy_j"]

        for _,r in g.iterrows():
            flp = models_info.get(r["model_id"],{}).get("flops_g",1.0)
            rm = r["memory_mib"]/b_mem if b_mem else 1
            rl = r["s_latency_ms"]/b_lat if b_lat else 1
            rf = flp/b_flp if b_flp else 1
            ra = b_acc/r["acc_top1"] if r["acc_top1"] else 1

            frm = (rm**w_mem * rl**w_lat * rf**w_flp)**(1/3)
            frm_q = frm * ra**w_acc

            if has_energy:
                re = r["s_energy_j"]/b_eng if b_eng else ""
                frm_e = (rm**w_mem * rl**w_lat * rf**w_flp * (re if re else 1)**w_eng)**(1/4) if re else ""
                frm_eq = frm_e * ra**w_acc if re else ""
            else:
                re = frm_e = frm_eq = ""

            rows.append([
                dev,fw,acc,r["tier"],r["model_id"],
                r["acc_top1"],r["acc_top5"],r["s_latency_ms"],r.get("s_energy_j",""),r["memory_mib"],
                rm,rl,rf,re,ra,(w_mem,w_lat,w_flp,w_eng,w_acc),
                frm,frm_e,frm_q,frm_eq
            ])

    if rows:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = f"{OUT_DIR}/frm_scores_{baseline_id}.csv"
        cols = [
            "device","framework","accelerator","tier","model_id",
            "acc_top1","acc_top5","s_latency_ms","s_energy_j","memory_mib",
            "ratio_mem","ratio_lat","ratio_flops","ratio_energy","ratio_acc",
            "weights","frm","frm_e","frm_q","frm_eq"
        ]
        pd.DataFrame(rows,columns=cols).to_csv(out_path,index=False)
        print(f"✔ Wrote {len(rows)} rows to {out_path}")