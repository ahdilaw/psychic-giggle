#!/usr/bin/env python3
import pandas as pd, json, sys, os

CSV = sys.argv[1] if len(sys.argv) > 1 else "raw_metrics.csv"
OUT = os.path.splitext(CSV)[0] + "_summary.json"

df = pd.read_csv(CSV)

summary = {
    "unique_devices": int(df["device"].nunique()),
    "devices_by_tier": df["tier"].value_counts().to_dict(),
    "devices_list_by_tier": {
        t: sorted(df[df["tier"]==t]["device"].unique().tolist())
        for t in df["tier"].unique()
    },
    "total_inferences": len(df),
    "inferences_by_tier": df["tier"].value_counts().to_dict(),
    "inferences_by_framework": df["framework"].value_counts().to_dict(),
    "inferences_by_model": df["model_id"].value_counts().to_dict(),
}

with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)

print(f"✔ wrote {OUT}")
