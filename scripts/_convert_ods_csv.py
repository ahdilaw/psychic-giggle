#!/usr/bin/env python3
#type: ignore
#Usage: _convert_ods_csv.py [/path/to/data]
import os, sys, pandas as pd

BASE = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\ahmed\Desktop\results\secure\data"
CPU_DIR = os.path.join(BASE, "cpu")
GPU_DIR = os.path.join(BASE, "gpu")

ODS_FILENAMES = [
    "frm_onnx_results.ods",
    "frm_torch_results.ods",
    "frm_tflite_results.ods",
]

SUMMARY_SHEET = "Summary"

def normalize_headers(cols):
    return [str(c).strip().lower().replace(" ", "_").replace(".", "_") for c in cols]

def convert(path_ods: str):
    cache_csv = path_ods + ".csv"
    try:
        df = pd.read_excel(path_ods, sheet_name=SUMMARY_SHEET)
        df.columns = normalize_headers(df.columns)
        df.to_csv(cache_csv, index=False)
        print(f"✔ Wrote {cache_csv}")
    except Exception as e:
        print(f"✘ {path_ods}: {e}")

def walk_and_convert(root: str):
    for dev in os.listdir(root):
        results = os.path.join(root, dev, "results")
        if not os.path.isdir(results):
            continue
        for name in ODS_FILENAMES:
            p = os.path.join(results, name)
            if os.path.isfile(p):
                convert(p)

def main():
    for root in (CPU_DIR, GPU_DIR):
        if os.path.isdir(root):
            walk_and_convert(root)

if __name__ == "__main__":
    main()
