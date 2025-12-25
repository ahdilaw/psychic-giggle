#!/usr/bin/env python3
# Usage: build_raw_metrics.py
#type: ignore
import os, re, sys, glob, json, pandas as pd
from typing import Dict, Tuple

CACHE_FILE = "model_id_cache.json"

BASE = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\ahmed\Desktop\results\secure\data"
CPU_DIR = os.path.join(BASE, "cpu")
GPU_DIR = os.path.join(BASE, "gpu")
EDGE_DIR = os.path.join(BASE, "edge")
OUT_CSV = os.path.join(BASE, "raw_metrics.csv")

# --------- normalize helpers ----------
def norm(s: str) -> str:
    return str(s).strip()

def norm_id(s: str) -> str:
    return norm(s).lower().replace(" ", "_")

def norm_cols(cols):
    return [norm(c).lower().replace(" ", "_").replace(".", "_") for c in cols]

# --------- header alias maps ----------
CPU_GPU_ALIASES = {
    "model_name": ["model_name", "model", "name"],
    "top1_acc"  : ["top1_acc", "acc_top1", "top1"],
    "top5_acc"  : ["top5_acc", "acc_top5", "top5"],
    "lat_p50_ms": ["lat_p50_ms", "lat_p50", "latency_p50_ms", "latency_p50", "lat_median_ms", "median_latency_ms", "p50_latency_ms"],
    "steady_host_mean_delta_median_mib": [
        "steady_host_mean_delta_median_mib",
        "host_steady_mean_delta_median_mib",
        "host_steady_mean_delta_mib_median",
        "steady_host_mean_delta_mib_median",
        "host_steady_mem_delta_med_mib"
    ],
    "energy_median_j_per_sample": [
        "energy_median_j_per_sample",
        "median_energy_j_per_sample",
        "energy_j_median_per_sample",
        "median_j_per_sample",
        "energy_per_sample_j_median"
    ],
}

EDGE_ALIASES = {
    "status": ["status","Status"],
    "device_name": ["device_name","Device name","Device_name"],
    "accelerator": ["accelerator","Accelerator"],
    "steady_ms": ["steady_inference_(ms)","Steady inference (ms)","steady_inference_ms","steady_inference"],
    "peak_mem_mb": ["peak_memory_(mb)","Peak memory (MB)","peak_memory_mb","peak_memory"],
}

def remap_columns(df: pd.DataFrame, aliases: Dict[str, list]) -> pd.DataFrame:
    df = df.copy()
    df.columns = norm_cols(df.columns)
    rename = {}
    for canon, alts in aliases.items():
        found = None
        for a in alts:
            a = a.lower().replace(" ", "_").replace(".", "_")
            if a in df.columns:
                found = a; break
        if found is not None and found != canon:
            rename[found] = canon
    if rename:
        df = df.rename(columns=rename)
    return df

# --------- model → model_id canonicalization ----------
ALLOWED_CODES = {"CN","DE","DN","EN","IN","LE","MN","MV2","MV3","MX","R18","R50","SN"}
MODEL_ID_RULES = [
    (re.compile(r"^convnext[_\-]?tiny(\.|$)"), "CN"),
    (re.compile(r"^deit(_tiny)?_distilled_patch16_224(\.|$)"), "DE"),
    (re.compile(r"^densenet121(\.|$)"), "DN"),
    (re.compile(r"^efficientnet[_\-]?b0(\.|$)"), "EN"),
    (re.compile(r"^inception[_\-]?v3(\.|$)"), "IN"),
    (re.compile(r"^levit[_\-]?128s(\.|$)"), "LE"),
    (re.compile(r"^mnasnet1[._\-]?0(\.|$)"), "MN"),
    (re.compile(r"^mobilenet[_\-]?v2(\.|$)"), "MV2"),
    (re.compile(r"^mobilenet[_\-]?v3[_\-]?small(\.|$)"), "MV3"),
    (re.compile(r"^mobilevit[_\-]?xxs(\.|$)"), "MX"),
    (re.compile(r"^resnet18(\.|$)"), "R18"),
    (re.compile(r"^resnet50(_v1(_mlperf)?)?(\.|$)"), "R50"),
    (re.compile(r"^squeezenet1[_\-]?1(\.|$)"), "SN"),
]

def _clean_stem(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"\.(onnx|tflite|pth|pt|pb|gguf)$", "", s)
    s = re.sub(r"[-_][0-9a-f]{6,}$", "", s)
    s = s.replace("torchvision_", "").replace(".", "_")
    s = re.sub(r"__+", "_", s)
    return s

# load cache once
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        MODEL_CACHE = json.load(f)
else:
    MODEL_CACHE = {}

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(MODEL_CACHE, f, indent=2)

def pick_model_id(name: str) -> str:
    stem = _clean_stem(name)

    # strict rules first
    for pat, mid in MODEL_ID_RULES:
        if pat.search(stem):
            return mid

    # cached manual mapping?
    if stem in MODEL_CACHE:
        return MODEL_CACHE[stem]

    # fallback → ask you in terminal
    print(f"[ask] Unknown model name: '{name}' (stem='{stem}')")
    print(f"Allowed codes: {sorted(ALLOWED_CODES)}")
    mid = input("Enter correct code for this model (or leave blank to skip): ").strip().upper()

    if not mid or mid not in ALLOWED_CODES:
        print(f"[skip] Skipping {name}, no valid code given.")
        return None

    # cache it
    MODEL_CACHE[stem] = mid
    save_cache()
    return mid

# Accuracy Mapper
def build_cpu_acc_map(df_cpu_gpu: pd.DataFrame):
    acc_map = {}
    if df_cpu_gpu.empty: 
        return acc_map
    cpu_only = df_cpu_gpu[df_cpu_gpu["tier"] == "cpu"].copy()
    if cpu_only.empty:
        return acc_map
    # keep first non-null per (framework, model_id)
    cpu_only = cpu_only.sort_values(["framework", "model_id"])
    for (fw, mid), grp in cpu_only.groupby(["framework", "model_id"]):
        a1 = grp["acc_top1"].dropna()
        a5 = grp["acc_top5"].dropna()
        acc_map[(fw, mid)] = (
            float(a1.iloc[0]) if len(a1) else None,
            float(a5.iloc[0]) if len(a5) else None,
        )
    return acc_map

# --------- CPU/GPU ingest from *.ods.csv ----------
def detect_framework_from_filename(path: str) -> str:
    fname = os.path.basename(path).lower()
    if "onnx" in fname: return "onnx"
    if "torch" in fname: return "torch"
    if "tflite" in fname: return "tflite"
    return "unknown"

def collect_cpu_gpu() -> pd.DataFrame:
    rows = []
    for tier_dir, tier in [(CPU_DIR, "cpu"), (GPU_DIR, "gpu")]:
        if not os.path.isdir(tier_dir): continue
        for dev in os.listdir(tier_dir):
            res_dir = os.path.join(tier_dir, dev, "results")
            if not os.path.isdir(res_dir): continue
            for path in glob.glob(os.path.join(res_dir, "*.ods.csv")):
                fw = detect_framework_from_filename(path)
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                df = remap_columns(df, CPU_GPU_ALIASES)
                if "model_name" not in df.columns or "lat_p50_ms" not in df.columns:
                    continue
                # energy optional; memory alias handled
                df["acc_top1"] = pd.to_numeric(df.get("top1_acc"), errors="coerce")
                df["acc_top5"] = pd.to_numeric(df.get("top5_acc"), errors="coerce")
                df["s_latency_ms"] = pd.to_numeric(df.get("lat_p50_ms"), errors="coerce")
                df["memory_mib"] = pd.to_numeric(df.get("steady_host_mean_delta_median_mib"), errors="coerce")
                df["s_energy_j"] = pd.to_numeric(df.get("energy_median_j_per_sample"), errors="coerce")

                for _, r in df.iterrows():
                    if pd.isna(r["s_latency_ms"]) or float(r["s_latency_ms"]) <= 0: 
                        continue
                    model_id = pick_model_id(str(r["model_name"]))
                    energy = 0.0 if tier in ("cpu",) else (float(r["s_energy_j"]) if pd.notna(r["s_energy_j"]) else 0.0)
                    rows.append({
                        "device": norm_id(dev),
                        "tier": tier,
                        "model_id": model_id,
                        "framework": fw,
                        "accelerator": "",
                        "acc_top1": float(r["acc_top1"]) if pd.notna(r["acc_top1"]) else None,
                        "acc_top5": float(r["acc_top5"]) if pd.notna(r["acc_top5"]) else None,
                        "s_latency_ms": float(r["s_latency_ms"]),
                        "s_energy_j": energy,
                        "memory_mib": float(r["memory_mib"]) if pd.notna(r["memory_mib"]) else None,
                    })
    return pd.DataFrame(rows)

# --------- EDGE ingest from frm_bench_*.csv ----------
def infer_model_from_edge_filename(path: str) -> str:
    # e.g., frm_bench_mobilenet_v2.csv → mobilenet_v2
    stem = os.path.basename(path).lower()
    stem = re.sub(r"\.csv$", "", stem)
    stem = re.sub(r"^frm_bench_", "", stem)
    return stem

def collect_edge(acc_map) -> pd.DataFrame:
    rows = []
    if not os.path.isdir(EDGE_DIR): 
        return pd.DataFrame(rows)
    for path in glob.glob(os.path.join(EDGE_DIR, "*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df = remap_columns(df, EDGE_ALIASES)
        need = {"status","device_name","accelerator","steady_ms","peak_mem_mb"}
        if not need.issubset(set(df.columns)): 
            continue

        model_guess = infer_model_from_edge_filename(path)
        model_id = pick_model_id(model_guess)
        fw = "tflite"  # edge is always tflite

        ok = df[df["status"].astype(str).str.lower().eq("succeeded")].copy()
        if ok.empty: 
            continue
        ok["s_latency_ms"] = pd.to_numeric(ok["steady_ms"], errors="coerce")
        ok["memory_mib"]   = pd.to_numeric(ok["peak_mem_mb"], errors="coerce")

        # pull acc from CPU of same framework (tflite) & same model_id
        acc1, acc5 = acc_map.get((fw, model_id), (None, None))

        for _, r in ok.iterrows():
            if pd.isna(r["s_latency_ms"]) or float(r["s_latency_ms"]) <= 0:
                continue
            rows.append({
                "device": norm_id(str(r["device_name"])),
                "tier": "edge",
                "model_id": model_id,
                "framework": fw,
                "accelerator": norm_id(str(r["accelerator"])) if pd.notna(r["accelerator"]) else "",
                "acc_top1": acc1,
                "acc_top5": acc5,
                "s_latency_ms": float(r["s_latency_ms"]),
                "s_energy_j": 0.0,
                "memory_mib": float(r["memory_mib"]) if pd.notna(r["memory_mib"]) else None,
            })
    return pd.DataFrame(rows)

# --------- main ----------
def main():
    cpu_gpu = collect_cpu_gpu()
    acc_map = build_cpu_acc_map(cpu_gpu)   # <-- new
    edge = collect_edge(acc_map)           # <-- changed signature
    df = pd.concat([cpu_gpu, edge], ignore_index=True) if not cpu_gpu.empty or not edge.empty else pd.DataFrame(
        columns=["device","tier","model_id","framework","accelerator","acc_top1","acc_top5","s_latency_ms","s_energy_j","memory_mib"]
    )
    cols = ["device","tier","model_id","framework","accelerator","acc_top1","acc_top5","s_latency_ms","s_energy_j","memory_mib"]
    df = df[cols].sort_values(["tier","device","model_id","framework"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✔ wrote {OUT_CSV} with {len(df)} rows")

if __name__ == "__main__":
    main()
