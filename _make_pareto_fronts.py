import pandas as pd, numpy as np, matplotlib.pyplot as plt, glob, os
from itertools import cycle
from matplotlib import colormaps as cm

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

IN_DIR  = "agg/frm"
OUT_DIR = "agg/frm/plots_svg"
os.makedirs(OUT_DIR, exist_ok=True)

x_cols = ["frm","frm_e","frm_q","frm_eq"]
y_cols = ["acc_top1","acc_top5","s_latency_ms","s_energy_j","memory_mib"]
Y_MAXIMIZE = {"acc_top1": True, "acc_top5": True, "s_latency_ms": False, "s_energy_j": False, "memory_mib": False}

filled_markers = ["o","s","^","v","D","p","P","X","h","H","<",">","d","8","*"]
model_markers = {}
marker_cycle  = cycle(filled_markers)
tab20 = cm["tab20"]; color_cycle = cycle(tab20.colors)

def fmt_group(gk):
    dev, fw, acc = gk
    return f"({dev}, {fw})" if (pd.isna(acc) or acc=="") else f"({dev}, {fw}, {acc})"

def coerce_num(s): return pd.to_numeric(s, errors="coerce")

def pareto_front_xy(x, y, maximize_y=True):
    order = np.argsort(x); x, y = x[order], y[order]
    pf_x, pf_y = [], []
    if maximize_y:
        best = -np.inf
        for xi, yi in zip(x, y):
            if yi > best: pf_x.append(xi); pf_y.append(yi); best = yi
    else:
        best = np.inf
        for xi, yi in zip(x, y):
            if yi < best: pf_x.append(xi); pf_y.append(yi); best = yi
    return np.array(pf_x), np.array(pf_y)

files = sorted(glob.glob(f"{IN_DIR}/*.csv"))
iterable = tqdm(files, desc="Rendering FRM plots", ncols=80) if TQDM else files

for path in iterable:
    df = pd.read_csv(path)
    if df.empty: 
        continue
    for c in set(x_cols + y_cols).intersection(df.columns):
        df[c] = coerce_num(df[c])

    fig, axes = plt.subplots(len(y_cols), len(x_cols), figsize=(36, 28))
    fig.subplots_adjust(right=0.8, wspace=0.28, hspace=0.28)

    group_colors = {}
    for gkey, g in df.groupby(["device","framework","accelerator"], dropna=False):
        color = group_colors.get(gkey) or next(color_cycle); group_colors[gkey] = color
        for mid in g["model_id"].dropna().unique():
            if mid not in model_markers: model_markers[mid] = next(marker_cycle)

        for i, yc in enumerate(y_cols):
            y = g[yc].values
            for j, xc in enumerate(x_cols):
                ax = axes[i, j]
                x = g[xc].values
                mask = np.isfinite(x) & np.isfinite(y)
                ax.axvline(1.0, color="0.7", ls="--", lw=0.8)
                if not mask.any(): 
                    continue
                for mid, subm in g.loc[mask].groupby("model_id"):
                    mk = model_markers.get(mid, "o")
                    ax.scatter(subm[xc].values, subm[yc].values, marker=mk, s=28, c=[color], alpha=0.85)
                pf_x, pf_y = pareto_front_xy(x[mask], y[mask], maximize_y=Y_MAXIMIZE[yc])
                if len(pf_x) >= 2: 
                    ax.plot(pf_x, pf_y, color=color, lw=1.6, alpha=0.95)

    for i, yc in enumerate(y_cols):
        for j, xc in enumerate(x_cols):
            ax = axes[i, j]
            ax.set_xlabel(xc); ax.set_ylabel(yc); ax.grid(True, alpha=0.25)

    model_handles = [plt.Line2D([0],[0], marker=mk, linestyle='None', markersize=8, color="k", label=str(mid))
                     for mid, mk in model_markers.items()]
    group_handles = [plt.Line2D([0],[0], color=col, lw=3, label=fmt_group(gk)) for gk, col in group_colors.items()]
    fig.legend(handles=(model_handles + group_handles), loc='center left',
               bbox_to_anchor=(0.82, 0.5), fontsize=8, frameon=False)

    fig.suptitle(os.path.basename(path), fontsize=16)
    out_svg = os.path.join(OUT_DIR, os.path.basename(path).replace(".csv", ".svg"))
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

print(f"Wrote plots to {OUT_DIR}")