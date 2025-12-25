# type: ignore
#Usage: streamlit run frm_dashboard.py
import streamlit as st
import pandas as pd, numpy as np
import plotly.graph_objects as go
import glob, os

st.set_page_config(page_title="FRM Explorer", layout="wide")

IN_DIR = "agg/frm"

X_COLS_MAIN = ["frm","frm_e","frm_q","frm_eq"]
Y_COLS      = ["acc_top1","acc_top5","s_latency_ms","s_energy_j","memory_mib"]
Y_MAXIMIZE  = {"acc_top1": True, "acc_top5": True, "s_latency_ms": False, "s_energy_j": False, "memory_mib": False}

def load_all():
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.csv")))
    dfs = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        # baseline from filename: frm_scores_<baseline>
        baseline = base.replace("frm_scores_", "")
        df = pd.read_csv(f)
        if df.empty: 
            continue
        df["source_file"] = os.path.basename(f)
        df["baseline_from_file"] = baseline
        # coerce numeric (handles blanks)
        for c in set(X_COLS_MAIN+Y_COLS).intersection(df.columns):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

@st.cache_data(show_spinner=False)
def data():
    return load_all()

def fmt_group(dev, fw, acc):
    return f"({dev}, {fw})" if (pd.isna(acc) or str(acc).strip()=="") else f"({dev}, {fw}, {acc})"

def pareto_front_xy(x, y, maximize_y=True):
    # minimize x, y maximize/minimize by flag
    order = np.argsort(x)
    x, y = x[order], y[order]
    pf_x, pf_y = [], []
    if maximize_y:
        best = -np.inf
        for xi, yi in zip(x, y):
            if yi > best:
                pf_x.append(xi); pf_y.append(yi); best = yi
    else:
        best = np.inf
        for xi, yi in zip(x, y):
            if yi < best:
                pf_x.append(xi); pf_y.append(yi); best = yi
    return np.array(pf_x), np.array(pf_y)

# ===== Sidebar controls =====
st.sidebar.title("FRM Explorer ⚙️")
df = data()
if df.empty:
    st.warning("No CSVs found in agg/frm/*.csv")
    st.stop()

# Baseline selection (from files or from data)
baselines = sorted(df["baseline_from_file"].dropna().unique().tolist())
sel_baseline = st.sidebar.selectbox("Baseline (from file name)", baselines)

# Filters
groups_all = (df[["device","framework","accelerator"]]
              .drop_duplicates()
              .apply(lambda r: fmt_group(r["device"], r["framework"], r["accelerator"]), axis=1)
              .tolist())
sel_groups = st.sidebar.multiselect("Groups (device, framework, [accelerator])", groups_all, default=groups_all)

models_all = sorted(df["model_id"].dropna().unique().tolist())
sel_models = st.sidebar.multiselect("Models (model_id)", models_all, default=models_all)

# Axes
col1, col2 = st.sidebar.columns(2)
x_main = col1.selectbox("X axis", X_COLS_MAIN, index=0)
y_main = col2.selectbox("Y axis", Y_COLS, index=0)

# Comparison overlays
with st.sidebar.expander("➕ Comparison overlays"):
    add1 = st.checkbox("Add overlay A", value=False)
    if add1:
        o1c1, o1c2 = st.columns(2)
        x1 = o1c1.selectbox("X A", X_COLS_MAIN, index=1, key="xa")
        y1 = o1c2.selectbox("Y A", Y_COLS, index=1, key="ya")
    add2 = st.checkbox("Add overlay B", value=False)
    if add2:
        o2c1, o2c2 = st.columns(2)
        x2 = o2c1.selectbox("X B", X_COLS_MAIN, index=2, key="xb")
        y2 = o2c2.selectbox("Y B", Y_COLS, index=2, key="yb")

# Styling / toggles
st.sidebar.markdown("---")
style = st.sidebar.radio("Style", ["Color", "Grayscale", "Black"], index=0, horizontal=True)
show_pf = st.sidebar.checkbox("Show Pareto fronts", True)
show_vline = st.sidebar.checkbox("Show baseline vline (x=1.0)", True)
show_legend = st.sidebar.checkbox("Show legend", True)
ms = st.sidebar.slider("Marker size", 6, 20, 9)
alpha = st.sidebar.slider("Marker opacity", 0.2, 1.0, 0.85)
xclip = st.sidebar.slider("Clip X max (0 = auto)", 0.0, 10.0, 0.0)
yclip = st.sidebar.slider("Clip Y max (0 = auto)", 0.0, 10.0, 0.0)

# ===== Filtered DF =====
df["group_lbl"] = df.apply(lambda r: fmt_group(r["device"], r["framework"], r["accelerator"]), axis=1)
mask = (df["baseline_from_file"]==sel_baseline) & df["group_lbl"].isin(sel_groups) & df["model_id"].isin(sel_models)
dff = df.loc[mask].copy()
if dff.empty:
    st.info("No data after filters. Try widening your selection.")
    st.stop()

# ===== Plot =====
fig = go.Figure()

# Color logic
if style == "Color":
    # distinct color per group
    color_map = {g:i for i,g in enumerate(sorted(dff["group_lbl"].unique().tolist()))}
    def color_for_group(g): 
        # Plotly qualitative palette cycles by 'category'
        return None  # let plotly pick by legend group/category
    marker_color = None
elif style == "Grayscale":
    color_map = {}
    def color_for_group(g): return "gray"
    marker_color = "gray"
else:  # Black
    color_map = {}
    def color_for_group(g): return "black"
    marker_color = "black"

def add_layer(xcol, ycol, name_suffix="", hollow=False):
    # per-group, per-model traces (legendgroup by group, marker symbol by model)
    # prepare symbols per model
    symbols = ["circle","square","triangle-up","diamond","x","cross","star","hexagon","triangle-down","circle-open","square-dot","diamond-open","triangle-left","triangle-right"]
    sym_map = {m: symbols[i % len(symbols)] for i, m in enumerate(sorted(dff["model_id"].dropna().unique()))}

    # scatter points
    for gname, g in dff.groupby("group_lbl"):
        c = color_for_group(gname) if marker_color is None else marker_color
        # plot each model id as its own trace for legend-driven toggling
        for mid, sub in g.groupby("model_id"):
            xx = pd.to_numeric(sub[xcol], errors="coerce")
            yy = pd.to_numeric(sub[ycol], errors="coerce")
            mask = np.isfinite(xx) & np.isfinite(yy)
            if not mask.any(): 
                continue
            fig.add_trace(go.Scatter(
                x=xx[mask], y=yy[mask],
                mode="markers",
                marker=dict(size=ms, opacity=alpha, color=c, symbol=sym_map.get(mid,"circle"), line=dict(width=1 if hollow else 0, color=c)),
                name=f"{mid} — {gname}{name_suffix}",
                legendgroup=gname,
                showlegend=show_legend
            ))
        # pareto front line (optional)
        if show_pf:
            xx = pd.to_numeric(g[xcol], errors="coerce").values
            yy = pd.to_numeric(g[ycol], errors="coerce").values
            m = np.isfinite(xx) & np.isfinite(yy)
            if m.any():
                pf_x, pf_y = pareto_front_xy(xx[m], yy[m], maximize_y=Y_MAXIMIZE[ycol])
                if len(pf_x) >= 2:
                    fig.add_trace(go.Scatter(
                        x=pf_x, y=pf_y, mode="lines",
                        line=dict(width=2, color=(color_for_group(gname) if marker_color is None else marker_color)),
                        name=f"Pareto — {gname}{name_suffix}",
                        legendgroup=gname,
                        showlegend=show_legend
                    ))

# main layer
add_layer(x_main, y_main, name_suffix="")

# overlays (hollow markers + dashed PF)
if st.session_state.get("xa") and st.session_state.get("ya") and st.checkbox("Use hollow markers for overlays", True, key="hollow1", help="Better visual separation"):
    pass  # state created
if 'xa' in st.session_state and 'ya' in st.session_state and st.session_state.get('xa') and st.session_state.get('ya'):
    add_layer(st.session_state['xa'], st.session_state['ya'], name_suffix=" (A)", hollow=True)
if 'xb' in st.session_state and 'yb' in st.session_state and st.session_state.get('xb') and st.session_state.get('yb'):
    add_layer(st.session_state['xb'], st.session_state['yb'], name_suffix=" (B)", hollow=True)

# baseline vline
shapes = []
if show_vline:
    shapes.append(dict(type="line", x0=1.0, x1=1.0, y0=0, y1=1, xref='x', yref='paper',
                       line=dict(color="rgba(120,120,120,0.7)", width=1, dash="dash")))

# axes bounds
xaxis = dict(title=x_main)
yaxis = dict(title=y_main)
if xclip > 0: xaxis["range"] = [0, xclip]
if yclip > 0: yaxis["range"] = [0, yclip]

fig.update_layout(
    height=800, margin=dict(l=10,r=10,t=50,b=10),
    shapes=shapes,
    xaxis=xaxis, yaxis=yaxis,
    legend=dict(itemsizing="trace", orientation="v")
)

st.subheader("📊 Interactive FRM Scatter")
st.plotly_chart(fig, use_container_width=True, theme=None)

# Data table (filtered)
st.subheader("🧾 Data (filtered)")
st.dataframe(dff.sort_values(["device","framework","accelerator","model_id"]))

# Exports
st.markdown("### 💾 Export")
colA, colB = st.columns(2)
with colA:
    svg = fig.to_image(format="svg")
    st.download_button("Download SVG", data=svg, file_name=f"frm_{x_main}_vs_{y_main}.svg", mime="image/svg+xml")
with colB:
    png = fig.to_image(format="png", scale=2)
    st.download_button("Download PNG", data=png, file_name=f"frm_{x_main}_vs_{y_main}.png", mime="image/png")
