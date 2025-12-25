"""
Predicting Ranking Flips Using Baseline Proxy
==============================================

Idea: If a model ranks #2 on GPU but #4 on Edge, can we predict this flip
by treating Edge's baseline measurements as a "proxy model" on GPU?

The hypothesis: The proxy model's FRM on GPU reveals hardware characteristics
that cause certain models to flip rankings.
"""

import pandas as pd
import numpy as np
import json
from scipy.stats import spearmanr

FRM_DIR = "agg/frm"
MODELS_JSON = "agg/models.json"

with open(MODELS_JSON) as f:
    models_info = json.load(f)

def create_group_id(row):
    acc = row['accelerator'] if pd.notna(row['accelerator']) else 'none'
    return f"{row['device']}|{row['framework']}|{acc}"

def get_device_tier(device_name):
    device_lower = str(device_name).lower()
    if any(x in device_lower for x in ['a100', 'h100', 'h200', 'l40', 'rtx', 'gpu']) and 'cpu' not in device_lower:
        return 'gpu'
    elif any(x in device_lower for x in ['cpu', 'gp_v', 'mem_v']):
        return 'cpu'
    else:
        return 'edge'

print("="*80)
print("RANKING FLIP PREDICTION USING BASELINE PROXY")
print("="*80)

baseline = "R50"
df = pd.read_csv(f"{FRM_DIR}/frm_scores_{baseline}.csv")

for col in ['frm', 'ratio_flops', 'ratio_lat', 'ratio_mem', 's_latency_ms', 'memory_mib', 'acc_top1']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['group_id'] = df.apply(create_group_id, axis=1)
df['device_tier'] = df['device'].apply(get_device_tier)

for model_id in df['model_id'].unique():
    if model_id in models_info:
        mask = df['model_id'] == model_id
        df.loc[mask, 'flops_g'] = models_info[model_id].get('flops_g', np.nan)

reference_device = 'h100_sxm|torch|none'
target_device = 'asus_zenfone_max_pro_m1|tflite|cpu'

print(f"\nReference device (dev1): {reference_device}")
print(f"Target device (dev2): {target_device}")

ref_data = df[df['group_id'] == reference_device].copy()
target_data = df[df['group_id'] == target_device].copy()

ref_agg = ref_data.groupby('model_id').agg({
    's_latency_ms': 'mean',
    'memory_mib': 'mean',
    'flops_g': 'mean',
    'frm': 'mean',
}).reset_index()

target_agg = target_data.groupby('model_id').agg({
    's_latency_ms': 'mean',
    'memory_mib': 'mean',
    'flops_g': 'mean',
    'frm': 'mean',
}).reset_index()

# Get baseline on both devices
ref_baseline_lat = ref_agg[ref_agg['model_id'] == 'R50']['s_latency_ms'].values[0]
ref_baseline_mem = ref_agg[ref_agg['model_id'] == 'R50']['memory_mib'].values[0]
ref_baseline_flops = models_info['R50']['flops_g']

target_baseline_lat = target_agg[target_agg['model_id'] == 'R50']['s_latency_ms'].values[0]
target_baseline_mem = target_agg[target_agg['model_id'] == 'R50']['memory_mib'].values[0]

print(f"\n{'='*80}")
print("STEP 1: Identify Ranking Flips")
print(f"{'='*80}")

# Get rankings on both devices
common_models = sorted(set(ref_agg['model_id']) & set(target_agg['model_id']))

ref_agg['rank'] = ref_agg['frm'].rank(method='min', ascending=False)
target_agg['rank'] = target_agg['frm'].rank(method='min', ascending=False)

rankings = []
for model in common_models:
    ref_frm = ref_agg[ref_agg['model_id'] == model]['frm'].values[0]
    ref_rank = ref_agg[ref_agg['model_id'] == model]['rank'].values[0]
    
    target_frm = target_agg[target_agg['model_id'] == model]['frm'].values[0]
    target_rank = target_agg[target_agg['model_id'] == model]['rank'].values[0]
    
    rank_change = int(target_rank - ref_rank)
    
    rankings.append({
        'model': model,
        'ref_frm': ref_frm,
        'ref_rank': int(ref_rank),
        'target_frm': target_frm,
        'target_rank': int(target_rank),
        'rank_change': rank_change,
    })

rank_df = pd.DataFrame(rankings).sort_values('ref_rank')

print(f"\n{'Model':<6} {'GPU FRM':<9} {'GPU Rank':<9} {'Edge FRM':<9} {'Edge Rank':<10} {'Change':<8}")
print("─"*80)
for _, row in rank_df.iterrows():
    change_str = f"{row['rank_change']:+d}" if row['rank_change'] != 0 else "  0"
    marker = "⚠️ FLIP" if abs(row['rank_change']) >= 2 else ""
    print(f"{row['model']:<6} {row['ref_frm']:>8.3f} {row['ref_rank']:>9.0f} "
          f"{row['target_frm']:>8.3f} {row['target_rank']:>10.0f} {change_str:>8} {marker}")

flips = rank_df[abs(rank_df['rank_change']) >= 2]
print(f"\n🎯 Found {len(flips)} significant ranking flips (±2 positions or more)")

print(f"\n{'='*80}")
print("STEP 2: Create Proxy Model from Target Baseline")
print(f"{'='*80}")

# Create "proxy" models: What if we treated target device's models as if they were on reference device?
# We take target device's latency/memory and compute "what would the FRM be on reference device?"

print(f"\nTarget baseline (R50) measurements:")
print(f"  Latency: {target_baseline_lat:.2f} ms")
print(f"  Memory: {target_baseline_mem:.2f} MB")

print(f"\nCreating proxy: 'What if we measured a model with these characteristics on GPU?'")

# For each model on target device, create its "proxy" on reference device
proxy_frms = []

for model in common_models:
    # Get target device measurements
    target_lat = target_agg[target_agg['model_id'] == model]['s_latency_ms'].values[0]
    target_mem = target_agg[target_agg['model_id'] == model]['memory_mib'].values[0]
    target_flops = target_agg[target_agg['model_id'] == model]['flops_g'].values[0]
    
    # Compute "proxy FRM" = if we had these latency/memory values on reference device
    # Normalize against reference baseline (as if measuring on GPU)
    proxy_lat_ratio = target_lat / ref_baseline_lat
    proxy_mem_ratio = target_mem / ref_baseline_mem
    proxy_flops_ratio = target_flops / ref_baseline_flops
    
    proxy_frm = (proxy_lat_ratio * proxy_mem_ratio * proxy_flops_ratio) ** (1/3)
    
    # Get actual reference FRM
    ref_frm = ref_agg[ref_agg['model_id'] == model]['frm'].values[0]
    
    proxy_frms.append({
        'model': model,
        'ref_frm': ref_frm,
        'proxy_frm': proxy_frm,
        'frm_diff': proxy_frm - ref_frm,
        'frm_ratio': proxy_frm / ref_frm if ref_frm > 0 else 0,
    })

proxy_df = pd.DataFrame(proxy_frms)

print(f"\n{'Model':<6} {'GPU FRM':<9} {'Proxy FRM':<10} {'Diff':<9} {'Ratio':<8}")
print("─"*80)
for _, row in proxy_df.iterrows():
    print(f"{row['model']:<6} {row['ref_frm']:>8.3f} {row['proxy_frm']:>9.3f} "
          f"{row['frm_diff']:>8.3f} {row['frm_ratio']:>7.2f}x")

print(f"\n{'='*80}")
print("STEP 3: Correlate Proxy Characteristics with Ranking Flips")
print(f"{'='*80}")

# Merge proxy data with ranking changes
analysis_df = rank_df.merge(proxy_df, on='model')

# Analyze: Do models with higher proxy_frm/ref_frm ratios flip more?
print(f"\nHypothesis: Models with higher proxy FRM ratio flip DOWN in rankings on edge")
print(f"(i.e., they become relatively worse on edge devices)")

# Group by flip direction
analysis_df['flip_direction'] = analysis_df['rank_change'].apply(
    lambda x: 'DOWN (worse)' if x > 2 else ('UP (better)' if x < -2 else 'STABLE')
)

print(f"\n{'Category':<15} {'Count':<7} {'Mean Proxy Ratio':<18} {'Mean FRM Diff':<15}")
print("─"*80)
for category in ['UP (better)', 'STABLE', 'DOWN (worse)']:
    subset = analysis_df[analysis_df['flip_direction'] == category]
    if len(subset) > 0:
        mean_ratio = subset['frm_ratio'].mean()
        mean_diff = subset['frm_diff'].mean()
        print(f"{category:<15} {len(subset):>6} {mean_ratio:>17.2f}x {mean_diff:>14.3f}")

# Correlation between proxy ratio and rank change
corr_ratio = analysis_df[['rank_change', 'frm_ratio']].corr().iloc[0, 1]
corr_diff = analysis_df[['rank_change', 'frm_diff']].corr().iloc[0, 1]

print(f"\n📊 Correlations with rank_change:")
print(f"   Proxy FRM ratio: {corr_ratio:+.3f}")
print(f"   Proxy FRM diff: {corr_diff:+.3f}")

if corr_ratio > 0.3:
    print(f"\n✅ POSITIVE correlation: Higher proxy ratio → ranks WORSE on edge (flips DOWN)")
elif corr_ratio < -0.3:
    print(f"\n✅ NEGATIVE correlation: Higher proxy ratio → ranks BETTER on edge (flips UP)")
else:
    print(f"\n⚠️  WEAK correlation: Proxy ratio doesn't strongly predict flips")

print(f"\n{'='*80}")
print("STEP 4: Flip Prediction Test")
print(f"{'='*80}")

# Use proxy ratio to predict flips
# If proxy_ratio > threshold, predict model will flip down
threshold = proxy_df['frm_ratio'].median()

print(f"\nUsing threshold: {threshold:.2f}x (median proxy ratio)")

predictions = []
for _, row in analysis_df.iterrows():
    actual_flip = row['rank_change']
    
    # CORRECTED LOGIC: 
    # Low proxy ratio → edge device is MORE efficient than GPU baseline suggests → ranks UP
    # High proxy ratio → edge device is LESS efficient → ranks DOWN/stable
    if row['frm_ratio'] < threshold:
        predicted_flip = "UP"  # Model improves on edge
    else:
        predicted_flip = "DOWN/STABLE"  # Model worse or same
    
    actual_category = "UP" if actual_flip < -1 else "DOWN/STABLE"
    correct = (predicted_flip == actual_category)
    
    predictions.append({
        'model': row['model'],
        'proxy_ratio': row['frm_ratio'],
        'actual_change': actual_flip,
        'predicted': predicted_flip,
        'actual': actual_category,
        'correct': correct,
    })

pred_result_df = pd.DataFrame(predictions)

print(f"\n{'Model':<6} {'Proxy Ratio':<12} {'Actual':<8} {'Predicted':<12} {'✓/✗':<5}")
print("─"*80)
for _, row in pred_result_df.iterrows():
    marker = "✓" if row['correct'] else "✗"
    print(f"{row['model']:<6} {row['proxy_ratio']:>11.2f}x {row['actual_change']:>7d} "
          f"{row['predicted']:<12} {marker:<5}")

accuracy = pred_result_df['correct'].mean()
print(f"\n{'='*80}")
print(f"Prediction Accuracy: {accuracy:.1%}")
print(f"{'='*80}")

if accuracy > 0.7:
    print(f"\n🎉 SUCCESS! Proxy method predicts ranking flips with {accuracy:.0%} accuracy!")
elif accuracy > 0.5:
    print(f"\n✅ MODERATE: Proxy method shows promise ({accuracy:.0%} accuracy)")
else:
    print(f"\n⚠️  LIMITED: Proxy method needs refinement ({accuracy:.0%} accuracy)")

# Show most predictive models
print(f"\n{'='*80}")
print("Most Indicative Cases")
print(f"{'='*80}")

extreme_ratios = analysis_df.nlargest(3, 'frm_ratio')[['model', 'frm_ratio', 'rank_change']]
print(f"\nHighest proxy ratios (predict worse performance on edge):")
for _, row in extreme_ratios.iterrows():
    print(f"  {row['model']}: {row['frm_ratio']:.2f}x → rank change: {row['rank_change']:+d}")

low_ratios = analysis_df.nsmallest(3, 'frm_ratio')[['model', 'frm_ratio', 'rank_change']]
print(f"\nLowest proxy ratios (predict better/stable performance on edge):")
for _, row in low_ratios.iterrows():
    print(f"  {row['model']}: {row['frm_ratio']:.2f}x → rank change: {row['rank_change']:+d}")

print("\n" + "="*80)
print("✅ Analysis complete!")
print("="*80)
