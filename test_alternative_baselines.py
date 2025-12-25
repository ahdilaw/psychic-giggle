"""
Multi-Baseline Proxy Method
============================

Instead of normalizing against R50, use OTHER baselines (MV3, EN, etc.)
to see if they better capture hardware-model interactions for transfer prediction.

The idea:
1. Measure R50 on both GPU and Phone
2. Normalize BOTH against MV3 (the "proxy baseline")
3. See if R50's normalized scores predict better across devices
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

baseline = "R50"
df = pd.read_csv(f"{FRM_DIR}/frm_scores_{baseline}.csv")

for col in ['frm', 's_latency_ms', 'memory_mib']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['group_id'] = df.apply(create_group_id, axis=1)
df['device_tier'] = df['device'].apply(get_device_tier)

for model_id in df['model_id'].unique():
    if model_id in models_info:
        mask = df['model_id'] == model_id
        df.loc[mask, 'flops_g'] = models_info[model_id].get('flops_g', np.nan)

reference_device = 'h100_sxm|torch|none'
target_device = 'google_pixel_2|tflite|cpu'  # Best performing from previous test

print("="*80)
print("MULTI-BASELINE PROXY METHOD")
print("="*80)
print(f"\nReference device (GPU): {reference_device}")
print(f"Target device (Phone): {target_device}")

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

common_models = sorted(set(ref_agg['model_id']) & set(target_agg['model_id']))

# Test different baselines for normalization
proxy_baselines = ['MV3', 'EN', 'MN', 'R18']  # Alternative baselines

print(f"\n{'='*80}")
print("STEP 1: Current Method (Normalized Against R50)")
print(f"{'='*80}")

# Current rankings with R50 baseline
ref_agg['rank_r50'] = ref_agg['frm'].rank(method='min', ascending=False)
target_agg['rank_r50'] = target_agg['frm'].rank(method='min', ascending=False)

print(f"\n{'Model':<6} {'GPU Rank':<9} {'Phone Rank':<11} {'Change':<8}")
print("─"*80)

r50_flips = 0
for model in common_models:
    ref_rank = ref_agg[ref_agg['model_id'] == model]['rank_r50'].values[0]
    target_rank = target_agg[target_agg['model_id'] == model]['rank_r50'].values[0]
    change = int(target_rank - ref_rank)
    
    if abs(change) >= 2:
        r50_flips += 1
        marker = "⚠️"
    else:
        marker = ""
    
    print(f"{model:<6} {ref_rank:>8.0f} {target_rank:>10.0f} {change:>7d} {marker}")

print(f"\n🎯 Total flips with R50 baseline: {r50_flips}")

# Now test alternative baselines
print(f"\n{'='*80}")
print("STEP 2: Alternative Baselines")
print(f"{'='*80}")

results_by_baseline = []

for proxy_baseline in proxy_baselines:
    if proxy_baseline not in ref_agg['model_id'].values or proxy_baseline not in target_agg['model_id'].values:
        continue
    
    print(f"\n{'─'*80}")
    print(f"Normalizing against: {proxy_baseline}")
    print(f"{'─'*80}")
    
    # Get proxy baseline measurements
    ref_proxy_lat = ref_agg[ref_agg['model_id'] == proxy_baseline]['s_latency_ms'].values[0]
    ref_proxy_mem = ref_agg[ref_agg['model_id'] == proxy_baseline]['memory_mib'].values[0]
    ref_proxy_flops = ref_agg[ref_agg['model_id'] == proxy_baseline]['flops_g'].values[0]
    
    target_proxy_lat = target_agg[target_agg['model_id'] == proxy_baseline]['s_latency_ms'].values[0]
    target_proxy_mem = target_agg[target_agg['model_id'] == proxy_baseline]['memory_mib'].values[0]
    target_proxy_flops = target_agg[target_agg['model_id'] == proxy_baseline]['flops_g'].values[0]
    
    print(f"\nProxy baseline {proxy_baseline} measurements:")
    print(f"  GPU: {ref_proxy_lat:.2f}ms lat, {ref_proxy_mem:.0f}MB mem")
    print(f"  Phone: {target_proxy_lat:.2f}ms lat, {target_proxy_mem:.0f}MB mem")
    
    # Re-compute FRM for all models using this baseline
    ref_frm_new = {}
    target_frm_new = {}
    
    for model in common_models:
        # GPU with new baseline
        ref_lat = ref_agg[ref_agg['model_id'] == model]['s_latency_ms'].values[0]
        ref_mem = ref_agg[ref_agg['model_id'] == model]['memory_mib'].values[0]
        ref_flops = ref_agg[ref_agg['model_id'] == model]['flops_g'].values[0]
        
        ref_lat_ratio = ref_lat / ref_proxy_lat
        ref_mem_ratio = ref_mem / ref_proxy_mem
        ref_flops_ratio = ref_flops / ref_proxy_flops
        
        ref_frm_new[model] = (ref_lat_ratio * ref_mem_ratio * ref_flops_ratio) ** (1/3)
        
        # Phone with new baseline
        target_lat = target_agg[target_agg['model_id'] == model]['s_latency_ms'].values[0]
        target_mem = target_agg[target_agg['model_id'] == model]['memory_mib'].values[0]
        target_flops = target_agg[target_agg['model_id'] == model]['flops_g'].values[0]
        
        target_lat_ratio = target_lat / target_proxy_lat
        target_mem_ratio = target_mem / target_proxy_mem
        target_flops_ratio = target_flops / target_proxy_flops
        
        target_frm_new[model] = (target_lat_ratio * target_mem_ratio * target_flops_ratio) ** (1/3)
    
    # Get new rankings
    ref_ranks = pd.Series(ref_frm_new).rank(method='min', ascending=False)
    target_ranks = pd.Series(target_frm_new).rank(method='min', ascending=False)
    
    # Count flips
    flip_count = 0
    rank_changes = []
    
    print(f"\n{'Model':<6} {'GPU FRM':<9} {'GPU Rank':<9} {'Phone FRM':<10} {'Phone Rank':<11} {'Change':<8}")
    print("─"*80)
    
    for model in common_models:
        ref_rank = ref_ranks[model]
        target_rank = target_ranks[model]
        change = int(target_rank - ref_rank)
        rank_changes.append(change)
        
        if abs(change) >= 2:
            flip_count += 1
            marker = "⚠️"
        else:
            marker = ""
        
        print(f"{model:<6} {ref_frm_new[model]:>8.3f} {ref_rank:>8.0f} "
              f"{target_frm_new[model]:>9.3f} {target_rank:>10.0f} {change:>7d} {marker}")
    
    # Compute ranking correlation
    rho = spearmanr(list(ref_frm_new.values()), list(target_frm_new.values()))[0]
    
    print(f"\n🎯 Flips with {proxy_baseline} baseline: {flip_count}")
    print(f"📊 Ranking correlation: ρ={rho:.3f}")
    
    results_by_baseline.append({
        'baseline': proxy_baseline,
        'n_flips': flip_count,
        'rho': rho,
        'mean_rank_change': np.mean(np.abs(rank_changes)),
    })

# Compare baselines
print(f"\n{'='*80}")
print("STEP 3: Comparison")
print(f"{'='*80}")

results_df = pd.DataFrame(results_by_baseline)

print(f"\n{'Baseline':<10} {'Flips':<7} {'ρ':<8} {'Mean |Δrank|':<15}")
print("─"*80)

# Compute R50 correlation on common models only
ref_frm_common = [ref_agg[ref_agg['model_id'] == m]['frm'].values[0] for m in common_models]
target_frm_common = [target_agg[target_agg['model_id'] == m]['frm'].values[0] for m in common_models]
r50_rho = spearmanr(ref_frm_common, target_frm_common)[0]

print(f"{'R50 (orig)':<10} {r50_flips:>6} {r50_rho:>7.3f} {'N/A':<15}")
for _, row in results_df.iterrows():
    print(f"{row['baseline']:<10} {row['n_flips']:>6} {row['rho']:>7.3f} {row['mean_rank_change']:>14.2f}")

# Find best baseline
best_baseline = results_df.loc[results_df['n_flips'].idxmin()]

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")

print(f"\n✅ Best baseline for stability: {best_baseline['baseline']}")
print(f"   - Reduces flips from {r50_flips} to {int(best_baseline['n_flips'])}")
print(f"   - Ranking correlation: ρ={best_baseline['rho']:.3f}")

if best_baseline['n_flips'] < r50_flips:
    improvement = (r50_flips - best_baseline['n_flips']) / r50_flips
    print(f"   - {improvement:.0%} reduction in ranking flips!")
    print(f"\n🎉 Using {best_baseline['baseline']} as baseline IMPROVES transfer prediction!")
else:
    print(f"\n⚠️  Alternative baselines don't improve over R50")

print("\n" + "="*80)
