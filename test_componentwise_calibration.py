"""
Component-Wise Proxy Calibration
=================================
Instead of a single hardware factor, calibrate latency and memory separately.
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
print("COMPONENT-WISE PROXY CALIBRATION")
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
edge_devices = df[df['device_tier'] == 'edge']['group_id'].unique()

print(f"\nReference device: {reference_device}")
print(f"Testing on: {edge_devices[0]}")

target_device = edge_devices[0]

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

# Get baselines on both devices
calibration_models = ['R50', 'MV3', 'EN']

print(f"\n{'='*80}")
print("COMPONENT-WISE CALIBRATION FACTORS")
print(f"{'='*80}")

lat_factors = []
mem_factors = []

for calib in calibration_models:
    if calib not in ref_agg['model_id'].values or calib not in target_agg['model_id'].values:
        continue
    
    ref_lat = ref_agg[ref_agg['model_id'] == calib]['s_latency_ms'].values[0]
    ref_mem = ref_agg[ref_agg['model_id'] == calib]['memory_mib'].values[0]
    
    target_lat = target_agg[target_agg['model_id'] == calib]['s_latency_ms'].values[0]
    target_mem = target_agg[target_agg['model_id'] == calib]['memory_mib'].values[0]
    
    lat_factor = target_lat / ref_lat
    mem_factor = target_mem / ref_mem
    
    lat_factors.append(lat_factor)
    mem_factors.append(mem_factor)
    
    print(f"\n{calib}:")
    print(f"  Latency: {ref_lat:.2f}ms → {target_lat:.2f}ms (factor: {lat_factor:.2f}x)")
    print(f"  Memory: {ref_mem:.2f}MB → {target_mem:.2f}MB (factor: {mem_factor:.2f}x)")

avg_lat_factor = np.mean(lat_factors)
avg_mem_factor = np.mean(mem_factors)

std_lat_factor = np.std(lat_factors)
std_mem_factor = np.std(mem_factors)

print(f"\n{'─'*80}")
print(f"Average Calibration Factors:")
print(f"  Latency: {avg_lat_factor:.2f}x (std: {std_lat_factor:.2f}, CV: {std_lat_factor/avg_lat_factor:.1%})")
print(f"  Memory: {avg_mem_factor:.2f}x (std: {std_mem_factor:.2f}, CV: {std_mem_factor/avg_mem_factor:.1%})")

# Get baseline for normalization
ref_baseline_lat = ref_agg[ref_agg['model_id'] == 'R50']['s_latency_ms'].values[0]
ref_baseline_mem = ref_agg[ref_agg['model_id'] == 'R50']['memory_mib'].values[0]
ref_baseline_flops = models_info.get('R50', {}).get('flops_g', 4.089)

target_baseline_lat = target_agg[target_agg['model_id'] == 'R50']['s_latency_ms'].values[0]
target_baseline_mem = target_agg[target_agg['model_id'] == 'R50']['memory_mib'].values[0]

print(f"\n{'='*80}")
print("PREDICTION METHODS COMPARISON")
print(f"{'='*80}")

common_models = sorted(set(ref_agg['model_id']) & set(target_agg['model_id']))

predictions = []

for model in common_models:
    # Get reference measurements
    ref_lat = ref_agg[ref_agg['model_id'] == model]['s_latency_ms'].values[0]
    ref_mem = ref_agg[ref_agg['model_id'] == model]['memory_mib'].values[0]
    ref_flops = ref_agg[ref_agg['model_id'] == model]['flops_g'].values[0]
    ref_frm = ref_agg[ref_agg['model_id'] == model]['frm'].values[0]
    
    # Get actual target measurements
    target_lat_actual = target_agg[target_agg['model_id'] == model]['s_latency_ms'].values[0]
    target_mem_actual = target_agg[target_agg['model_id'] == model]['memory_mib'].values[0]
    target_frm_actual = target_agg[target_agg['model_id'] == model]['frm'].values[0]
    
    # Method 1: Direct FRM transfer (current approach)
    pred_frm_direct = ref_frm
    
    # Method 2: Component-wise calibration
    # Predict target latency and memory using calibration factors
    pred_lat = ref_lat * avg_lat_factor
    pred_mem = ref_mem * avg_mem_factor
    
    # Compute predicted FRM using predicted values
    pred_lat_ratio = pred_lat / target_baseline_lat
    pred_mem_ratio = pred_mem / target_baseline_mem
    pred_flops_ratio = ref_flops / ref_baseline_flops
    
    pred_frm_componentwise = (pred_lat_ratio * pred_mem_ratio * pred_flops_ratio) ** (1/3)
    
    # Method 3: Using actual target latency/memory but ref FLOPs ratio
    # (This tests if FLOPs ratio transfers perfectly)
    actual_lat_ratio = target_lat_actual / target_baseline_lat
    actual_mem_ratio = target_mem_actual / target_baseline_mem
    
    pred_frm_flops_transfer = (actual_lat_ratio * actual_mem_ratio * pred_flops_ratio) ** (1/3)
    
    predictions.append({
        'model': model,
        'ref_frm': ref_frm,
        'pred_direct': pred_frm_direct,
        'pred_componentwise': pred_frm_componentwise,
        'pred_flops_transfer': pred_frm_flops_transfer,
        'actual': target_frm_actual,
    })

pred_df = pd.DataFrame(predictions)

# Calculate errors
pred_df['err_direct'] = abs(pred_df['pred_direct'] - pred_df['actual']) / pred_df['actual']
pred_df['err_componentwise'] = abs(pred_df['pred_componentwise'] - pred_df['actual']) / pred_df['actual']
pred_df['err_flops_transfer'] = abs(pred_df['pred_flops_transfer'] - pred_df['actual']) / pred_df['actual']

print(f"\n{'Model':<6} {'Ref':<7} {'Direct':<7} {'Compnt':<7} {'FLOPs':<7} {'Actual':<7} {'E(D)':<6} {'E(C)':<6} {'E(F)':<6}")
print("─"*80)
for _, row in pred_df.iterrows():
    print(f"{row['model']:<6} {row['ref_frm']:>6.3f} {row['pred_direct']:>6.3f} "
          f"{row['pred_componentwise']:>6.3f} {row['pred_flops_transfer']:>6.3f} "
          f"{row['actual']:>6.3f} {row['err_direct']:>5.1%} {row['err_componentwise']:>5.1%} "
          f"{row['err_flops_transfer']:>5.1%}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

print(f"\nMean Absolute Percentage Error:")
print(f"  Method 1 (Direct FRM transfer): {pred_df['err_direct'].mean():.1%}")
print(f"  Method 2 (Component-wise calib): {pred_df['err_componentwise'].mean():.1%}")
print(f"  Method 3 (FLOPs ratio transfer): {pred_df['err_flops_transfer'].mean():.1%}")

print(f"\nRanking Correlation (Spearman ρ):")
rho_direct = spearmanr(pred_df['pred_direct'], pred_df['actual'])[0]
rho_componentwise = spearmanr(pred_df['pred_componentwise'], pred_df['actual'])[0]
rho_flops_transfer = spearmanr(pred_df['pred_flops_transfer'], pred_df['actual'])[0]

print(f"  Method 1 (Direct): {rho_direct:.3f}")
print(f"  Method 2 (Component-wise): {rho_componentwise:.3f}")
print(f"  Method 3 (FLOPs transfer): {rho_flops_transfer:.3f}")

print(f"\nBest method per model:")
best_direct = sum((pred_df['err_direct'] <= pred_df['err_componentwise']) & 
                  (pred_df['err_direct'] <= pred_df['err_flops_transfer']))
best_componentwise = sum((pred_df['err_componentwise'] < pred_df['err_direct']) & 
                         (pred_df['err_componentwise'] < pred_df['err_flops_transfer']))
best_flops = sum((pred_df['err_flops_transfer'] < pred_df['err_direct']) & 
                 (pred_df['err_flops_transfer'] < pred_df['err_componentwise']))

print(f"  Direct FRM: {best_direct} models")
print(f"  Component-wise: {best_componentwise} models")
print(f"  FLOPs transfer: {best_flops} models")

print("\n" + "="*80)
print("✅ Analysis complete!")
print("="*80)
