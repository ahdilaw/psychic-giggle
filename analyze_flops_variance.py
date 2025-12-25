"""
The Real Reason: FLOPs Ratio Variance
======================================

Hypothesis: R50 causes flips because models have very DIFFERENT FLOPs
compared to R50, creating high variance in the FLOPs ratio component.

MV3 causes no flips because models have more SIMILAR FLOPs to MV3.
"""

import pandas as pd
import numpy as np
import json

FRM_DIR = "agg/frm"
MODELS_JSON = "agg/models.json"

with open(MODELS_JSON) as f:
    models_info = json.load(f)

def create_group_id(row):
    acc = row['accelerator'] if pd.notna(row['accelerator']) else 'none'
    return f"{row['device']}|{row['framework']}|{acc}"

baseline = "R50"
df = pd.read_csv(f"{FRM_DIR}/frm_scores_{baseline}.csv")

for col in ['frm', 's_latency_ms', 'memory_mib']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['group_id'] = df.apply(create_group_id, axis=1)

for model_id in df['model_id'].unique():
    if model_id in models_info:
        mask = df['model_id'] == model_id
        df.loc[mask, 'flops_g'] = models_info[model_id].get('flops_g', np.nan)

reference_device = 'h100_sxm|torch|none'
target_device = 'google_pixel_2|tflite|cpu'

ref_data = df[df['group_id'] == reference_device].copy()
target_data = df[df['group_id'] == target_device].copy()

ref_agg = ref_data.groupby('model_id').agg({
    's_latency_ms': 'mean',
    'memory_mib': 'mean',
    'flops_g': 'mean',
}).reset_index()

target_agg = target_data.groupby('model_id').agg({
    's_latency_ms': 'mean',
    'memory_mib': 'mean',
    'flops_g': 'mean',
}).reset_index()

common_models = sorted(set(ref_agg['model_id']) & set(target_agg['model_id']))

print("="*80)
print("THE REAL REASON: FLOPs Ratio Variance")
print("="*80)

baselines = ['R50', 'MV3', 'EN', 'MN', 'R18']

print(f"\n{'='*80}")
print("FLOPS DISTRIBUTION")
print(f"{'='*80}")

print(f"\n{'Model':<6} {'FLOPs (G)':<12} {'vs R50':<12} {'vs MV3':<12} {'vs EN':<12}")
print("─"*80)

for model in common_models:
    flops = models_info.get(model, {}).get('flops_g', np.nan)
    
    r50_flops = models_info.get('R50', {}).get('flops_g', 4.089)
    mv3_flops = models_info.get('MV3', {}).get('flops_g', 0.056)
    en_flops = models_info.get('EN', {}).get('flops_g', 0.389)
    
    ratio_r50 = flops / r50_flops if r50_flops > 0 else 0
    ratio_mv3 = flops / mv3_flops if mv3_flops > 0 else 0
    ratio_en = flops / en_flops if en_flops > 0 else 0
    
    print(f"{model:<6} {flops:>11.3f} {ratio_r50:>11.2f}x {ratio_mv3:>11.2f}x {ratio_en:>11.2f}x")

print(f"\n{'='*80}")
print("FLOPS RATIO VARIANCE")
print(f"{'='*80}")

for baseline_model in baselines:
    if baseline_model not in models_info:
        continue
    
    base_flops = models_info[baseline_model]['flops_g']
    
    flops_ratios = []
    for model in common_models:
        model_flops = models_info.get(model, {}).get('flops_g', np.nan)
        if not np.isnan(model_flops) and base_flops > 0:
            flops_ratios.append(model_flops / base_flops)
    
    mean_ratio = np.mean(flops_ratios)
    std_ratio = np.std(flops_ratios)
    cv = std_ratio / mean_ratio if mean_ratio > 0 else 0
    min_ratio = np.min(flops_ratios)
    max_ratio = np.max(flops_ratios)
    range_ratio = max_ratio - min_ratio
    
    print(f"\n{baseline_model}:")
    print(f"  Base FLOPs: {base_flops:.3f}G")
    print(f"  Ratio range: [{min_ratio:.2f}x, {max_ratio:.2f}x] (span: {range_ratio:.2f}x)")
    print(f"  Mean ratio: {mean_ratio:.2f}x")
    print(f"  Std ratio: {std_ratio:.2f}x")
    print(f"  CV: {cv:.3f}")

print(f"\n{'='*80}")
print("THE INSIGHT")
print(f"{'='*80}")

print("""
Look at the FLOPs ratio RANGE (min to max):

R50 (4.089G):  [0.01x, 2.03x] - span: 2.02x
MV3 (0.056G):  [1.00x, 148x] - span: 147x
EN (0.389G):   [0.14x, 21.3x] - span: 21.2x

Wait... MV3 has HUGE range (147x) but ZERO flips!
R50 has SMALL range (2x) but 6 flips!

This contradicts my hypothesis!

Let me think differently...
""")

print(f"\n{'='*80}")
print("ALTERNATIVE EXPLANATION: Component Balance")
print(f"{'='*80}")

print("\nLet's check the GEOMETRIC MEAN balance across components:")

for baseline_model in ['R50', 'MV3']:
    print(f"\n{'-'*80}")
    print(f"Baseline: {baseline_model}")
    print(f"{'-'*80}")
    
    base_ref_lat = ref_agg[ref_agg['model_id'] == baseline_model]['s_latency_ms'].values[0]
    base_ref_mem = ref_agg[ref_agg['model_id'] == baseline_model]['memory_mib'].values[0]
    base_ref_flops = models_info[baseline_model]['flops_g']
    
    base_target_lat = target_agg[target_agg['model_id'] == baseline_model]['s_latency_ms'].values[0]
    base_target_mem = target_agg[target_agg['model_id'] == baseline_model]['memory_mib'].values[0]
    
    print(f"\n{'Model':<6} {'GPU Components':<30} {'Phone Components':<30} {'Balance':<10}")
    print(f"{'':6} {'Lat':<8} {'Mem':<8} {'FLOPs':<8} {'Lat':<8} {'Mem':<8} {'FLOPs':<8} {'GPU':<10} {'Phone':<10}")
    print("─"*80)
    
    gpu_balances = []
    phone_balances = []
    
    for model in common_models:
        # GPU ratios
        ref_lat = ref_agg[ref_agg['model_id'] == model]['s_latency_ms'].values[0]
        ref_mem = ref_agg[ref_agg['model_id'] == model]['memory_mib'].values[0]
        ref_flops = models_info.get(model, {}).get('flops_g', np.nan)
        
        lat_r = ref_lat / base_ref_lat
        mem_r = ref_mem / base_ref_mem
        flops_r = ref_flops / base_ref_flops
        
        # Phone ratios
        target_lat = target_agg[target_agg['model_id'] == model]['s_latency_ms'].values[0]
        target_mem = target_agg[target_agg['model_id'] == model]['memory_mib'].values[0]
        target_flops = ref_flops  # FLOPs don't change
        
        lat_t = target_lat / base_target_lat
        mem_t = target_mem / base_target_mem
        flops_t = target_flops / base_ref_flops  # Same as GPU
        
        # Balance: std dev of log-ratios (how spread out are the components?)
        gpu_balance = np.std([np.log(lat_r), np.log(mem_r), np.log(flops_r)])
        phone_balance = np.std([np.log(lat_t), np.log(mem_t), np.log(flops_t)])
        
        gpu_balances.append(gpu_balance)
        phone_balances.append(phone_balance)
        
        print(f"{model:<6} {lat_r:>7.2f} {mem_r:>7.2f} {flops_r:>7.2f} "
              f"{lat_t:>7.2f} {mem_t:>7.2f} {flops_t:>7.2f} "
              f"{gpu_balance:>9.3f} {phone_balance:>9.3f}")
    
    print(f"\nMean component balance (std of log-ratios):")
    print(f"  GPU: {np.mean(gpu_balances):.3f}")
    print(f"  Phone: {np.mean(phone_balances):.3f}")
    print(f"  Difference: {abs(np.mean(gpu_balances) - np.mean(phone_balances)):.3f}")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")

print("""
I need to check: Do the components (lat, mem, flops) maintain the same
PROPORTIONAL RELATIONSHIP when transferring from GPU to Phone?

If R50 normalization causes components to scale disproportionately,
that could cause flips. If MV3 keeps proportions stable, no flips.
""")

print("="*80)
