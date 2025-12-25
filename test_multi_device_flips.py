"""
Multi-Device Flip Prediction Test
==================================
Test proxy calibration method across multiple GPU→Edge pairs
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

# Test multiple GPU devices
gpu_devices = ['h100_sxm|torch|none', 'a100_sxm|torch|none', 'rtx_3090|torch|none']
edge_devices = df[df['device_tier'] == 'edge']['group_id'].unique()[:5]

print("="*80)
print("MULTI-DEVICE FLIP PREDICTION TEST")
print("="*80)

results_all = []

for ref_device in gpu_devices:
    for target_device in edge_devices:
        ref_data = df[df['group_id'] == ref_device].copy()
        target_data = df[df['group_id'] == target_device].copy()
        
        if len(ref_data) < 8 or len(target_data) < 8:
            continue
        
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
        
        common_models = set(ref_agg['model_id']) & set(target_agg['model_id'])
        if len(common_models) < 8:
            continue
        
        # Get baselines
        if 'R50' not in ref_agg['model_id'].values or 'R50' not in target_agg['model_id'].values:
            continue
        
        ref_baseline_lat = ref_agg[ref_agg['model_id'] == 'R50']['s_latency_ms'].values[0]
        ref_baseline_mem = ref_agg[ref_agg['model_id'] == 'R50']['memory_mib'].values[0]
        ref_baseline_flops = models_info['R50']['flops_g']
        
        target_baseline_lat = target_agg[target_agg['model_id'] == 'R50']['s_latency_ms'].values[0]
        target_baseline_mem = target_agg[target_agg['model_id'] == 'R50']['memory_mib'].values[0]
        
        # Rankings
        ref_agg['rank'] = ref_agg['frm'].rank(method='min', ascending=False)
        target_agg['rank'] = target_agg['frm'].rank(method='min', ascending=False)
        
        # Compute flips and proxy ratios
        flip_count = 0
        correct_predictions = 0
        total_predictions = 0
        
        proxy_ratios = []
        
        for model in common_models:
            ref_frm = ref_agg[ref_agg['model_id'] == model]['frm'].values[0]
            ref_rank = ref_agg[ref_agg['model_id'] == model]['rank'].values[0]
            
            target_frm = target_agg[target_agg['model_id'] == model]['frm'].values[0]
            target_rank = target_agg[target_agg['model_id'] == model]['rank'].values[0]
            
            rank_change = int(target_rank - ref_rank)
            
            if abs(rank_change) >= 2:
                flip_count += 1
            
            # Compute proxy ratio
            target_lat = target_agg[target_agg['model_id'] == model]['s_latency_ms'].values[0]
            target_mem = target_agg[target_agg['model_id'] == model]['memory_mib'].values[0]
            target_flops = target_agg[target_agg['model_id'] == model]['flops_g'].values[0]
            
            proxy_lat_ratio = target_lat / ref_baseline_lat
            proxy_mem_ratio = target_mem / ref_baseline_mem
            proxy_flops_ratio = target_flops / ref_baseline_flops
            
            proxy_frm = (proxy_lat_ratio * proxy_mem_ratio * proxy_flops_ratio) ** (1/3)
            proxy_ratio = proxy_frm / ref_frm if ref_frm > 0 else 0
            
            proxy_ratios.append(proxy_ratio)
        
        # Use median as threshold
        threshold = np.median(proxy_ratios)
        
        # Predict flips
        for model in common_models:
            ref_frm = ref_agg[ref_agg['model_id'] == model]['frm'].values[0]
            ref_rank = ref_agg[ref_agg['model_id'] == model]['rank'].values[0]
            target_rank = target_agg[target_agg['model_id'] == model]['rank'].values[0]
            rank_change = int(target_rank - ref_rank)
            
            # Compute proxy
            target_lat = target_agg[target_agg['model_id'] == model]['s_latency_ms'].values[0]
            target_mem = target_agg[target_agg['model_id'] == model]['memory_mib'].values[0]
            target_flops = target_agg[target_agg['model_id'] == model]['flops_g'].values[0]
            
            proxy_lat_ratio = target_lat / ref_baseline_lat
            proxy_mem_ratio = target_mem / ref_baseline_mem
            proxy_flops_ratio = target_flops / ref_baseline_flops
            
            proxy_frm = (proxy_lat_ratio * proxy_mem_ratio * proxy_flops_ratio) ** (1/3)
            proxy_ratio = proxy_frm / ref_frm if ref_frm > 0 else 0
            
            # Predict
            if proxy_ratio < threshold:
                predicted = "UP"
            else:
                predicted = "STABLE/DOWN"
            
            actual = "UP" if rank_change < -1 else "STABLE/DOWN"
            
            if predicted == actual:
                correct_predictions += 1
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        results_all.append({
            'ref_device': ref_device,
            'target_device': target_device,
            'n_models': len(common_models),
            'n_flips': flip_count,
            'accuracy': accuracy,
            'threshold': threshold,
        })

# Analyze results
results_df = pd.DataFrame(results_all)

print(f"\nTested {len(results_df)} device pairs")
print(f"Mean prediction accuracy: {results_df['accuracy'].mean():.1%}")
print(f"Median accuracy: {results_df['accuracy'].median():.1%}")
print(f"Best accuracy: {results_df['accuracy'].max():.1%}")
print(f"Worst accuracy: {results_df['accuracy'].min():.1%}")

print(f"\n{'='*80}")
print("Best Performing Pairs")
print(f"{'='*80}")
best = results_df.nlargest(5, 'accuracy')
print(f"\n{'Ref Device':<25} {'Target Device':<35} {'Acc':<7} {'Flips':<7}")
print("─"*80)
for _, row in best.iterrows():
    ref_short = row['ref_device'].split('|')[0]
    target_short = row['target_device'].split('|')[0][:30]
    print(f"{ref_short:<25} {target_short:<35} {row['accuracy']:>6.1%} {row['n_flips']:>6}")

print(f"\n{'='*80}")
print("Worst Performing Pairs")
print(f"{'='*80}")
worst = results_df.nsmallest(5, 'accuracy')
print(f"\n{'Ref Device':<25} {'Target Device':<35} {'Acc':<7} {'Flips':<7}")
print("─"*80)
for _, row in worst.iterrows():
    ref_short = row['ref_device'].split('|')[0]
    target_short = row['target_device'].split('|')[0][:30]
    print(f"{ref_short:<25} {target_short:<35} {row['accuracy']:>6.1%} {row['n_flips']:>6}")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")

if results_df['accuracy'].mean() > 0.65:
    print(f"\n✅ Proxy method works well! ({results_df['accuracy'].mean():.0%} mean accuracy)")
    print("Baseline measurements CAN predict ranking flips across devices")
elif results_df['accuracy'].mean() > 0.55:
    print(f"\n⚠️  Proxy method shows moderate promise ({results_df['accuracy'].mean():.0%})")
    print("Better than random, but needs refinement")
else:
    print(f"\n❌ Proxy method doesn't work ({results_df['accuracy'].mean():.0%})")
    print("Baseline measurements don't predict flips reliably")

print("\n" + "="*80)
