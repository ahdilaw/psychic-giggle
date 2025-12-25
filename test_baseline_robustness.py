"""
Comprehensive Baseline Selection Validation
============================================

Test whether baseline selection (R50 vs MV3) affects ranking stability
across 10 diverse device pairs spanning GPU→Edge, GPU→GPU, GPU→CPU, Edge→Edge.

This addresses the main CVPR weakness: single device pair validation.
"""

import pandas as pd
import numpy as np
import json
from scipy.stats import spearmanr
from itertools import combinations

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

def compute_frm_with_baseline(df, baseline_model, device_group):
    """Recompute FRM scores using specified baseline on specified device"""
    device_data = df[df['group_id'] == device_group].copy()
    baseline_data = device_data[device_data['model_id'] == baseline_model]
    
    if len(baseline_data) == 0:
        return None
    
    baseline_lat = baseline_data['s_latency_ms'].mean()
    baseline_mem = baseline_data['memory_mib'].mean()
    baseline_flops = baseline_data[baseline_data['flops_g'].notna()]['flops_g'].mean()
    
    result = []
    for model in device_data['model_id'].unique():
        model_data = device_data[device_data['model_id'] == model]
        lat = model_data['s_latency_ms'].mean()
        mem = model_data['memory_mib'].mean()
        flops = model_data[model_data['flops_g'].notna()]['flops_g'].mean()
        
        if pd.notna(lat) and pd.notna(mem) and pd.notna(flops):
            lat_ratio = lat / baseline_lat
            mem_ratio = mem / baseline_mem
            flops_ratio = flops / baseline_flops
            frm = (lat_ratio * mem_ratio * flops_ratio) ** (1/3)
            
            result.append({
                'model_id': model,
                'frm': frm,
                'lat_ratio': lat_ratio,
                'mem_ratio': mem_ratio,
                'flops_ratio': flops_ratio
            })
    
    return pd.DataFrame(result)

def count_ranking_flips(ranks1, ranks2, threshold=2):
    """Count how many models have rank changes >= threshold"""
    flips = 0
    for model in ranks1.index:
        if model in ranks2.index:
            change = abs(ranks1[model] - ranks2[model])
            if change >= threshold:
                flips += 1
    return flips

# Load data
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

# Define strategic device pairs
device_pairs = [
    # GPU → Edge (5 pairs)
    ('h100_sxm|torch|none', 'oneplus_nord_ce_2_lite_5g|tflite|cpu', 'GPU→Edge'),
    ('a100_sxm|torch|none', 'samsung_galaxy_s23_ultra_(global)|tflite|cpu', 'GPU→Edge'),
    ('rtx_4090|torch|none', 'samsung_galaxy_s22|tflite|cpu', 'GPU→Edge'),
    ('rtx_3090|torch|none', 'samsung_galaxy_s21_ultra_5g|tflite|cpu', 'GPU→Edge'),
    ('h200_sxm|torch|none', 'asus_zenfone_max_pro_m1|tflite|cpu', 'GPU→Edge'),
    
    # GPU → GPU (2 pairs)
    ('h100_sxm|torch|none', 'a100_sxm|torch|none', 'GPU→GPU'),
    ('rtx_4090|torch|none', 'rtx_3090|torch|none', 'GPU→GPU'),
    
    # GPU → CPU (2 pairs)
    ('h100_sxm|torch|none', 'cpu_gp_v5|onnx|none', 'GPU→CPU'),
    ('rtx_4090|torch|none', 'cpu_mem_v5|onnx|none', 'GPU→CPU'),
    
    # Edge → Edge (1 pair)
    ('samsung_galaxy_s23_ultra_(global)|tflite|cpu', 'oneplus_nord_ce_2_lite_5g|tflite|cpu', 'Edge→Edge'),
]

# Test baselines
test_baselines = ['R50', 'MV3', 'EN', 'MN', 'R18']

print("="*100)
print("COMPREHENSIVE BASELINE SELECTION VALIDATION")
print("="*100)
print(f"\nTesting {len(device_pairs)} device pairs across {len(test_baselines)} baselines")
print(f"Purpose: Validate whether small baselines (MV3/EN/MN/R18) consistently outperform large baseline (R50)")
print()

results = []

for ref_device, target_device, pair_type in device_pairs:
    # Check if both devices exist
    if ref_device not in df['group_id'].values or target_device not in df['group_id'].values:
        print(f"⚠️  Skipping {ref_device} → {target_device}: devices not found")
        continue
    
    print("─"*100)
    print(f"Device Pair: {ref_device} → {target_device} ({pair_type})")
    print("─"*100)
    
    for baseline_model in test_baselines:
        # Recompute FRM with this baseline on both devices
        ref_frm = compute_frm_with_baseline(df, baseline_model, ref_device)
        target_frm = compute_frm_with_baseline(df, baseline_model, target_device)
        
        if ref_frm is None or target_frm is None or len(ref_frm) < 5 or len(target_frm) < 5:
            continue
        
        # Find common models
        common_models = sorted(set(ref_frm['model_id']) & set(target_frm['model_id']))
        if len(common_models) < 5:
            continue
        
        # Compute ranks
        ref_frm = ref_frm.set_index('model_id')
        target_frm = target_frm.set_index('model_id')
        
        ref_ranks = ref_frm.loc[common_models, 'frm'].rank(method='min', ascending=False)
        target_ranks = target_frm.loc[common_models, 'frm'].rank(method='min', ascending=False)
        
        # Compute metrics
        rho, p_value = spearmanr(ref_ranks, target_ranks)
        flips = count_ranking_flips(ref_ranks, target_ranks, threshold=2)
        
        results.append({
            'ref_device': ref_device,
            'target_device': target_device,
            'pair_type': pair_type,
            'baseline': baseline_model,
            'spearman_rho': rho,
            'p_value': p_value,
            'num_flips': flips,
            'num_models': len(common_models)
        })
        
        print(f"  {baseline_model}: ρ={rho:.3f}, flips={flips}/{len(common_models)}")
    
    print()

# Convert to DataFrame and analyze
results_df = pd.DataFrame(results)

if len(results_df) == 0:
    print("❌ No valid device pairs found. Check device names.")
else:
    print("="*100)
    print("SUMMARY: Baseline Performance Across All Device Pairs")
    print("="*100)
    
    summary = results_df.groupby('baseline').agg({
        'spearman_rho': ['mean', 'std', 'min', 'max'],
        'num_flips': ['mean', 'sum', 'min', 'max'],
        'num_models': 'count'
    }).round(3)
    
    print(summary)
    print()
    
    # Statistical test: R50 vs others
    print("="*100)
    print("STATISTICAL COMPARISON: R50 vs Small Baselines")
    print("="*100)
    
    r50_results = results_df[results_df['baseline'] == 'R50']
    small_baselines = ['MV3', 'EN', 'MN', 'R18']
    
    for small_baseline in small_baselines:
        small_results = results_df[results_df['baseline'] == small_baseline]
        
        # Match device pairs
        common_pairs = set(zip(r50_results['ref_device'], r50_results['target_device'])) & \
                       set(zip(small_results['ref_device'], small_results['target_device']))
        
        if len(common_pairs) == 0:
            continue
        
        r50_matched = []
        small_matched = []
        
        for ref, tgt in common_pairs:
            r50_row = r50_results[(r50_results['ref_device'] == ref) & 
                                   (r50_results['target_device'] == tgt)]
            small_row = small_results[(small_results['ref_device'] == ref) & 
                                      (small_results['target_device'] == tgt)]
            
            if len(r50_row) > 0 and len(small_row) > 0:
                r50_matched.append(r50_row['num_flips'].values[0])
                small_matched.append(small_row['num_flips'].values[0])
        
        if len(r50_matched) > 0:
            from scipy.stats import wilcoxon
            
            print(f"\n{small_baseline} vs R50 (n={len(r50_matched)} pairs):")
            print(f"  R50 flips: {np.mean(r50_matched):.1f} ± {np.std(r50_matched):.1f}")
            print(f"  {small_baseline} flips: {np.mean(small_matched):.1f} ± {np.std(small_matched):.1f}")
            
            if len(r50_matched) >= 3 and np.std(r50_matched) + np.std(small_matched) > 0:
                try:
                    stat, p = wilcoxon(r50_matched, small_matched, alternative='greater')
                    print(f"  Wilcoxon test: p={p:.4f} {'✓ Significant' if p < 0.05 else '✗ Not significant'}")
                except:
                    print(f"  Wilcoxon test: insufficient variance")
    
    # Save results
    results_df.to_csv('baseline_robustness_results.csv', index=False)
    print(f"\n✅ Results saved to baseline_robustness_results.csv")
    
    # Final verdict
    print("\n" + "="*100)
    print("VERDICT")
    print("="*100)
    
    r50_mean_flips = results_df[results_df['baseline'] == 'R50']['num_flips'].mean()
    small_mean_flips = results_df[results_df['baseline'].isin(['MV3', 'EN', 'MN', 'R18'])]['num_flips'].mean()
    
    r50_mean_rho = results_df[results_df['baseline'] == 'R50']['spearman_rho'].mean()
    small_mean_rho = results_df[results_df['baseline'].isin(['MV3', 'EN', 'MN', 'R18'])]['spearman_rho'].mean()
    
    print(f"\nR50 (large baseline):")
    print(f"  Average ranking flips: {r50_mean_flips:.1f}")
    print(f"  Average Spearman ρ: {r50_mean_rho:.3f}")
    
    print(f"\nSmall baselines (MV3/EN/MN/R18):")
    print(f"  Average ranking flips: {small_mean_flips:.1f}")
    print(f"  Average Spearman ρ: {small_mean_rho:.3f}")
    
    improvement = ((r50_mean_flips - small_mean_flips) / r50_mean_flips * 100) if r50_mean_flips > 0 else 0
    print(f"\n🎯 Small baselines reduce flips by {improvement:.0f}%")
    
    if small_mean_flips < r50_mean_flips * 0.5:
        print("✅ STRONG EVIDENCE: Small baselines consistently outperform R50 across device pairs")
    elif small_mean_flips < r50_mean_flips:
        print("✓ MODERATE EVIDENCE: Small baselines generally outperform R50")
    else:
        print("✗ WEAK EVIDENCE: No consistent advantage for small baselines")
