"""
Alternative Metric Formulations: Testing Universal vs Device-Specific Normalization
====================================================================================

This script tests different ways to combine FLOPs, Memory, and Latency:

1. Device-specific normalization (your current approach)
2. Universal normalization (FLOPs and Params constant, only latency varies)
3. Absolute metrics with z-score normalization
4. Log-scale composite

Goal: Determine which formulation produces the most transferable rankings.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

FRM_DIR = "agg/frm"
MODELS_JSON = "agg/models.json"
OUTPUT_DIR = "analysis_results/normalization_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

# Load model info
with open(MODELS_JSON) as f:
    models_info = json.load(f)

def load_data(baseline):
    file_path = f"{FRM_DIR}/frm_scores_{baseline}.csv"
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    for col in ['frm', 'ratio_flops', 'ratio_lat', 'ratio_mem', 'acc_top1', 
                's_latency_ms', 'memory_mib', 'model_id']:
        if col in df.columns:
            if col != 'model_id':
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

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

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   NORMALIZATION COMPARISON: Device-Specific vs Universal            ║
║                                                                      ║
║   Question: Should we normalize to a device-specific baseline or    ║
║            use universal (architecture-only) metrics?                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

baseline = "R50"

df = load_data(baseline)
if df is None:
    print(f"❌ Could not load data for baseline {baseline}")
    exit(1)

df['group_id'] = df.apply(create_group_id, axis=1)
df['device_tier'] = df['device'].apply(get_device_tier)

# Add universal metrics (architecture properties)
for model_id in df['model_id'].unique():
    if model_id in models_info:
        mask = df['model_id'] == model_id
        df.loc[mask, 'flops_g'] = models_info[model_id].get('flops_g', np.nan)
        df.loc[mask, 'params_m'] = models_info[model_id].get('params_m', np.nan)

# Universal baseline (ResNet50)
universal_baseline_flops = models_info.get('R50', {}).get('flops_g', 4.089)  # GFLOPs
universal_baseline_params = models_info.get('R50', {}).get('params_m', 25.557)  # Million params

print(f"\n📊 Universal Baseline (ResNet50):")
print(f"   FLOPs: {universal_baseline_flops:.3f} GFLOPs")
print(f"   Params: {universal_baseline_params:.3f} M")

# Test different metric formulations
results = []

groups = df['group_id'].unique()

for group in groups:
    group_data = df[df['group_id'] == group].copy()
    
    if len(group_data) < 5:
        continue
    
    # Aggregate duplicates
    group_agg = group_data.groupby('model_id').agg({
        'frm': 'mean',  # Method 1: Current approach (device-specific)
        'ratio_flops': 'mean',
        'ratio_lat': 'mean',
        'ratio_mem': 'mean',
        's_latency_ms': 'mean',
        'memory_mib': 'mean',
        'flops_g': 'mean',
        'params_m': 'mean',
        'acc_top1': 'mean',
        'device': 'first',
        'framework': 'first',
        'device_tier': 'first'
    }).reset_index()
    
    if len(group_agg) < 5:
        continue
    
    # === Method 1: Current FRM (device-specific normalization) ===
    group_agg['metric_1_frm'] = group_agg['frm']
    
    # === Method 2: Universal FLOPs + Device-specific latency ===
    # FLOPs ratio is universal (same across devices)
    # Latency ratio is device-specific
    # Params ratio is universal
    group_agg['universal_flops_ratio'] = group_agg['flops_g'] / universal_baseline_flops
    group_agg['universal_params_ratio'] = group_agg['params_m'] / universal_baseline_params
    
    # Combine: Universal FLOPs + Universal Params + Device-specific Latency
    group_agg['metric_2_universal'] = (
        group_agg['universal_flops_ratio'] * 
        group_agg['universal_params_ratio'] * 
        group_agg['ratio_lat']
    ) ** (1/3)
    
    # === Method 3: Pure Universal (FLOPs + Params only, ignore runtime) ===
    group_agg['metric_3_pure_universal'] = (
        group_agg['universal_flops_ratio'] * 
        group_agg['universal_params_ratio']
    ) ** (1/2)
    
    # === Method 4: Z-score normalized absolute values ===
    # Standardize each metric independently
    for metric in ['s_latency_ms', 'memory_mib', 'flops_g']:
        if group_agg[metric].std() > 0:
            group_agg[f'{metric}_zscore'] = (group_agg[metric] - group_agg[metric].mean()) / group_agg[metric].std()
        else:
            group_agg[f'{metric}_zscore'] = 0
    
    # Lower is better, so negate (we want efficiency)
    group_agg['metric_4_zscore'] = -(
        group_agg['s_latency_ms_zscore'] + 
        group_agg['memory_mib_zscore'] + 
        group_agg['flops_g_zscore']
    ) / 3
    
    # Compute rankings
    for method in ['metric_1_frm', 'metric_2_universal', 'metric_3_pure_universal', 'metric_4_zscore']:
        group_agg[f'{method}_rank'] = group_agg[method].rank(method='average')
    
    # Also rank by FLOPs alone
    group_agg['flops_only_rank'] = group_agg['ratio_flops'].rank(method='average')
    
    # Store results
    results.append({
        'group': group,
        'device': group_agg['device'].iloc[0],
        'framework': group_agg['framework'].iloc[0],
        'tier': group_agg['device_tier'].iloc[0],
        'n_models': len(group_agg),
        # Method 1 vs FLOPs
        'rho_method1_flops': spearmanr(group_agg['metric_1_frm_rank'], group_agg['flops_only_rank'])[0],
        # Method 2 vs FLOPs
        'rho_method2_flops': spearmanr(group_agg['metric_2_universal_rank'], group_agg['flops_only_rank'])[0],
        # Method 3 vs FLOPs (should be very high - it's almost the same)
        'rho_method3_flops': spearmanr(group_agg['metric_3_pure_universal_rank'], group_agg['flops_only_rank'])[0],
        # Method 4 vs FLOPs
        'rho_method4_flops': spearmanr(group_agg['metric_4_zscore_rank'], group_agg['flops_only_rank'])[0],
        # Method 1 vs Method 2 (how similar are they?)
        'rho_method1_method2': spearmanr(group_agg['metric_1_frm_rank'], group_agg['metric_2_universal_rank'])[0],
    })

results_df = pd.DataFrame(results)

print(f"\n{'='*80}")
print("COMPARISON: Device-Specific vs Universal Normalization")
print(f"{'='*80}\n")

print(f"📊 Groups analyzed: {len(results_df)}\n")

print("Method Descriptions:")
print("  Method 1 (Current FRM): Device-specific normalization for all metrics")
print("  Method 2 (Universal): Universal FLOPs+Params, device-specific latency")
print("  Method 3 (Pure Universal): FLOPs+Params only (no latency)")
print("  Method 4 (Z-score): Standardized absolute values\n")

print("="*80)
print("How different are these methods from FLOPs-only?")
print("="*80)
print(f"{'Method':<30} {'Mean ρ':<12} {'Median ρ':<12} {'Std ρ':<12}")
print("-"*80)
print(f"{'Method 1 (Current)':<30} {results_df['rho_method1_flops'].mean():>11.3f} {results_df['rho_method1_flops'].median():>11.3f} {results_df['rho_method1_flops'].std():>11.3f}")
print(f"{'Method 2 (Universal)':<30} {results_df['rho_method2_flops'].mean():>11.3f} {results_df['rho_method2_flops'].median():>11.3f} {results_df['rho_method2_flops'].std():>11.3f}")
print(f"{'Method 3 (Pure Universal)':<30} {results_df['rho_method3_flops'].mean():>11.3f} {results_df['rho_method3_flops'].median():>11.3f} {results_df['rho_method3_flops'].std():>11.3f}")
print(f"{'Method 4 (Z-score)':<30} {results_df['rho_method4_flops'].mean():>11.3f} {results_df['rho_method4_flops'].median():>11.3f} {results_df['rho_method4_flops'].std():>11.3f}")
print("-"*80)

print(f"\n💡 Interpretation:")
print(f"   Higher ρ = More similar to FLOPs-only ranking")
print(f"   Lower ρ = More deviation from FLOPs (adds more signal or noise)")

# Cross-tier transferability test
print(f"\n{'='*80}")
print("CROSS-TIER TRANSFERABILITY TEST")
print(f"{'='*80}\n")

print("Testing: GPU → Edge transfer correlation")
print("(Can rankings from GPU predict rankings on Edge?)\n")

# For each method, test GPU → Edge transfer
for tier_pair in [('gpu', 'edge'), ('cpu', 'edge')]:
    source_tier, target_tier = tier_pair
    
    print(f"\n{source_tier.upper()} → {target_tier.upper()}:")
    
    # Get source and target groups
    source_groups = results_df[results_df['tier'] == source_tier]['group'].values
    target_groups = results_df[results_df['tier'] == target_tier]['group'].values
    
    if len(source_groups) == 0 or len(target_groups) == 0:
        print(f"   ⚠️  Insufficient data")
        continue
    
    # For each method, compute average ranking in source
    # Then test correlation with each target group
    
    for method_name, method_col in [
        ('Method 1 (Current)', 'metric_1_frm'),
        ('Method 2 (Universal)', 'metric_2_universal'),
        ('Method 3 (Pure Universal)', 'metric_3_pure_universal')
    ]:
        transfer_corrs = []
        
        # Get source tier data
        source_data = df[df['device_tier'] == source_tier].copy()
        source_groups_list = source_data['group_id'].unique()
        
        # Build source ranking matrix
        source_rankings = {}
        for src_group in source_groups_list:
            src_group_data = source_data[source_data['group_id'] == src_group].copy()
            if len(src_group_data) < 5:
                continue
            
            src_agg = src_group_data.groupby('model_id').agg({
                's_latency_ms': 'mean',
                'memory_mib': 'mean',
                'flops_g': 'mean',
                'params_m': 'mean',
                'ratio_flops': 'mean',
                'ratio_lat': 'mean',
                'ratio_mem': 'mean',
                'frm': 'mean'
            }).reset_index()
            
            # Compute metric
            if method_col == 'metric_1_frm':
                src_agg['metric'] = src_agg['frm']
            elif method_col == 'metric_2_universal':
                src_agg['metric'] = (
                    (src_agg['flops_g'] / universal_baseline_flops) *
                    (src_agg['params_m'] / universal_baseline_params) *
                    src_agg['ratio_lat']
                ) ** (1/3)
            elif method_col == 'metric_3_pure_universal':
                src_agg['metric'] = (
                    (src_agg['flops_g'] / universal_baseline_flops) *
                    (src_agg['params_m'] / universal_baseline_params)
                ) ** (1/2)
            
            source_rankings[src_group] = src_agg.set_index('model_id')['metric'].rank(method='average')
        
        if len(source_rankings) == 0:
            continue
        
        # Average source rankings
        source_df = pd.DataFrame(source_rankings)
        source_mean_rank = source_df.mean(axis=1)
        
        # Test against target groups
        target_data = df[df['device_tier'] == target_tier].copy()
        target_groups_list = target_data['group_id'].unique()
        
        for tgt_group in target_groups_list[:20]:  # Limit to avoid too much computation
            tgt_group_data = target_data[target_data['group_id'] == tgt_group].copy()
            if len(tgt_group_data) < 5:
                continue
            
            tgt_agg = tgt_group_data.groupby('model_id').agg({
                's_latency_ms': 'mean',
                'memory_mib': 'mean',
                'flops_g': 'mean',
                'params_m': 'mean',
                'ratio_flops': 'mean',
                'ratio_lat': 'mean',
                'ratio_mem': 'mean',
                'frm': 'mean'
            }).reset_index()
            
            # Compute metric
            if method_col == 'metric_1_frm':
                tgt_agg['metric'] = tgt_agg['frm']
            elif method_col == 'metric_2_universal':
                tgt_agg['metric'] = (
                    (tgt_agg['flops_g'] / universal_baseline_flops) *
                    (tgt_agg['params_m'] / universal_baseline_params) *
                    tgt_agg['ratio_lat']
                ) ** (1/3)
            elif method_col == 'metric_3_pure_universal':
                tgt_agg['metric'] = (
                    (tgt_agg['flops_g'] / universal_baseline_flops) *
                    (tgt_agg['params_m'] / universal_baseline_params)
                ) ** (1/2)
            
            tgt_rank = tgt_agg.set_index('model_id')['metric'].rank(method='average')
            
            # Common models
            common = source_mean_rank.index.intersection(tgt_rank.index)
            if len(common) >= 5:
                rho, _ = spearmanr(source_mean_rank[common], tgt_rank[common])
                transfer_corrs.append(rho)
        
        if len(transfer_corrs) > 0:
            print(f"   {method_name:<30} ρ={np.mean(transfer_corrs):.3f} ± {np.std(transfer_corrs):.3f}")

print(f"\n{'='*80}")
print("✅ KEY INSIGHT:")
print(f"{'='*80}\n")
print("If Method 2 (Universal) transfers BETTER than Method 1 (Current),")
print("then you SHOULD have used universal normalization!")
print("\nIf Method 1 transfers BETTER, then device-specific normalization")
print("is justified (captures platform-specific characteristics).")
print(f"\n{'='*80}\n")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Correlation with FLOPs
methods = ['Method 1\n(Current)', 'Method 2\n(Universal)', 'Method 3\n(Pure\nUniversal)', 'Method 4\n(Z-score)']
mean_rhos = [
    results_df['rho_method1_flops'].mean(),
    results_df['rho_method2_flops'].mean(),
    results_df['rho_method3_flops'].mean(),
    results_df['rho_method4_flops'].mean()
]

axes[0, 0].bar(methods, mean_rhos, color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'], alpha=0.8)
axes[0, 0].set_ylabel('Mean Spearman ρ with FLOPs-only', fontsize=11)
axes[0, 0].set_title('Similarity to FLOPs-only Ranking', fontsize=13, fontweight='bold')
axes[0, 0].set_ylim([0, 1])
axes[0, 0].grid(alpha=0.3, axis='y')
axes[0, 0].tick_params(axis='x', labelsize=9)

# 2. Method 1 vs Method 2 correlation
axes[0, 1].hist(results_df['rho_method1_method2'], bins=30, alpha=0.7, color='#3498db', edgecolor='black')
axes[0, 1].axvline(results_df['rho_method1_method2'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {results_df['rho_method1_method2'].mean():.3f}")
axes[0, 1].set_xlabel('Spearman ρ', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Current vs Universal: How Similar?', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Distribution comparison
data_for_box = [
    results_df['rho_method1_flops'].dropna().values,
    results_df['rho_method2_flops'].dropna().values,
    results_df['rho_method3_flops'].dropna().values,
    results_df['rho_method4_flops'].dropna().values
]

bp = axes[1, 0].boxplot(data_for_box, labels=['M1\n(Curr)', 'M2\n(Univ)', 'M3\n(Pure)', 'M4\n(Z)'],
                        patch_artist=True)
for patch, color in zip(bp['boxes'], ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1, 0].set_ylabel('Spearman ρ with FLOPs-only', fontsize=11)
axes[1, 0].set_title('Distribution of Correlations', fontsize=13, fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='y')
axes[1, 0].set_ylim([0, 1])

# 4. Text summary
axes[1, 1].axis('off')
summary_text = f"""
SUMMARY

Current Approach (Method 1):
• Device-specific normalization
• ρ(FLOPs) = {results_df['rho_method1_flops'].mean():.3f}

Universal Approach (Method 2):
• Universal FLOPs + Params
• Device-specific latency
• ρ(FLOPs) = {results_df['rho_method2_flops'].mean():.3f}

Pure Universal (Method 3):
• FLOPs + Params only
• No runtime metrics
• ρ(FLOPs) = {results_df['rho_method3_flops'].mean():.3f}

Similarity (M1 vs M2):
• ρ = {results_df['rho_method1_method2'].mean():.3f}

CONCLUSION:
{'Universal normalization might be better!' if results_df['rho_method2_flops'].mean() < results_df['rho_method1_flops'].mean() else 'Current normalization is justified.'}
"""

axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/normalization_comparison.png", dpi=300, bbox_inches='tight')
print(f"💾 Saved: {OUTPUT_DIR}/plots/normalization_comparison.png\n")
plt.close()

results_df.to_csv(f"{OUTPUT_DIR}/normalization_comparison_results.csv", index=False)
print(f"💾 Saved: {OUTPUT_DIR}/normalization_comparison_results.csv\n")

print(f"{'='*80}\n")
