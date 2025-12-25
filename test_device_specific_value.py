"""
Test: Does Device-Specific Normalization Capture Real Hardware-Model Interactions?
==================================================================================

Hypothesis: Different devices have different speedup ratios for the same model.
If MobileNetV2/ResNet50 latency ratio is DIFFERENT on GPU vs Edge, 
then device-specific normalization captures real signal.

If the ratios are the SAME everywhere, then universal normalization is sufficient.
"""

import pandas as pd
import numpy as np
import json
import os
from scipy.stats import kruskal, f_oneway

FRM_DIR = "agg/frm"
MODELS_JSON = "agg/models.json"

# Load models info
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

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   TESTING: Does Device-Specific Normalization Add Value?            ║
║                                                                      ║
║   Question: Are latency RATIOS different across devices?            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

baseline = "R50"
df = pd.read_csv(f"{FRM_DIR}/frm_scores_{baseline}.csv")

for col in ['ratio_flops', 'ratio_lat', 'ratio_mem', 's_latency_ms', 'memory_mib', 'acc_top1']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['group_id'] = df.apply(create_group_id, axis=1)
df['device_tier'] = df['device'].apply(get_device_tier)

# Add FLOPs from models.json
for model_id in df['model_id'].unique():
    if model_id in models_info:
        mask = df['model_id'] == model_id
        df.loc[mask, 'flops_g'] = models_info[model_id].get('flops_g', np.nan)
        df.loc[mask, 'params_m'] = models_info[model_id].get('params_m', np.nan)

print("="*80)
print("CRITICAL TEST: Are Latency Ratios Consistent Across Devices?")
print("="*80)

# For each model, look at ratio_lat distribution across device tiers
models = df['model_id'].unique()

print(f"\nAnalyzing {len(models)} models across {df['device_tier'].nunique()} device tiers\n")

results = []

for model in models:
    model_data = df[df['model_id'] == model].copy()
    
    # Get ratio_lat by tier
    gpu_ratios = model_data[model_data['device_tier'] == 'gpu']['ratio_lat'].dropna().values
    cpu_ratios = model_data[model_data['device_tier'] == 'cpu']['ratio_lat'].dropna().values
    edge_ratios = model_data[model_data['device_tier'] == 'edge']['ratio_lat'].dropna().values
    
    if len(gpu_ratios) >= 3 and len(edge_ratios) >= 3:
        # Test if GPU and Edge ratios are significantly different
        try:
            stat, p_value = kruskal(gpu_ratios, edge_ratios)
        except:
            p_value = 1.0
        
        results.append({
            'model_id': model,
            'gpu_mean': np.mean(gpu_ratios),
            'gpu_std': np.std(gpu_ratios),
            'gpu_cv': np.std(gpu_ratios) / np.mean(gpu_ratios) if np.mean(gpu_ratios) > 0 else np.nan,
            'edge_mean': np.mean(edge_ratios),
            'edge_std': np.std(edge_ratios),
            'edge_cv': np.std(edge_ratios) / np.mean(edge_ratios) if np.mean(edge_ratios) > 0 else np.nan,
            'ratio_diff': np.mean(gpu_ratios) - np.mean(edge_ratios),
            'ratio_diff_pct': 100 * (np.mean(gpu_ratios) - np.mean(edge_ratios)) / np.mean(gpu_ratios) if np.mean(gpu_ratios) > 0 else np.nan,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

results_df = pd.DataFrame(results)

print("Model Latency Ratios: GPU vs Edge")
print("-"*80)
print(f"{'Model':<15} {'GPU Mean':<12} {'Edge Mean':<12} {'Diff %':<12} {'Significant':<12}")
print("-"*80)

for _, row in results_df.iterrows():
    sig_marker = "✅ YES" if row['significant'] else "❌ NO"
    print(f"{row['model_id']:<15} {row['gpu_mean']:>11.3f} {row['edge_mean']:>11.3f} {row['ratio_diff_pct']:>11.1f}% {sig_marker:<12}")

print("-"*80)

# Summary statistics
significant_count = results_df['significant'].sum()
total_count = len(results_df)

print(f"\n📊 Summary:")
print(f"   Total models tested: {total_count}")
print(f"   Models with SIGNIFICANT tier differences: {significant_count} ({100*significant_count/total_count:.1f}%)")
print(f"   Mean absolute difference: {results_df['ratio_diff'].abs().mean():.3f}")
print(f"   Mean % difference: {results_df['ratio_diff_pct'].abs().mean():.1f}%")

print("\n💡 Interpretation:")
if significant_count / total_count > 0.5:
    print("   ✅ Latency ratios ARE device-dependent!")
    print("   ✅ Device-specific normalization captures REAL signal")
    print("   ✅ Different devices benefit different architectures differently")
else:
    print("   ❌ Latency ratios are relatively consistent across devices")
    print("   ❌ Universal normalization might be sufficient")

print("\n"+"="*80)
print("DEEP DIVE: Examples of Hardware-Model Interactions")
print("="*80)

# Show top 5 models with biggest differences
top_diff = results_df.nlargest(5, 'ratio_diff_pct')

print("\n🔍 Models MORE efficient on Edge (relative to baseline):")
print("-"*80)
for _, row in top_diff.iterrows():
    if row['ratio_diff'] > 0:  # GPU higher = Edge more efficient
        print(f"   {row['model_id']:<15} Edge is {row['ratio_diff_pct']:.1f}% faster (relative to R50)")
        print(f"                   GPU: {row['gpu_mean']:.3f}×, Edge: {row['edge_mean']:.3f}×")

bottom_diff = results_df.nsmallest(5, 'ratio_diff_pct')

print("\n🔍 Models LESS efficient on Edge (relative to baseline):")
print("-"*80)
for _, row in bottom_diff.iterrows():
    if row['ratio_diff'] < 0:  # GPU lower = Edge less efficient
        print(f"   {row['model_id']:<15} Edge is {abs(row['ratio_diff_pct']):.1f}% slower (relative to R50)")
        print(f"                   GPU: {row['gpu_mean']:.3f}×, Edge: {row['edge_mean']:.3f}×")

print("\n"+"="*80)
print("WITHIN-TIER VARIANCE TEST")
print("="*80)
print("\nQuestion: How much do ratios vary WITHIN the same tier?")
print("(High variance = device-specific characteristics matter)\n")

# For each model and tier, compute CV
variance_results = []

for model in models:
    model_data = df[df['model_id'] == model].copy()
    
    for tier in ['gpu', 'edge']:
        tier_data = model_data[model_data['device_tier'] == tier]['ratio_lat'].dropna().values
        
        if len(tier_data) >= 3:
            cv = np.std(tier_data) / np.mean(tier_data) if np.mean(tier_data) > 0 else np.nan
            variance_results.append({
                'model_id': model,
                'tier': tier,
                'cv': cv,
                'n_devices': len(tier_data)
            })

variance_df = pd.DataFrame(variance_results)

print("Coefficient of Variation (CV) within tiers:")
print("-"*80)

for tier in ['gpu', 'edge']:
    tier_cv = variance_df[variance_df['tier'] == tier]['cv'].dropna()
    print(f"{tier.upper():<10} Mean CV: {tier_cv.mean():.3f}, Median: {tier_cv.median():.3f}")

print("\n💡 Interpretation:")
mean_cv = variance_df['cv'].dropna().mean()
if mean_cv > 0.3:
    print(f"   ✅ High within-tier variance (CV={mean_cv:.3f})")
    print("   ✅ Even within same tier, different devices behave differently")
    print("   ✅ Device-specific normalization is ESSENTIAL")
else:
    print(f"   ❌ Low within-tier variance (CV={mean_cv:.3f})")
    print("   ❌ Devices within same tier behave similarly")

print("\n"+"="*80)
print("FINAL ANSWER: Should We Use Device-Specific Normalization?")
print("="*80)

cross_tier_significant = significant_count / total_count
within_tier_variance = variance_df['cv'].dropna().mean()

print(f"\n📊 Evidence Summary:")
print(f"   1. Cross-tier differences: {100*cross_tier_significant:.1f}% of models show significant differences")
print(f"   2. Within-tier variance: CV = {within_tier_variance:.3f}")

if cross_tier_significant > 0.7 and within_tier_variance > 0.3:
    print(f"\n✅ VERDICT: Device-specific normalization is JUSTIFIED")
    print(f"   • Latency ratios vary significantly across device tiers")
    print(f"   • High variance even within the same tier")
    print(f"   • Captures real hardware-model interactions")
elif cross_tier_significant > 0.7:
    print(f"\n⚠️  VERDICT: Mixed - tier matters, but within-tier consistent")
    print(f"   • Could use tier-specific normalization instead")
else:
    print(f"\n❌ VERDICT: Universal normalization might be sufficient")
    print(f"   • Latency ratios are relatively consistent")

print("\n"+"="*80)

# Save results
results_df.to_csv("analysis_results/normalization_comparison/device_tier_differences.csv", index=False)
print(f"\n💾 Saved: analysis_results/normalization_comparison/device_tier_differences.csv")
