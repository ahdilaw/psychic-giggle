"""
CRITICAL TEST: Is FRM just measuring FLOPs?
============================================

This script answers the fundamental question:
"Does FRM add value beyond just using FLOPs alone?"

Test 1: Correlation between FLOPs rankings and FRM rankings
Test 2: Cases where FRM and FLOPs disagree
Test 3: Which one predicts accuracy-efficiency trade-offs better?

If FRM ≈ FLOPs, then all 1567 runs were for nothing.
If FRM ≠ FLOPs, then there's a real contribution.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import os

FRM_DIR = "agg/frm"
OUTPUT_DIR = "analysis_results/flops_vs_frm"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

def load_data(baseline):
    file_path = f"{FRM_DIR}/frm_scores_{baseline}.csv"
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    for col in ['frm', 'ratio_flops', 'ratio_lat', 'ratio_mem', 'acc_top1']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def create_group_id(row):
    acc = row['accelerator'] if pd.notna(row['accelerator']) else 'none'
    return f"{row['device']}|{row['framework']}|{acc}"

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            CRITICAL TEST: FRM vs FLOPs                              ║
║                                                                      ║
║  Question: Is FRM just a complicated way to measure FLOPs?          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

for baseline in ["R50", "MV2", "EN"]:
    print(f"\n{'='*80}")
    print(f"BASELINE: {baseline}")
    print(f"{'='*80}\n")
    
    df = load_data(baseline)
    if df is None:
        continue
    
    df['group_id'] = df.apply(create_group_id, axis=1)
    
    # For each group, compute rankings
    results = []
    
    groups = df['group_id'].unique()
    for group in groups:
        group_data = df[df['group_id'] == group].copy()
        
        # Need at least 5 models
        if len(group_data) < 5:
            continue
        
        # Aggregate duplicates
        group_agg = group_data.groupby('model_id').agg({
            'frm': 'mean',
            'ratio_flops': 'mean',
            'ratio_lat': 'mean',
            'ratio_mem': 'mean',
            'acc_top1': 'mean'
        }).reset_index()
        
        if len(group_agg) < 5:
            continue
        
        # Compute rankings (lower is better for efficiency)
        group_agg['rank_frm'] = group_agg['frm'].rank(method='average')
        group_agg['rank_flops'] = group_agg['ratio_flops'].rank(method='average')
        group_agg['rank_lat'] = group_agg['ratio_lat'].rank(method='average')
        group_agg['rank_mem'] = group_agg['ratio_mem'].rank(method='average')
        
        # Correlation: FRM rank vs FLOPs rank
        rho_frm_flops, p_frm_flops = spearmanr(group_agg['rank_frm'], group_agg['rank_flops'])
        
        # Correlation: FRM rank vs Latency rank
        rho_frm_lat, _ = spearmanr(group_agg['rank_frm'], group_agg['rank_lat'])
        
        # Correlation: FRM rank vs Memory rank
        rho_frm_mem, _ = spearmanr(group_agg['rank_frm'], group_agg['rank_mem'])
        
        # How many rank disagreements?
        rank_diff = np.abs(group_agg['rank_frm'] - group_agg['rank_flops'])
        mean_rank_diff = rank_diff.mean()
        max_rank_diff = rank_diff.max()
        
        results.append({
            'group': group,
            'n_models': len(group_agg),
            'rho_frm_flops': rho_frm_flops,
            'rho_frm_lat': rho_frm_lat,
            'rho_frm_mem': rho_frm_mem,
            'mean_rank_diff': mean_rank_diff,
            'max_rank_diff': max_rank_diff,
            'p_value': p_frm_flops
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"📊 Groups analyzed: {len(results_df)}\n")
    
    print("="*80)
    print("CORRELATION: FRM Rankings vs Component Rankings")
    print("="*80)
    print(f"{'Comparison':<30} {'Mean ρ':<12} {'Median ρ':<12} {'Std ρ':<12}")
    print("-"*80)
    print(f"{'FRM vs FLOPs':<30} {results_df['rho_frm_flops'].mean():>11.3f} {results_df['rho_frm_flops'].median():>11.3f} {results_df['rho_frm_flops'].std():>11.3f}")
    print(f"{'FRM vs Latency':<30} {results_df['rho_frm_lat'].mean():>11.3f} {results_df['rho_frm_lat'].median():>11.3f} {results_df['rho_frm_lat'].std():>11.3f}")
    print(f"{'FRM vs Memory':<30} {results_df['rho_frm_mem'].mean():>11.3f} {results_df['rho_frm_mem'].median():>11.3f} {results_df['rho_frm_mem'].std():>11.3f}")
    print("-"*80)
    
    print(f"\n📈 RANK DISAGREEMENT ANALYSIS:")
    print(f"   Mean rank difference: {results_df['mean_rank_diff'].mean():.2f} positions")
    print(f"   Max rank difference: {results_df['max_rank_diff'].mean():.2f} positions")
    print(f"   Groups with ρ > 0.95: {(results_df['rho_frm_flops'] > 0.95).sum()} / {len(results_df)}")
    print(f"   Groups with ρ > 0.99: {(results_df['rho_frm_flops'] > 0.99).sum()} / {len(results_df)}")
    
    # Critical interpretation
    mean_rho = results_df['rho_frm_flops'].mean()
    
    print(f"\n{'='*80}")
    print("🎯 INTERPRETATION:")
    print(f"{'='*80}")
    
    if mean_rho > 0.98:
        print("❌ VERDICT: FRM ≈ FLOPs (ρ > 0.98)")
        print("   FRM rankings are essentially identical to FLOPs rankings.")
        print("   Adding latency and memory provides negligible value.")
        print("   CONCLUSION: Just use FLOPs. Your 1567 runs confirmed FLOPs works.")
    elif mean_rho > 0.90:
        print("⚠️  VERDICT: FRM ≈ FLOPs with minor deviations (0.90 < ρ < 0.98)")
        print("   FRM rankings are very similar to FLOPs rankings.")
        print("   The deviations might be noise or might be signal.")
        print("   NEED: Analyze cases where FRM ≠ FLOPs to see if deviations are meaningful.")
    elif mean_rho > 0.70:
        print("✅ VERDICT: FRM incorporates additional signal (0.70 < ρ < 0.90)")
        print("   FRM rankings differ meaningfully from FLOPs alone.")
        print("   Latency and memory contribute real information.")
        print("   CONCLUSION: FRM adds value beyond FLOPs.")
    else:
        print("🔥 VERDICT: FRM is very different from FLOPs (ρ < 0.70)")
        print("   FRM rankings are substantially different.")
        print("   This is either very good (new signal) or very bad (too much noise).")
    
    print(f"{'='*80}\n")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution of correlations
    axes[0, 0].hist(results_df['rho_frm_flops'], bins=30, alpha=0.7, 
                    color='#3498db', edgecolor='black')
    axes[0, 0].axvline(results_df['rho_frm_flops'].mean(), color='red', 
                       linestyle='--', linewidth=2, 
                       label=f'Mean: {results_df["rho_frm_flops"].mean():.3f}')
    axes[0, 0].set_xlabel('Spearman ρ (FRM vs FLOPs)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'How Similar are FRM and FLOPs Rankings?\n(Baseline: {baseline})')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Rank disagreement
    axes[0, 1].hist(results_df['mean_rank_diff'], bins=30, alpha=0.7,
                    color='#e74c3c', edgecolor='black')
    axes[0, 1].axvline(results_df['mean_rank_diff'].mean(), color='red',
                       linestyle='--', linewidth=2,
                       label=f'Mean: {results_df["mean_rank_diff"].mean():.2f}')
    axes[0, 1].set_xlabel('Mean Rank Difference (positions)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'How Much Do Rankings Disagree?\n(Baseline: {baseline})')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Comparison: FRM vs all components
    component_corrs = [
        results_df['rho_frm_flops'].values,
        results_df['rho_frm_lat'].values,
        results_df['rho_frm_mem'].values
    ]
    bp = axes[1, 0].boxplot(component_corrs, 
                            labels=['FRM↔FLOPs', 'FRM↔Latency', 'FRM↔Memory'],
                            patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#3498db', '#2ecc71', '#9b59b6']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_ylabel('Spearman ρ')
    axes[1, 0].set_title(f'FRM Correlation with Each Component\n(Baseline: {baseline})')
    axes[1, 0].grid(alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1])
    
    # 4. Scatter: rho vs n_models
    axes[1, 1].scatter(results_df['n_models'], results_df['rho_frm_flops'],
                       alpha=0.6, s=50, color='#3498db')
    axes[1, 1].set_xlabel('Number of Models')
    axes[1, 1].set_ylabel('Spearman ρ (FRM vs FLOPs)')
    axes[1, 1].set_title(f'Does Sample Size Affect Correlation?\n(Baseline: {baseline})')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].axhline(y=0.95, color='red', linestyle='--', label='ρ=0.95 threshold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/frm_vs_flops_{baseline}.png", dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {OUTPUT_DIR}/plots/frm_vs_flops_{baseline}.png\n")
    plt.close()
    
    # Save detailed results
    results_df.to_csv(f"{OUTPUT_DIR}/frm_vs_flops_detailed_{baseline}.csv", index=False)

print(f"\n{'='*80}")
print("✅ ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"\n📁 Results: {OUTPUT_DIR}/")
print("\n🎯 KEY QUESTION: Is mean ρ(FRM, FLOPs) > 0.95?")
print("   If YES → FRM ≈ FLOPs → No contribution")
print("   If NO  → FRM ≠ FLOPs → Real signal → Possible contribution")
print(f"\n{'='*80}\n")
