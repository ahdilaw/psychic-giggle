"""
FRM Transferability Analysis: Cross-Tier Efficiency Prediction
================================================================

CRITICAL QUESTION: Can FRM rankings from one hardware tier predict rankings on another?

This script tests:
1. Cross-tier transferability (GPU → Edge, CPU → Edge, etc.)
2. Comparison: FRM vs FRM_Q vs FRM_E vs FRM_EQ
3. Baseline: Can normalized latency ratios alone transfer?
4. Statistical validation of transfer learning

This addresses the REAL contribution:
"Can we benchmark on cheap/accessible hardware and predict efficiency on target hardware?"

Author: Research Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from scipy.stats import spearmanr, kendalltau
from itertools import combinations, product
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Configuration
BASELINES = ["R50", "MV2", "EN"]
FRM_DIR = "agg/frm"
OUTPUT_DIR = "analysis_results/transferability"
MODELS_JSON = "agg/models.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/tables", exist_ok=True)

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_baseline_data(baseline):
    """Load FRM scores for a specific baseline"""
    file_path = f"{FRM_DIR}/frm_scores_{baseline}.csv"
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    numeric_cols = ['frm', 'frm_e', 'frm_q', 'frm_eq', 'acc_top1', 's_latency_ms', 
                    'memory_mib', 'ratio_mem', 'ratio_lat', 'ratio_flops', 'ratio_energy']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_device_tier(device_name):
    """Classify device into tier"""
    device_lower = str(device_name).lower()
    if any(x in device_lower for x in ['a100', 'h100', 'h200', 'l40', 'rtx', 'gpu']) and 'cpu' not in device_lower:
        return 'gpu'
    elif any(x in device_lower for x in ['cpu', 'gp_v', 'mem_v']):
        return 'cpu'
    else:
        return 'edge'

def create_group_id(row):
    """Create unique group identifier"""
    acc = row['accelerator'] if pd.notna(row['accelerator']) else 'none'
    return f"{row['device']}|{row['framework']}|{acc}"

# ============================================================================
# 1. CROSS-TIER TRANSFERABILITY TEST
# ============================================================================

def test_cross_tier_transferability(baseline, metric='frm'):
    """
    Test if rankings from source tier predict rankings in target tier
    
    Args:
        baseline: Model to use as normalization baseline
        metric: 'frm', 'frm_q', 'frm_e', 'frm_eq', or 'ratio_lat'
    """
    print(f"\n{'='*80}")
    print(f"CROSS-TIER TRANSFERABILITY: {metric.upper()} (Baseline: {baseline})")
    print(f"{'='*80}")
    
    df = load_baseline_data(baseline)
    if df is None:
        return None
    
    df['group_id'] = df.apply(create_group_id, axis=1)
    df['device_tier'] = df['device'].apply(get_device_tier)
    
    # Build ranking matrices per tier
    tiers = ['gpu', 'cpu', 'edge']
    tier_rankings = {}
    
    for tier in tiers:
        tier_data = df[df['device_tier'] == tier].copy()
        tier_groups = tier_data['group_id'].unique()
        
        ranking_matrix = {}
        for group in tier_groups:
            group_data = tier_data[tier_data['group_id'] == group].copy()
            if len(group_data) < 5:
                continue
            
            # Handle different metrics
            if metric in group_data.columns:
                group_clean = group_data.dropna(subset=[metric])
                group_clean = group_clean.groupby('model_id').agg({metric: 'mean'}).reset_index()
                
                if len(group_clean) >= 5:
                    # Lower is better for all our metrics (efficiency scores)
                    ranking_matrix[group] = group_clean.set_index('model_id')[metric].rank(method='average')
        
        if len(ranking_matrix) > 0:
            tier_rankings[tier] = pd.DataFrame(ranking_matrix)
    
    # Test all tier pairs
    transfer_results = []
    tier_pairs = [
        ('gpu', 'edge'),
        ('cpu', 'edge'),
        ('gpu', 'cpu'),
        ('edge', 'gpu'),
        ('edge', 'cpu'),
        ('cpu', 'gpu')
    ]
    
    for source_tier, target_tier in tier_pairs:
        if source_tier not in tier_rankings or target_tier not in tier_rankings:
            continue
        
        source_df = tier_rankings[source_tier]
        target_df = tier_rankings[target_tier]
        
        # Compute average ranking in source tier
        source_mean_rank = source_df.mean(axis=1).dropna()
        
        # For each target group, check correlation with source mean
        for target_group in target_df.columns:
            target_rank = target_df[target_group].dropna()
            
            # Find common models
            common_models = source_mean_rank.index.intersection(target_rank.index)
            if len(common_models) < 5:
                continue
            
            # Compute correlation
            rho, p_val = spearmanr(
                source_mean_rank[common_models],
                target_rank[common_models]
            )
            
            # Also compute Kendall's tau (more robust)
            tau, tau_p = kendalltau(
                source_mean_rank[common_models],
                target_rank[common_models]
            )
            
            transfer_results.append({
                'metric': metric,
                'source_tier': source_tier,
                'target_tier': target_tier,
                'target_group': target_group,
                'n_source_groups': len(source_df.columns),
                'n_models': len(common_models),
                'spearman_rho': rho,
                'spearman_p': p_val,
                'kendall_tau': tau,
                'kendall_p': tau_p
            })
    
    if len(transfer_results) == 0:
        print("⚠️  Insufficient data for transferability analysis")
        return None
    
    transfer_df = pd.DataFrame(transfer_results)
    
    # Summary statistics
    print(f"\n📊 Transfer Scenarios Tested: {len(transfer_df)}")
    print(f"{'─'*80}")
    
    for (source, target), group_df in transfer_df.groupby(['source_tier', 'target_tier']):
        print(f"\n{source.upper()} → {target.upper()}:")
        print(f"   Target groups tested: {len(group_df)}")
        print(f"   Mean Spearman ρ: {group_df['spearman_rho'].mean():.3f} ± {group_df['spearman_rho'].std():.3f}")
        print(f"   Median Spearman ρ: {group_df['spearman_rho'].median():.3f}")
        print(f"   Significant transfers (p<0.05): {(group_df['spearman_p'] < 0.05).sum()} / {len(group_df)}")
        print(f"   Mean Kendall τ: {group_df['kendall_tau'].mean():.3f} ± {group_df['kendall_tau'].std():.3f}")
    
    return transfer_df

# ============================================================================
# 2. COMPARE ALL FRM VARIANTS
# ============================================================================

def compare_all_frm_variants(baseline):
    """Compare transferability of FRM, FRM_Q, FRM_E, FRM_EQ, and baseline ratios"""
    print(f"\n{'='*80}")
    print(f"COMPARING ALL METRICS: Baseline = {baseline}")
    print(f"{'='*80}\n")
    
    metrics = ['frm', 'frm_q', 'frm_e', 'frm_eq', 'ratio_lat', 'ratio_mem', 'ratio_flops']
    all_results = {}
    
    for metric in metrics:
        print(f"\n{'─'*80}")
        print(f"Testing: {metric.upper()}")
        print(f"{'─'*80}")
        
        results = test_cross_tier_transferability(baseline, metric)
        if results is not None and len(results) > 0:
            all_results[metric] = results
    
    # Comparative analysis
    if len(all_results) == 0:
        return None
    
    # Build comparison table
    comparison_data = []
    
    for metric, results in all_results.items():
        for (source, target), group_df in results.groupby(['source_tier', 'target_tier']):
            comparison_data.append({
                'metric': metric,
                'transfer': f"{source}→{target}",
                'mean_rho': group_df['spearman_rho'].mean(),
                'median_rho': group_df['spearman_rho'].median(),
                'std_rho': group_df['spearman_rho'].std(),
                'success_rate': (group_df['spearman_p'] < 0.05).mean(),
                'n_tests': len(group_df)
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Overall summary by metric
    print(f"\n{'='*80}")
    print("OVERALL TRANSFERABILITY COMPARISON")
    print(f"{'='*80}\n")
    
    metric_summary = comp_df.groupby('metric').agg({
        'mean_rho': 'mean',
        'median_rho': 'median',
        'success_rate': 'mean',
        'n_tests': 'sum'
    }).round(3)
    
    print(metric_summary.to_string())
    print(f"\n{'─'*80}")
    print("📊 Higher mean_rho = Better cross-tier transferability")
    print("📊 Higher success_rate = More significant transfers")
    print(f"{'─'*80}\n")
    
    # Statistical test: Are FRM variants better than ratio_lat?
    if 'ratio_lat' in all_results:
        frm_metrics = ['frm', 'frm_q', 'frm_e', 'frm_eq']
        
        for frm_variant in frm_metrics:
            if frm_variant in all_results:
                # Get all transfer correlations
                frm_vals = all_results[frm_variant]['spearman_rho'].values
                lat_vals = all_results['ratio_lat']['spearman_rho'].values
                
                # Need equal length for comparison
                min_len = min(len(frm_vals), len(lat_vals))
                if min_len > 10:
                    frm_sample = frm_vals[:min_len]
                    lat_sample = lat_vals[:min_len]
                    
                    u_stat, p_val = stats.mannwhitneyu(frm_sample, lat_sample, alternative='greater')
                    print(f"🔬 Mann-Whitney U: {frm_variant.upper()} vs RATIO_LAT")
                    print(f"   U-statistic: {u_stat:.2f}, p-value: {p_val:.4f}")
                    
                    if p_val < 0.05:
                        print(f"   ✅ {frm_variant.upper()} transfers SIGNIFICANTLY better (p<0.05)")
                    else:
                        print(f"   ⚠️  No significant difference (p={p_val:.4f})")
                    print()
    
    # Visualization: Heatmap
    pivot_table = comp_df.pivot_table(
        values='mean_rho',
        index='metric',
        columns='transfer',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Mean Spearman ρ'})
    ax.set_title(f'Cross-Tier Transferability Heatmap (Baseline: {baseline})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Transfer Direction', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/transferability_heatmap_{baseline}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/plots/transferability_heatmap_{baseline}.pdf", bbox_inches='tight')
    print(f"💾 Saved: {OUTPUT_DIR}/plots/transferability_heatmap_{baseline}.png\n")
    plt.close()
    
    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    metric_summary_sorted = metric_summary.sort_values('mean_rho', ascending=False)
    x = np.arange(len(metric_summary_sorted))
    
    bars = ax.bar(x, metric_summary_sorted['mean_rho'], alpha=0.8, 
                  color=['#2ecc71' if m.startswith('frm') else '#e74c3c' for m in metric_summary_sorted.index])
    ax.set_ylabel('Mean Spearman ρ', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_title(f'Overall Transferability Performance (Baseline: {baseline})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_summary_sorted.index, rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/transferability_comparison_{baseline}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/plots/transferability_comparison_{baseline}.pdf", bbox_inches='tight')
    print(f"💾 Saved: {OUTPUT_DIR}/plots/transferability_comparison_{baseline}.png\n")
    plt.close()
    
    # Save tables
    comp_df.to_csv(f"{OUTPUT_DIR}/tables/transferability_detailed_{baseline}.csv", index=False)
    metric_summary.to_csv(f"{OUTPUT_DIR}/tables/transferability_summary_{baseline}.csv")
    print(f"💾 Saved: {OUTPUT_DIR}/tables/transferability_detailed_{baseline}.csv")
    print(f"💾 Saved: {OUTPUT_DIR}/tables/transferability_summary_{baseline}.csv\n")
    
    return comp_df, metric_summary

# ============================================================================
# 3. WITHIN-TIER STABILITY (Original Analysis)
# ============================================================================

def within_tier_stability(baseline):
    """Test stability WITHIN each tier (GPU-to-GPU, Edge-to-Edge, etc.)"""
    print(f"\n{'='*80}")
    print(f"WITHIN-TIER STABILITY: Baseline = {baseline}")
    print(f"{'='*80}\n")
    
    df = load_baseline_data(baseline)
    if df is None:
        return None
    
    df['group_id'] = df.apply(create_group_id, axis=1)
    df['device_tier'] = df['device'].apply(get_device_tier)
    
    metrics = ['frm', 'frm_q', 'frm_e', 'frm_eq', 'ratio_lat']
    tier_results = []
    
    for tier in ['gpu', 'cpu', 'edge']:
        tier_data = df[df['device_tier'] == tier].copy()
        tier_groups = tier_data['group_id'].unique()
        
        if len(tier_groups) < 2:
            continue
        
        for metric in metrics:
            if metric not in tier_data.columns:
                continue
            
            # Build ranking matrix
            ranking_matrix = {}
            for group in tier_groups:
                group_data = tier_data[tier_data['group_id'] == group].copy()
                if len(group_data) < 5:
                    continue
                
                group_clean = group_data.dropna(subset=[metric])
                group_clean = group_clean.groupby('model_id').agg({metric: 'mean'}).reset_index()
                
                if len(group_clean) >= 5:
                    ranking_matrix[group] = group_clean.set_index('model_id')[metric].rank(method='average')
            
            if len(ranking_matrix) < 2:
                continue
            
            rank_df = pd.DataFrame(ranking_matrix)
            
            # Compute pairwise correlations
            corrs = []
            for col1, col2 in combinations(rank_df.columns, 2):
                common = rank_df[[col1, col2]].dropna().index
                if len(common) >= 5:
                    rho, _ = spearmanr(rank_df.loc[common, col1], rank_df.loc[common, col2])
                    corrs.append(rho)
            
            if len(corrs) > 0:
                tier_results.append({
                    'tier': tier,
                    'metric': metric,
                    'mean_rho': np.mean(corrs),
                    'median_rho': np.median(corrs),
                    'std_rho': np.std(corrs),
                    'n_groups': len(rank_df.columns),
                    'n_pairs': len(corrs)
                })
    
    tier_df = pd.DataFrame(tier_results)
    
    print("📊 Within-Tier Stability Summary:")
    print(f"{'─'*80}")
    for tier in ['gpu', 'cpu', 'edge']:
        tier_subset = tier_df[tier_df['tier'] == tier]
        if len(tier_subset) > 0:
            print(f"\n{tier.upper()}:")
            print(tier_subset[['metric', 'mean_rho', 'median_rho', 'n_groups']].to_string(index=False))
    
    print(f"\n{'─'*80}\n")
    
    return tier_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║        FRM TRANSFERABILITY ANALYSIS - The Real Test                     ║
    ║                                                                          ║
    ║  Question: Can we benchmark on one device and predict rankings          ║
    ║           on devices we haven't tested?                                  ║
    ║                                                                          ║
    ║  This is the REAL contribution: Saving benchmarking costs while         ║
    ║  maintaining ranking fidelity.                                           ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n📂 Output directory: {OUTPUT_DIR}/")
    print(f"📊 Analyzing baselines: {', '.join(BASELINES)}\n")
    
    all_baseline_results = {}
    
    for baseline in BASELINES:
        print(f"\n{'#'*80}")
        print(f"# PROCESSING BASELINE: {baseline}")
        print(f"{'#'*80}\n")
        
        # 1. Within-tier stability (for reference)
        within_results = within_tier_stability(baseline)
        
        # 2. Cross-tier transferability (THE KEY TEST)
        transfer_detailed, transfer_summary = compare_all_frm_variants(baseline)
        
        all_baseline_results[baseline] = {
            'within_tier': within_results,
            'transfer_detailed': transfer_detailed,
            'transfer_summary': transfer_summary
        }
    
    # Final cross-baseline comparison
    print(f"\n{'='*80}")
    print("FINAL SUMMARY ACROSS ALL BASELINES")
    print(f"{'='*80}\n")
    
    for baseline, results in all_baseline_results.items():
        if results['transfer_summary'] is not None:
            print(f"\n{baseline}:")
            print(results['transfer_summary'].to_string())
            print()
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
    print(f"   📊 Plots: {OUTPUT_DIR}/plots/")
    print(f"   📄 Tables: {OUTPUT_DIR}/tables/")
    print(f"\n{'='*80}\n")
    
    print("🎯 KEY QUESTIONS ANSWERED:")
    print("   1. Can GPU rankings predict Edge device rankings?")
    print("   2. Is FRM better than ratio_lat for transfer?")
    print("   3. Which variant (FRM/FRM_Q/FRM_E/FRM_EQ) transfers best?")
    print("   4. Are composite metrics truly better than single ratios?")
    print(f"\n{'='*80}\n")
