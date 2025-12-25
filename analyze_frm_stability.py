"""
FRM Stability Analysis: Cross-Platform Statistical Evaluation
==============================================================

This script performs comprehensive statistical analysis on FRM scores across
different (device, framework, accelerator) groups to evaluate:
1. Cross-device rank correlation stability
2. Variance analysis of FRM components
3. Statistical significance tests
4. Comparison with baseline metrics (latency-only, accuracy-only)

Author: Research Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from scipy.stats import spearmanr, kendalltau, friedmanchisquare, kruskal
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
BASELINES = ["R50", "MV2", "EN"]  # ResNet50, MobileNetV2, EfficientNet
FRM_DIR = "agg/frm"
OUTPUT_DIR = "analysis_results"
MODELS_JSON = "agg/models.json"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/tables", exist_ok=True)

# Configure plotting
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_baseline_data(baseline):
    """Load FRM scores for a specific baseline"""
    file_path = f"{FRM_DIR}/frm_scores_{baseline}.csv"
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: {file_path} not found")
        return None
    
    df = pd.read_csv(file_path)
    # Convert numeric columns
    numeric_cols = ['frm', 'frm_e', 'frm_q', 'frm_eq', 'acc_top1', 'acc_top5', 
                    's_latency_ms', 'memory_mib', 'ratio_mem', 'ratio_lat', 'ratio_flops']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def create_group_id(row):
    """Create unique group identifier"""
    acc = row['accelerator'] if pd.notna(row['accelerator']) else 'none'
    return f"{row['device']}|{row['framework']}|{acc}"

def get_device_tier(device_name):
    """Classify device into tier"""
    device_lower = str(device_name).lower()
    if any(x in device_lower for x in ['a100', 'h100', 'h200', 'l40', 'rtx', 'gpu']):
        return 'gpu'
    elif any(x in device_lower for x in ['cpu', 'gp_v', 'mem_v']):
        return 'cpu'
    else:
        return 'edge'

# ============================================================================
# 1. RANK CORRELATION ANALYSIS
# ============================================================================

def compute_rank_correlations(baseline):
    """Compute pairwise rank correlations across groups"""
    print(f"\n{'='*70}")
    print(f"RANK CORRELATION ANALYSIS: Baseline = {baseline}")
    print(f"{'='*70}")
    
    df = load_baseline_data(baseline)
    if df is None:
        return None
    
    # Create group identifier
    df['group_id'] = df.apply(create_group_id, axis=1)
    df['device_tier'] = df['device'].apply(get_device_tier)
    
    # Get groups with sufficient data
    group_counts = df.groupby('group_id')['model_id'].count()
    valid_groups = group_counts[group_counts >= 5].index.tolist()
    
    print(f"📊 Total groups: {len(valid_groups)}")
    print(f"📊 Models per analysis: {df['model_id'].nunique()}")
    
    results = {
        'baseline': baseline,
        'correlations': {},
        'by_tier': {},
        'summary_stats': {}
    }
    
    # Build ranking matrix for FRM
    ranking_matrix_frm = {}
    ranking_matrix_lat = {}
    ranking_matrix_acc = {}
    
    for group in valid_groups:
        group_data = df[df['group_id'] == group].copy()
        if len(group_data) < 5:
            continue
            
        # FRM rankings (lower is better = more efficient)
        group_data_clean = group_data.dropna(subset=['frm'])
        # Handle duplicates by taking mean
        group_data_clean = group_data_clean.groupby('model_id').agg({'frm': 'mean'}).reset_index()
        if len(group_data_clean) >= 5:
            ranking_matrix_frm[group] = group_data_clean.set_index('model_id')['frm'].rank(method='average')
        
        # Latency rankings (lower is better)
        group_data_clean = group_data.dropna(subset=['s_latency_ms'])
        group_data_clean = group_data_clean.groupby('model_id').agg({'s_latency_ms': 'mean'}).reset_index()
        if len(group_data_clean) >= 5:
            ranking_matrix_lat[group] = group_data_clean.set_index('model_id')['s_latency_ms'].rank(method='average')
        
        # Accuracy rankings (higher is better, so we negate for consistency)
        group_data_clean = group_data.dropna(subset=['acc_top1'])
        group_data_clean = group_data_clean.groupby('model_id').agg({'acc_top1': 'mean'}).reset_index()
        if len(group_data_clean) >= 5:
            ranking_matrix_acc[group] = group_data_clean.set_index('model_id')['acc_top1'].rank(method='average', ascending=False)
    
    # Convert to DataFrame for easier correlation computation
    rank_df_frm = pd.DataFrame(ranking_matrix_frm)
    rank_df_lat = pd.DataFrame(ranking_matrix_lat)
    rank_df_acc = pd.DataFrame(ranking_matrix_acc)
    
    print(f"\n📈 Ranking matrices built:")
    print(f"   - FRM rankings: {rank_df_frm.shape[1]} groups, {rank_df_frm.shape[0]} models")
    print(f"   - Latency rankings: {rank_df_lat.shape[1]} groups, {rank_df_lat.shape[0]} models")
    print(f"   - Accuracy rankings: {rank_df_acc.shape[1]} groups, {rank_df_acc.shape[0]} models")
    
    # Compute correlation matrices
    def safe_spearman(df):
        """Compute Spearman correlation with handling for missing data"""
        n_cols = len(df.columns)
        corr_matrix = np.zeros((n_cols, n_cols))
        p_matrix = np.ones((n_cols, n_cols))
        
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                elif i < j:
                    # Find common models
                    common_idx = df[[col1, col2]].dropna().index
                    if len(common_idx) >= 5:
                        rho, p = spearmanr(df.loc[common_idx, col1], df.loc[common_idx, col2])
                        corr_matrix[i, j] = corr_matrix[j, i] = rho
                        p_matrix[i, j] = p_matrix[j, i] = p
                    else:
                        corr_matrix[i, j] = corr_matrix[j, i] = np.nan
                        p_matrix[i, j] = p_matrix[j, i] = np.nan
        
        return pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns), \
               pd.DataFrame(p_matrix, index=df.columns, columns=df.columns)
    
    corr_frm, pval_frm = safe_spearman(rank_df_frm)
    corr_lat, pval_lat = safe_spearman(rank_df_lat)
    corr_acc, pval_acc = safe_spearman(rank_df_acc)
    
    # Extract upper triangle (excluding diagonal)
    def get_upper_triangle(corr_df):
        mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
        return corr_df.where(mask).stack().values
    
    frm_corrs = get_upper_triangle(corr_frm)
    lat_corrs = get_upper_triangle(corr_lat)
    acc_corrs = get_upper_triangle(corr_acc)
    
    # Remove NaN values
    frm_corrs = frm_corrs[~np.isnan(frm_corrs)]
    lat_corrs = lat_corrs[~np.isnan(lat_corrs)]
    acc_corrs = acc_corrs[~np.isnan(acc_corrs)]
    
    print(f"\n{'─'*70}")
    print("📊 RANK CORRELATION SUMMARY (Spearman's ρ)")
    print(f"{'─'*70}")
    print(f"\n{'Metric':<20} {'Mean':<10} {'Std':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    print(f"{'─'*70}")
    print(f"{'FRM':<20} {np.mean(frm_corrs):>9.3f} {np.std(frm_corrs):>9.3f} {np.median(frm_corrs):>9.3f} {np.min(frm_corrs):>9.3f} {np.max(frm_corrs):>9.3f}")
    print(f"{'Latency-only':<20} {np.mean(lat_corrs):>9.3f} {np.std(lat_corrs):>9.3f} {np.median(lat_corrs):>9.3f} {np.min(lat_corrs):>9.3f} {np.max(lat_corrs):>9.3f}")
    print(f"{'Accuracy-only':<20} {np.mean(acc_corrs):>9.3f} {np.std(acc_corrs):>9.3f} {np.median(acc_corrs):>9.3f} {np.min(acc_corrs):>9.3f} {np.max(acc_corrs):>9.3f}")
    print(f"{'─'*70}")
    
    # Statistical significance test: FRM vs Latency stability
    if len(frm_corrs) > 0 and len(lat_corrs) > 0:
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(frm_corrs, lat_corrs, alternative='greater')
        print(f"\n🔬 Mann-Whitney U Test (FRM > Latency):")
        print(f"   U-statistic: {u_stat:.2f}, p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"   ✅ FRM rankings are SIGNIFICANTLY more stable than latency (p < 0.05)")
        else:
            print(f"   ⚠️  No significant difference (p = {p_value:.4f})")
    
    # Store results
    results['summary_stats'] = {
        'frm': {'mean': float(np.mean(frm_corrs)), 'std': float(np.std(frm_corrs)), 
                'median': float(np.median(frm_corrs)), 'n_pairs': len(frm_corrs)},
        'latency': {'mean': float(np.mean(lat_corrs)), 'std': float(np.std(lat_corrs)),
                   'median': float(np.median(lat_corrs)), 'n_pairs': len(lat_corrs)},
        'accuracy': {'mean': float(np.mean(acc_corrs)), 'std': float(np.std(acc_corrs)),
                    'median': float(np.median(acc_corrs)), 'n_pairs': len(acc_corrs)}
    }
    
    # Visualization: Correlation distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(frm_corrs, bins=30, alpha=0.7, color='#2ecc71', edgecolor='black')
    axes[0].axvline(np.mean(frm_corrs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(frm_corrs):.3f}')
    axes[0].set_xlabel('Spearman Correlation')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'FRM Rank Stability\n(Baseline: {baseline})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(lat_corrs, bins=30, alpha=0.7, color='#e74c3c', edgecolor='black')
    axes[1].axvline(np.mean(lat_corrs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lat_corrs):.3f}')
    axes[1].set_xlabel('Spearman Correlation')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Latency Rank Stability\n(Baseline: {baseline})')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].hist(acc_corrs, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
    axes[2].axvline(np.mean(acc_corrs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(acc_corrs):.3f}')
    axes[2].set_xlabel('Spearman Correlation')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Accuracy Rank Stability\n(Baseline: {baseline})')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/rank_correlation_dist_{baseline}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/plots/rank_correlation_dist_{baseline}.pdf", bbox_inches='tight')
    print(f"\n💾 Saved: {OUTPUT_DIR}/plots/rank_correlation_dist_{baseline}.png")
    plt.close()
    
    # Tier-wise analysis
    print(f"\n{'─'*70}")
    print("📊 TIER-WISE RANK CORRELATION")
    print(f"{'─'*70}")
    
    for tier in ['gpu', 'cpu', 'edge']:
        tier_groups = [g for g in valid_groups if get_device_tier(g.split('|')[0]) == tier]
        if len(tier_groups) < 2:
            continue
        
        rank_df_tier = rank_df_frm[[g for g in tier_groups if g in rank_df_frm.columns]]
        if rank_df_tier.shape[1] < 2:
            continue
        
        corr_tier, _ = safe_spearman(rank_df_tier)
        tier_corrs = get_upper_triangle(corr_tier)
        tier_corrs = tier_corrs[~np.isnan(tier_corrs)]
        
        if len(tier_corrs) > 0:
            print(f"\n{tier.upper()}:")
            print(f"   Groups: {len(tier_groups)}, Correlations: {len(tier_corrs)}")
            print(f"   Mean ρ: {np.mean(tier_corrs):.3f} ± {np.std(tier_corrs):.3f}")
            print(f"   Median ρ: {np.median(tier_corrs):.3f}")
            
            results['by_tier'][tier] = {
                'mean': float(np.mean(tier_corrs)),
                'std': float(np.std(tier_corrs)),
                'median': float(np.median(tier_corrs)),
                'n_groups': len(tier_groups),
                'n_pairs': len(tier_corrs)
            }
    
    return results

# ============================================================================
# 2. VARIANCE ANALYSIS
# ============================================================================

def variance_analysis(baseline):
    """Analyze variance in FRM components across groups"""
    print(f"\n{'='*70}")
    print(f"VARIANCE ANALYSIS: Baseline = {baseline}")
    print(f"{'='*70}")
    
    df = load_baseline_data(baseline)
    if df is None:
        return None
    
    df['group_id'] = df.apply(create_group_id, axis=1)
    
    # For each model, compute variance across groups
    models = df['model_id'].unique()
    
    variance_results = []
    
    for model in models:
        model_data = df[df['model_id'] == model].copy()
        
        if len(model_data) < 3:  # Need at least 3 groups
            continue
        
        frm_var = model_data['frm'].var() if model_data['frm'].notna().sum() > 2 else np.nan
        lat_var = model_data['s_latency_ms'].var() if model_data['s_latency_ms'].notna().sum() > 2 else np.nan
        mem_var = model_data['memory_mib'].var() if model_data['memory_mib'].notna().sum() > 2 else np.nan
        
        # Coefficient of variation (CV = std/mean) - normalized variance
        frm_cv = model_data['frm'].std() / model_data['frm'].mean() if model_data['frm'].notna().sum() > 2 and model_data['frm'].mean() > 0 else np.nan
        lat_cv = model_data['s_latency_ms'].std() / model_data['s_latency_ms'].mean() if model_data['s_latency_ms'].notna().sum() > 2 and model_data['s_latency_ms'].mean() > 0 else np.nan
        mem_cv = model_data['memory_mib'].std() / model_data['memory_mib'].mean() if model_data['memory_mib'].notna().sum() > 2 and model_data['memory_mib'].mean() > 0 else np.nan
        
        variance_results.append({
            'model_id': model,
            'n_groups': len(model_data),
            'frm_var': frm_var,
            'latency_var': lat_var,
            'memory_var': mem_var,
            'frm_cv': frm_cv,
            'latency_cv': lat_cv,
            'memory_cv': mem_cv,
            'frm_mean': model_data['frm'].mean(),
            'latency_mean': model_data['s_latency_ms'].mean(),
            'memory_mean': model_data['memory_mib'].mean()
        })
    
    var_df = pd.DataFrame(variance_results)
    
    print(f"\n📊 Coefficient of Variation (CV) Summary:")
    print(f"{'─'*70}")
    print(f"{'Metric':<20} {'Mean CV':<12} {'Median CV':<12} {'Std CV':<12}")
    print(f"{'─'*70}")
    
    for metric in ['frm', 'latency', 'memory']:
        cv_col = f'{metric}_cv'
        cv_values = var_df[cv_col].dropna()
        if len(cv_values) > 0:
            print(f"{metric.upper():<20} {cv_values.mean():>11.3f} {cv_values.median():>11.3f} {cv_values.std():>11.3f}")
    
    print(f"{'─'*70}")
    print(f"\n💡 Lower CV = More stable across platforms")
    
    # Statistical test: Are CVs different?
    frm_cvs = var_df['frm_cv'].dropna().values
    lat_cvs = var_df['latency_cv'].dropna().values
    
    if len(frm_cvs) > 5 and len(lat_cvs) > 5:
        # Wilcoxon signed-rank test (paired)
        common_models = var_df.dropna(subset=['frm_cv', 'latency_cv'])
        if len(common_models) > 5:
            stat, p_value = stats.wilcoxon(common_models['frm_cv'], common_models['latency_cv'])
            print(f"\n🔬 Wilcoxon Signed-Rank Test (FRM CV vs Latency CV):")
            print(f"   Statistic: {stat:.2f}, p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                mean_diff = common_models['frm_cv'].mean() - common_models['latency_cv'].mean()
                if mean_diff < 0:
                    print(f"   ✅ FRM is SIGNIFICANTLY more stable (lower CV) than latency (p < 0.05)")
                else:
                    print(f"   ⚠️  Latency is more stable than FRM (p < 0.05)")
            else:
                print(f"   ⚠️  No significant difference in stability")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # CV comparison
    cv_data = var_df[['frm_cv', 'latency_cv', 'memory_cv']].dropna()
    if len(cv_data) > 0:
        bp = axes[0].boxplot([cv_data['frm_cv'], cv_data['latency_cv'], cv_data['memory_cv']],
                             labels=['FRM', 'Latency', 'Memory'],
                             patch_artist=True,
                             boxprops=dict(facecolor='lightblue', alpha=0.7),
                             medianprops=dict(color='red', linewidth=2))
        axes[0].set_ylabel('Coefficient of Variation (CV)')
        axes[0].set_title(f'Cross-Platform Stability\n(Baseline: {baseline})')
        axes[0].grid(alpha=0.3, axis='y')
        axes[0].set_ylim(bottom=0)
    
    # Per-model CV
    var_df_plot = var_df.sort_values('frm_cv')
    x_pos = np.arange(len(var_df_plot))
    axes[1].bar(x_pos, var_df_plot['frm_cv'], alpha=0.7, label='FRM CV', color='#2ecc71')
    axes[1].bar(x_pos, var_df_plot['latency_cv'], alpha=0.5, label='Latency CV', color='#e74c3c')
    axes[1].set_xlabel('Models (sorted by FRM CV)')
    axes[1].set_ylabel('Coefficient of Variation')
    axes[1].set_title(f'Per-Model Stability\n(Baseline: {baseline})')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(var_df_plot['model_id'], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/variance_analysis_{baseline}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/plots/variance_analysis_{baseline}.pdf", bbox_inches='tight')
    print(f"\n💾 Saved: {OUTPUT_DIR}/plots/variance_analysis_{baseline}.png")
    plt.close()
    
    # Save table
    var_df.to_csv(f"{OUTPUT_DIR}/tables/variance_analysis_{baseline}.csv", index=False)
    print(f"💾 Saved: {OUTPUT_DIR}/tables/variance_analysis_{baseline}.csv")
    
    return var_df

# ============================================================================
# 3. FRAMEWORK INVARIANCE TEST
# ============================================================================

def framework_invariance_analysis(baseline):
    """Test if FRM rankings are invariant across frameworks"""
    print(f"\n{'='*70}")
    print(f"FRAMEWORK INVARIANCE ANALYSIS: Baseline = {baseline}")
    print(f"{'='*70}")
    
    df = load_baseline_data(baseline)
    if df is None:
        return None
    
    # Get devices that have multiple frameworks
    device_framework_counts = df.groupby('device')['framework'].nunique()
    multi_framework_devices = device_framework_counts[device_framework_counts > 1].index.tolist()
    
    print(f"\n📊 Devices with multiple frameworks: {len(multi_framework_devices)}")
    
    invariance_results = []
    
    for device in multi_framework_devices[:20]:  # Analyze first 20 to avoid too much output
        device_data = df[df['device'] == device].copy()
        frameworks = device_data['framework'].unique()
        
        if len(frameworks) < 2:
            continue
        
        # Build ranking for each framework
        framework_rankings = {}
        for fw in frameworks:
            fw_data = device_data[device_data['framework'] == fw].dropna(subset=['frm'])
            if len(fw_data) >= 5:
                framework_rankings[fw] = fw_data.set_index('model_id')['frm'].rank(method='average')
        
        if len(framework_rankings) < 2:
            continue
        
        # Compute pairwise correlations
        fw_pairs = list(combinations(framework_rankings.keys(), 2))
        for fw1, fw2 in fw_pairs:
            common_models = framework_rankings[fw1].index.intersection(framework_rankings[fw2].index)
            if len(common_models) >= 5:
                rho, p_val = spearmanr(
                    framework_rankings[fw1][common_models],
                    framework_rankings[fw2][common_models]
                )
                invariance_results.append({
                    'device': device,
                    'framework_1': fw1,
                    'framework_2': fw2,
                    'n_models': len(common_models),
                    'spearman_rho': rho,
                    'p_value': p_val
                })
    
    if len(invariance_results) == 0:
        print("⚠️  Not enough data for framework invariance analysis")
        return None
    
    inv_df = pd.DataFrame(invariance_results)
    
    print(f"\n📊 Framework Pair Comparisons: {len(inv_df)}")
    print(f"{'─'*70}")
    print(f"Mean Spearman ρ: {inv_df['spearman_rho'].mean():.3f} ± {inv_df['spearman_rho'].std():.3f}")
    print(f"Median Spearman ρ: {inv_df['spearman_rho'].median():.3f}")
    print(f"Significant correlations (p < 0.05): {(inv_df['p_value'] < 0.05).sum()} / {len(inv_df)}")
    print(f"{'─'*70}")
    
    # Show top invariant and variant cases
    print(f"\n✅ Most Framework-Invariant Devices (Top 5):")
    top_inv = inv_df.nlargest(5, 'spearman_rho')
    for _, row in top_inv.iterrows():
        print(f"   {row['device']:<30} {row['framework_1']:<10} vs {row['framework_2']:<10} ρ={row['spearman_rho']:.3f}")
    
    print(f"\n⚠️  Least Framework-Invariant Devices (Bottom 5):")
    bottom_inv = inv_df.nsmallest(5, 'spearman_rho')
    for _, row in bottom_inv.iterrows():
        print(f"   {row['device']:<30} {row['framework_1']:<10} vs {row['framework_2']:<10} ρ={row['spearman_rho']:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(inv_df['spearman_rho'], bins=20, alpha=0.7, color='#9b59b6', edgecolor='black')
    axes[0].axvline(inv_df['spearman_rho'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f"Mean: {inv_df['spearman_rho'].mean():.3f}")
    axes[0].set_xlabel('Spearman Correlation')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Framework Invariance Distribution\n(Baseline: {baseline})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Scatter: correlation vs n_models
    axes[1].scatter(inv_df['n_models'], inv_df['spearman_rho'], alpha=0.6, s=50, color='#9b59b6')
    axes[1].set_xlabel('Number of Common Models')
    axes[1].set_ylabel('Spearman Correlation')
    axes[1].set_title(f'Framework Invariance vs Sample Size\n(Baseline: {baseline})')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/framework_invariance_{baseline}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/plots/framework_invariance_{baseline}.pdf", bbox_inches='tight')
    print(f"\n💾 Saved: {OUTPUT_DIR}/plots/framework_invariance_{baseline}.png")
    plt.close()
    
    inv_df.to_csv(f"{OUTPUT_DIR}/tables/framework_invariance_{baseline}.csv", index=False)
    print(f"💾 Saved: {OUTPUT_DIR}/tables/framework_invariance_{baseline}.csv")
    
    return inv_df

# ============================================================================
# 4. COMPARATIVE SUMMARY
# ============================================================================

def generate_comparative_summary(baselines):
    """Generate cross-baseline comparative summary"""
    print(f"\n{'='*70}")
    print(f"COMPARATIVE SUMMARY ACROSS BASELINES")
    print(f"{'='*70}\n")
    
    summary_data = []
    
    for baseline in baselines:
        print(f"Processing {baseline}...")
        
        # Rank correlations
        rank_results = compute_rank_correlations(baseline)
        
        # Variance analysis
        var_results = variance_analysis(baseline)
        
        # Framework invariance
        fw_results = framework_invariance_analysis(baseline)
        
        if rank_results:
            summary_data.append({
                'baseline': baseline,
                'frm_rank_stability_mean': rank_results['summary_stats']['frm']['mean'],
                'frm_rank_stability_std': rank_results['summary_stats']['frm']['std'],
                'latency_rank_stability_mean': rank_results['summary_stats']['latency']['mean'],
                'latency_rank_stability_std': rank_results['summary_stats']['latency']['std'],
                'accuracy_rank_stability_mean': rank_results['summary_stats']['accuracy']['mean'],
                'accuracy_rank_stability_std': rank_results['summary_stats']['accuracy']['std'],
                'frm_cv_mean': var_results['frm_cv'].mean() if var_results is not None else np.nan,
                'latency_cv_mean': var_results['latency_cv'].mean() if var_results is not None else np.nan,
                'framework_invariance_mean': fw_results['spearman_rho'].mean() if fw_results is not None else np.nan
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n{'='*70}")
    print("FINAL COMPARATIVE TABLE")
    print(f"{'='*70}\n")
    print(summary_df.to_string(index=False))
    print(f"\n{'='*70}\n")
    
    # Save
    summary_df.to_csv(f"{OUTPUT_DIR}/comparative_summary.csv", index=False)
    print(f"💾 Saved: {OUTPUT_DIR}/comparative_summary.csv")
    
    # Generate LaTeX table
    latex_table = summary_df.to_latex(index=False, float_format="%.3f")
    with open(f"{OUTPUT_DIR}/comparative_summary.tex", 'w') as f:
        f.write(latex_table)
    print(f"💾 Saved: {OUTPUT_DIR}/comparative_summary.tex")
    
    # Visualization: Comparative bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(summary_df))
    width = 0.25
    
    # Rank stability
    axes[0, 0].bar(x - width, summary_df['frm_rank_stability_mean'], width, 
                   label='FRM', color='#2ecc71', alpha=0.8)
    axes[0, 0].bar(x, summary_df['latency_rank_stability_mean'], width,
                   label='Latency', color='#e74c3c', alpha=0.8)
    axes[0, 0].bar(x + width, summary_df['accuracy_rank_stability_mean'], width,
                   label='Accuracy', color='#3498db', alpha=0.8)
    axes[0, 0].set_ylabel('Mean Spearman ρ')
    axes[0, 0].set_title('Rank Correlation Stability')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(summary_df['baseline'])
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    # CV comparison
    axes[0, 1].bar(x - width/2, summary_df['frm_cv_mean'], width,
                   label='FRM CV', color='#2ecc71', alpha=0.8)
    axes[0, 1].bar(x + width/2, summary_df['latency_cv_mean'], width,
                   label='Latency CV', color='#e74c3c', alpha=0.8)
    axes[0, 1].set_ylabel('Mean CV')
    axes[0, 1].set_title('Coefficient of Variation (Lower = More Stable)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(summary_df['baseline'])
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Framework invariance
    axes[1, 0].bar(x, summary_df['framework_invariance_mean'], width*2,
                   color='#9b59b6', alpha=0.8)
    axes[1, 0].set_ylabel('Mean Spearman ρ')
    axes[1, 0].set_title('Framework Invariance')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(summary_df['baseline'])
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Stability advantage (FRM - Latency)
    stability_advantage = summary_df['frm_rank_stability_mean'] - summary_df['latency_rank_stability_mean']
    colors = ['#2ecc71' if val > 0 else '#e74c3c' for val in stability_advantage]
    axes[1, 1].bar(x, stability_advantage, width*2, color=colors, alpha=0.8)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_ylabel('Δ Spearman ρ (FRM - Latency)')
    axes[1, 1].set_title('FRM Stability Advantage over Latency')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(summary_df['baseline'])
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/comparative_summary.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/plots/comparative_summary.pdf", bbox_inches='tight')
    print(f"💾 Saved: {OUTPUT_DIR}/plots/comparative_summary.png")
    plt.close()
    
    return summary_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║          FRM STABILITY ANALYSIS - Statistical Evaluation          ║
    ║                                                                    ║
    ║  This script analyzes FRM metric stability across platforms       ║
    ║  and compares it with baseline metrics (latency, accuracy)        ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n📂 Output directory: {OUTPUT_DIR}/")
    print(f"📊 Analyzing baselines: {', '.join(BASELINES)}\n")
    
    # Run comprehensive analysis
    results = generate_comparative_summary(BASELINES)
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
    print(f"   📊 Plots: {OUTPUT_DIR}/plots/")
    print(f"   📄 Tables: {OUTPUT_DIR}/tables/")
    print(f"   📝 Summary: {OUTPUT_DIR}/comparative_summary.csv")
    print(f"\n{'='*70}\n")
    
    print("🎯 KEY FINDINGS TO LOOK FOR:")
    print("   1. Is FRM rank correlation > latency rank correlation?")
    print("   2. Is FRM coefficient of variation < latency CV?")
    print("   3. Are framework-to-framework correlations high (> 0.7)?")
    print("   4. Statistical significance (p < 0.05) in all tests?")
    print(f"\n{'='*70}\n")
