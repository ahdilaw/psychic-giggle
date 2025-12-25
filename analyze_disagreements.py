"""
FRM vs FLOPs Disagreement Analysis
===================================

This script identifies cases where FRM and FLOPs rankings disagree,
and analyzes whether these disagreements are meaningful.

Key Questions:
1. Which models swap positions when using FRM vs FLOPs?
2. Are there patterns (e.g., memory-heavy models, latency-bound models)?
3. Which ranking better predicts actual deployment efficiency?
4. Do disagreements occur more on certain platforms (GPU/CPU/Edge)?

This will tell us if the 10% deviation from FLOPs is signal or noise.
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
OUTPUT_DIR = "analysis_results/disagreements"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/tables", exist_ok=True)

# Load model info
with open(MODELS_JSON) as f:
    models_info = json.load(f)

def load_data(baseline):
    file_path = f"{FRM_DIR}/frm_scores_{baseline}.csv"
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    for col in ['frm', 'ratio_flops', 'ratio_lat', 'ratio_mem', 'acc_top1', 's_latency_ms', 'memory_mib']:
        if col in df.columns:
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
║        DISAGREEMENT ANALYSIS: When FRM ≠ FLOPs                      ║
║                                                                      ║
║  Goal: Find cases where rankings differ and determine which is       ║
║        "correct" based on actual deployment characteristics         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

all_disagreements = []
disagreement_summary = []

for baseline in ["R50", "MV2", "EN"]:
    print(f"\n{'='*80}")
    print(f"BASELINE: {baseline}")
    print(f"{'='*80}\n")
    
    df = load_data(baseline)
    if df is None:
        continue
    
    df['group_id'] = df.apply(create_group_id, axis=1)
    df['device_tier'] = df['device'].apply(get_device_tier)
    
    groups = df['group_id'].unique()
    
    for group in groups:
        group_data = df[df['group_id'] == group].copy()
        
        if len(group_data) < 5:
            continue
        
        # Aggregate duplicates
        group_agg = group_data.groupby('model_id').agg({
            'frm': 'mean',
            'ratio_flops': 'mean',
            'ratio_lat': 'mean',
            'ratio_mem': 'mean',
            'acc_top1': 'mean',
            's_latency_ms': 'mean',
            'memory_mib': 'mean',
            'device': 'first',
            'framework': 'first',
            'accelerator': 'first',
            'device_tier': 'first'
        }).reset_index()
        
        if len(group_agg) < 5:
            continue
        
        # Compute rankings
        group_agg['rank_frm'] = group_agg['frm'].rank(method='average')
        group_agg['rank_flops'] = group_agg['ratio_flops'].rank(method='average')
        
        # Find disagreements (rank difference > 2 positions)
        group_agg['rank_diff'] = np.abs(group_agg['rank_frm'] - group_agg['rank_flops'])
        
        disagreements = group_agg[group_agg['rank_diff'] >= 2.0].copy()
        
        if len(disagreements) > 0:
            # For each disagreement, find what it swapped with
            for _, dis_model in disagreements.iterrows():
                # Find models that are ranked differently by FRM vs FLOPs
                frm_rank = dis_model['rank_frm']
                flops_rank = dis_model['rank_flops']
                
                # Get model info
                model_info = models_info.get(dis_model['model_id'], {})
                
                all_disagreements.append({
                    'baseline': baseline,
                    'group': group,
                    'device': dis_model['device'],
                    'framework': dis_model['framework'],
                    'accelerator': dis_model['accelerator'],
                    'tier': dis_model['device_tier'],
                    'model_id': dis_model['model_id'],
                    'model_name': model_info.get('model', dis_model['model_id']),
                    'rank_frm': frm_rank,
                    'rank_flops': flops_rank,
                    'rank_diff': dis_model['rank_diff'],
                    'frm_value': dis_model['frm'],
                    'flops_ratio': dis_model['ratio_flops'],
                    'lat_ratio': dis_model['ratio_lat'],
                    'mem_ratio': dis_model['ratio_mem'],
                    'accuracy': dis_model['acc_top1'],
                    'latency_ms': dis_model['s_latency_ms'],
                    'memory_mib': dis_model['memory_mib'],
                    'params_m': model_info.get('params_m', np.nan),
                    'flops_g': model_info.get('flops_g', np.nan),
                    'direction': 'FRM_better' if frm_rank < flops_rank else 'FLOPs_better'
                })
        
        # Group-level summary
        rho, _ = spearmanr(group_agg['rank_frm'], group_agg['rank_flops'])
        disagreement_summary.append({
            'baseline': baseline,
            'group': group,
            'device': group_agg['device'].iloc[0],
            'framework': group_agg['framework'].iloc[0],
            'tier': group_agg['device_tier'].iloc[0],
            'n_models': len(group_agg),
            'n_disagreements': len(disagreements),
            'mean_rank_diff': group_agg['rank_diff'].mean(),
            'max_rank_diff': group_agg['rank_diff'].max(),
            'rho_frm_flops': rho
        })

# Convert to DataFrames
dis_df = pd.DataFrame(all_disagreements)
summary_df = pd.DataFrame(disagreement_summary)

print(f"\n{'='*80}")
print("OVERALL DISAGREEMENT STATISTICS")
print(f"{'='*80}\n")

print(f"📊 Total groups analyzed: {len(summary_df)}")
print(f"📊 Groups with disagreements (rank diff ≥ 2): {(summary_df['n_disagreements'] > 0).sum()}")
print(f"📊 Total disagreement cases: {len(dis_df)}")
print(f"📊 Mean disagreements per group: {summary_df['n_disagreements'].mean():.2f}")

if len(dis_df) > 0:
    print(f"\n{'='*80}")
    print("DISAGREEMENT PATTERNS")
    print(f"{'='*80}\n")
    
    # By tier
    print("📍 By Device Tier:")
    tier_counts = dis_df.groupby('tier').size()
    tier_total = summary_df.groupby('tier').size()
    for tier in ['gpu', 'cpu', 'edge']:
        if tier in tier_counts.index:
            count = tier_counts[tier]
            total = tier_total[tier] if tier in tier_total.index else 0
            pct = (count / len(dis_df) * 100) if len(dis_df) > 0 else 0
            print(f"   {tier.upper():<10} {count:>4} disagreements ({pct:>5.1f}%) in {total} groups")
    
    # By direction
    print(f"\n📍 By Direction:")
    direction_counts = dis_df['direction'].value_counts()
    for direction, count in direction_counts.items():
        pct = count / len(dis_df) * 100
        print(f"   {direction:<20} {count:>4} cases ({pct:>5.1f}%)")
    
    # Most frequent disagreement models
    print(f"\n📍 Models Most Frequently Disagreed Upon:")
    model_counts = dis_df.groupby('model_name').size().sort_values(ascending=False).head(10)
    for model, count in model_counts.items():
        pct = count / len(dis_df) * 100
        print(f"   {model:<30} {count:>4} times ({pct:>5.1f}%)")
    
    # Characteristics of disagreement models
    print(f"\n{'='*80}")
    print("MODEL CHARACTERISTICS IN DISAGREEMENTS")
    print(f"{'='*80}\n")
    
    # Compare models where FRM ranks better vs FLOPs ranks better
    frm_better = dis_df[dis_df['direction'] == 'FRM_better']
    flops_better = dis_df[dis_df['direction'] == 'FLOPs_better']
    
    print("Models where FRM ranks HIGHER than FLOPs:")
    print(f"   Count: {len(frm_better)}")
    print(f"   Mean latency ratio: {frm_better['lat_ratio'].mean():.3f}")
    print(f"   Mean memory ratio: {frm_better['mem_ratio'].mean():.3f}")
    print(f"   Mean FLOPs ratio: {frm_better['flops_ratio'].mean():.3f}")
    print(f"   Mean accuracy: {frm_better['accuracy'].mean():.3f}")
    
    print(f"\nModels where FLOPs ranks HIGHER than FRM:")
    print(f"   Count: {len(flops_better)}")
    print(f"   Mean latency ratio: {flops_better['lat_ratio'].mean():.3f}")
    print(f"   Mean memory ratio: {flops_better['mem_ratio'].mean():.3f}")
    print(f"   Mean FLOPs ratio: {flops_better['flops_ratio'].mean():.3f}")
    print(f"   Mean accuracy: {flops_better['accuracy'].mean():.3f}")
    
    # Statistical test
    if len(frm_better) > 5 and len(flops_better) > 5:
        from scipy import stats
        
        print(f"\n🔬 Statistical Tests (FRM_better vs FLOPs_better):")
        
        # Latency ratio
        u_lat, p_lat = stats.mannwhitneyu(frm_better['lat_ratio'].dropna(), 
                                          flops_better['lat_ratio'].dropna(), 
                                          alternative='two-sided')
        print(f"   Latency ratio: U={u_lat:.1f}, p={p_lat:.4f} {'✅ Significant' if p_lat < 0.05 else '❌ Not significant'}")
        
        # Memory ratio
        u_mem, p_mem = stats.mannwhitneyu(frm_better['mem_ratio'].dropna(), 
                                          flops_better['mem_ratio'].dropna(), 
                                          alternative='two-sided')
        print(f"   Memory ratio: U={u_mem:.1f}, p={p_mem:.4f} {'✅ Significant' if p_mem < 0.05 else '❌ Not significant'}")
        
        # Accuracy
        u_acc, p_acc = stats.mannwhitneyu(frm_better['accuracy'].dropna(), 
                                          flops_better['accuracy'].dropna(), 
                                          alternative='two-sided')
        print(f"   Accuracy: U={u_acc:.1f}, p={p_acc:.4f} {'✅ Significant' if p_acc < 0.05 else '❌ Not significant'}")
    
    # Top disagreement cases
    print(f"\n{'='*80}")
    print("TOP 20 DISAGREEMENT CASES (Largest Rank Differences)")
    print(f"{'='*80}\n")
    
    top_disagreements = dis_df.nlargest(20, 'rank_diff')
    
    print(f"{'Model':<15} {'Device':<25} {'Tier':<6} {'FRM Rank':<10} {'FLOPs Rank':<11} {'Diff':<6} {'Direction':<15}")
    print("-"*100)
    for _, row in top_disagreements.iterrows():
        print(f"{row['model_id']:<15} {row['device'][:25]:<25} {row['tier']:<6} {row['rank_frm']:>9.1f} {row['rank_flops']:>10.1f} {row['rank_diff']:>5.1f} {row['direction']:<15}")
    
    # Platform-specific patterns
    print(f"\n{'='*80}")
    print("PLATFORM-SPECIFIC DISAGREEMENT PATTERNS")
    print(f"{'='*80}\n")
    
    for tier in ['gpu', 'cpu', 'edge']:
        tier_dis = dis_df[dis_df['tier'] == tier]
        if len(tier_dis) > 0:
            print(f"\n{tier.upper()}:")
            print(f"   Total disagreements: {len(tier_dis)}")
            print(f"   Most affected models:")
            tier_models = tier_dis.groupby('model_name').size().sort_values(ascending=False).head(5)
            for model, count in tier_models.items():
                print(f"      {model:<30} {count} times")
    
    # Save detailed tables
    dis_df.to_csv(f"{OUTPUT_DIR}/tables/all_disagreements.csv", index=False)
    summary_df.to_csv(f"{OUTPUT_DIR}/tables/disagreement_summary.csv", index=False)
    print(f"\n💾 Saved: {OUTPUT_DIR}/tables/all_disagreements.csv")
    print(f"💾 Saved: {OUTPUT_DIR}/tables/disagreement_summary.csv")
    
    # Visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # 1. Disagreement heatmap by (tier, framework)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Disagreement count by tier
    tier_summary = summary_df.groupby('tier').agg({
        'n_disagreements': 'sum',
        'n_models': 'sum'
    })
    tier_summary['disagreement_rate'] = tier_summary['n_disagreements'] / tier_summary['n_models']
    
    axes[0, 0].bar(tier_summary.index, tier_summary['n_disagreements'], 
                   color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
    axes[0, 0].set_xlabel('Device Tier', fontsize=12)
    axes[0, 0].set_ylabel('Total Disagreements', fontsize=12)
    axes[0, 0].set_title('Disagreements by Device Tier', fontsize=14, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    # Disagreement characteristics
    char_data = []
    for direction in dis_df['direction'].unique():
        subset = dis_df[dis_df['direction'] == direction]
        char_data.append({
            'direction': direction,
            'lat_ratio': subset['lat_ratio'].mean(),
            'mem_ratio': subset['mem_ratio'].mean(),
            'flops_ratio': subset['flops_ratio'].mean()
        })
    
    char_df = pd.DataFrame(char_data)
    x = np.arange(len(char_df))
    width = 0.25
    
    axes[0, 1].bar(x - width, char_df['flops_ratio'], width, label='FLOPs Ratio', color='#3498db', alpha=0.8)
    axes[0, 1].bar(x, char_df['lat_ratio'], width, label='Latency Ratio', color='#e74c3c', alpha=0.8)
    axes[0, 1].bar(x + width, char_df['mem_ratio'], width, label='Memory Ratio', color='#2ecc71', alpha=0.8)
    axes[0, 1].set_ylabel('Mean Ratio Value', fontsize=12)
    axes[0, 1].set_title('Characteristics of Disagreement Models', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(char_df['direction'], rotation=15, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Rank difference distribution
    axes[1, 0].hist(dis_df['rank_diff'], bins=30, alpha=0.7, color='#9b59b6', edgecolor='black')
    axes[1, 0].axvline(dis_df['rank_diff'].mean(), color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {dis_df['rank_diff'].mean():.2f}")
    axes[1, 0].set_xlabel('Rank Difference (positions)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Distribution of Rank Differences', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Model frequency in disagreements
    top_models = dis_df.groupby('model_name').size().sort_values(ascending=False).head(10)
    axes[1, 1].barh(range(len(top_models)), top_models.values, color='#e67e22', alpha=0.8)
    axes[1, 1].set_yticks(range(len(top_models)))
    axes[1, 1].set_yticklabels(top_models.index)
    axes[1, 1].set_xlabel('Number of Disagreements', fontsize=12)
    axes[1, 1].set_title('Models Most Frequently in Disagreement', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/disagreement_analysis.png", dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {OUTPUT_DIR}/plots/disagreement_analysis.png")
    plt.close()
    
    # 2. Scatter plot: FRM rank vs FLOPs rank for all disagreements
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = {'FRM_better': '#2ecc71', 'FLOPs_better': '#e74c3c'}
    for direction in dis_df['direction'].unique():
        subset = dis_df[dis_df['direction'] == direction]
        ax.scatter(subset['rank_flops'], subset['rank_frm'], 
                  alpha=0.6, s=80, c=colors[direction], label=direction, edgecolors='black')
    
    # Diagonal line (perfect agreement)
    max_rank = max(dis_df['rank_flops'].max(), dis_df['rank_frm'].max())
    ax.plot([0, max_rank], [0, max_rank], 'k--', linewidth=2, label='Perfect Agreement')
    
    # Annotate some extreme cases
    extreme_cases = dis_df.nlargest(5, 'rank_diff')
    for _, row in extreme_cases.iterrows():
        ax.annotate(row['model_id'], 
                   xy=(row['rank_flops'], row['rank_frm']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('FLOPs Rank', fontsize=12)
    ax.set_ylabel('FRM Rank', fontsize=12)
    ax.set_title('Disagreement Cases: FRM vs FLOPs Rankings', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/frm_vs_flops_scatter.png", dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {OUTPUT_DIR}/plots/frm_vs_flops_scatter.png")
    plt.close()
    
    # 3. Create detailed case study examples
    print(f"\n{'='*80}")
    print("DETAILED CASE STUDIES")
    print(f"{'='*80}\n")
    
    # Find interesting cases
    print("Case 1: FRM ranks much higher than FLOPs")
    frm_much_better = dis_df[dis_df['direction'] == 'FRM_better'].nlargest(3, 'rank_diff')
    for idx, (_, row) in enumerate(frm_much_better.iterrows(), 1):
        print(f"\n  Example {idx}:")
        print(f"    Model: {row['model_name']} ({row['model_id']})")
        print(f"    Device: {row['device']} ({row['tier']})")
        print(f"    FRM rank: {row['rank_frm']:.0f}, FLOPs rank: {row['rank_flops']:.0f} (Δ={row['rank_diff']:.1f})")
        print(f"    Ratios: FLOPs={row['flops_ratio']:.3f}, Latency={row['lat_ratio']:.3f}, Memory={row['mem_ratio']:.3f}")
        print(f"    Accuracy: {row['accuracy']:.3f}")
        print(f"    💡 Interpretation: Low latency/memory relative to FLOPs → FRM recognizes practical efficiency")
    
    print(f"\n\nCase 2: FLOPs ranks much higher than FRM")
    flops_much_better = dis_df[dis_df['direction'] == 'FLOPs_better'].nlargest(3, 'rank_diff')
    for idx, (_, row) in enumerate(flops_much_better.iterrows(), 1):
        print(f"\n  Example {idx}:")
        print(f"    Model: {row['model_name']} ({row['model_id']})")
        print(f"    Device: {row['device']} ({row['tier']})")
        print(f"    FRM rank: {row['rank_frm']:.0f}, FLOPs rank: {row['rank_flops']:.0f} (Δ={row['rank_diff']:.1f})")
        print(f"    Ratios: FLOPs={row['flops_ratio']:.3f}, Latency={row['lat_ratio']:.3f}, Memory={row['mem_ratio']:.3f}")
        print(f"    Accuracy: {row['accuracy']:.3f}")
        print(f"    💡 Interpretation: High latency/memory relative to FLOPs → FRM penalizes runtime overhead")

else:
    print("\n⚠️  No disagreements found (all rank differences < 2)")

print(f"\n{'='*80}")
print("✅ ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"\n📁 Results: {OUTPUT_DIR}/")
print(f"   📊 Tables: {OUTPUT_DIR}/tables/")
print(f"   📈 Plots: {OUTPUT_DIR}/plots/")
print(f"\n{'='*80}\n")

print("🎯 NEXT STEPS:")
print("   1. Review disagreement cases in all_disagreements.csv")
print("   2. Compare with literature on model efficiency")
print("   3. Determine which ranking (FRM or FLOPs) better predicts:")
print("      - Actual deployment latency")
print("      - Memory constraints on edge devices")
print("      - Accuracy-efficiency trade-offs")
print("   4. Build narrative around meaningful deviations")
print(f"\n{'='*80}\n")
