"""
Test: Treating Baseline-on-Dev2 as "Proxy Model" on Dev1
========================================================

Your idea: Instead of directly transferring FRM, create "proxy models"
that represent how Dev2's hardware performs, measured using Dev1's metrics.

Example:
- ResNet50 on GPU: 2.5ms latency
- ResNet50 on Edge: 152ms latency
- Create "proxy_R50_edge" on GPU with 152ms latency
- Compute its FRM on GPU → tells us hardware difference

Then use this proxy FRM to calibrate transfer predictions.
"""

import pandas as pd
import numpy as np
import json
import os
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from collections import defaultdict

FRM_DIR = "agg/frm"
MODELS_JSON = "agg/models.json"
OUTPUT_DIR = "analysis_results/proxy_calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/tables", exist_ok=True)

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
║   PROXY CALIBRATION TEST                                            ║
║                                                                      ║
║   Treating baseline measurements from Dev2 as "proxy models"        ║
║   on Dev1 to calibrate cross-device transfer                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

baseline = "R50"
df = pd.read_csv(f"{FRM_DIR}/frm_scores_{baseline}.csv")

for col in ['frm', 'ratio_flops', 'ratio_lat', 'ratio_mem', 's_latency_ms', 'memory_mib', 'acc_top1']:
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
print("TEST 1: Proxy Model Calibration")
print("="*80)

# Select reference devices
reference_device = 'rtx_4090|pytorch|none'  # GPU
target_devices = df[df['device_tier'] == 'edge']['group_id'].unique()[:10]  # Sample 10 edge devices

# Calibration baselines
calibration_models = ['R50', 'MV3', 'EN', 'MV2']  # Measure these on both devices

print(f"\nReference device: {reference_device}")
print(f"Calibration models: {calibration_models}")
print(f"Testing on {len(target_devices)} target devices")

results_all = []

for target_device in target_devices:
    print(f"\n{'='*80}")
    print(f"Target device: {target_device}")
    print(f"{'='*80}")
    
    # Get reference device data
    ref_data = df[df['group_id'] == reference_device].copy()
    target_data = df[df['group_id'] == target_device].copy()
    
    if len(ref_data) < 5 or len(target_data) < 5:
        continue
    
    # Aggregate duplicates
    ref_agg = ref_data.groupby('model_id').agg({
        's_latency_ms': 'mean',
        'memory_mib': 'mean',
        'flops_g': 'mean',
        'frm': 'mean',
        'ratio_lat': 'mean',
        'ratio_mem': 'mean',
        'ratio_flops': 'mean',
    }).reset_index()
    
    target_agg = target_data.groupby('model_id').agg({
        's_latency_ms': 'mean',
        'memory_mib': 'mean',
        'flops_g': 'mean',
        'frm': 'mean',
        'ratio_lat': 'mean',
        'ratio_mem': 'mean',
        'ratio_flops': 'mean',
    }).reset_index()
    
    # Get baseline (R50) measurements
    ref_baseline_lat = ref_agg[ref_agg['model_id'] == 'R50']['s_latency_ms'].values[0]
    ref_baseline_mem = ref_agg[ref_agg['model_id'] == 'R50']['memory_mib'].values[0]
    
    target_baseline_lat = target_agg[target_agg['model_id'] == 'R50']['s_latency_ms'].values[0]
    target_baseline_mem = target_agg[target_agg['model_id'] == 'R50']['memory_mib'].values[0]
    
    print(f"\nBaseline (R50) measurements:")
    print(f"  Ref device: {ref_baseline_lat:.2f}ms latency, {ref_baseline_mem:.2f}MB memory")
    print(f"  Target device: {target_baseline_lat:.2f}ms latency, {target_baseline_mem:.2f}MB memory")
    
    # ==========================================
    # METHOD: Create proxy models
    # ==========================================
    print(f"\n{'─'*80}")
    print("Creating proxy models from target device measurements")
    print(f"{'─'*80}")
    
    proxy_frm = {}
    
    for calib_model in calibration_models:
        # Get target device measurements for this calibration model
        target_model_data = target_agg[target_agg['model_id'] == calib_model]
        if len(target_model_data) == 0:
            continue
        
        target_lat = target_model_data['s_latency_ms'].values[0]
        target_mem = target_model_data['memory_mib'].values[0]
        target_flops = target_model_data['flops_g'].values[0]
        
        # Create "proxy" FRM: What would this model's FRM be on ref device
        # if it had target device's latency/memory characteristics?
        
        # Compute ratios as if this were measured on ref device
        proxy_lat_ratio = target_lat / ref_baseline_lat
        proxy_mem_ratio = target_mem / ref_baseline_mem
        proxy_flops_ratio = target_flops / models_info.get('R50', {}).get('flops_g', 4.089)
        
        # Geometric mean
        proxy_frm_value = (proxy_lat_ratio * proxy_mem_ratio * proxy_flops_ratio) ** (1/3)
        
        # Get actual FRM on ref device
        ref_frm = ref_agg[ref_agg['model_id'] == calib_model]['frm'].values[0]
        
        proxy_frm[calib_model] = {
            'proxy_frm': proxy_frm_value,
            'actual_ref_frm': ref_frm,
            'hardware_factor': proxy_frm_value / ref_frm if ref_frm > 0 else 1.0,
        }
        
        print(f"\n  {calib_model}:")
        print(f"    Actual FRM on ref: {ref_frm:.3f}")
        print(f"    Proxy FRM (target→ref): {proxy_frm_value:.3f}")
        print(f"    Hardware factor: {proxy_frm[calib_model]['hardware_factor']:.3f}")
    
    # Compute average hardware factor
    hardware_factors = [p['hardware_factor'] for p in proxy_frm.values()]
    avg_hardware_factor = np.mean(hardware_factors)
    std_hardware_factor = np.std(hardware_factors)
    
    print(f"\n{'─'*80}")
    print(f"Hardware calibration factors:")
    print(f"  Mean: {avg_hardware_factor:.3f}")
    print(f"  Std: {std_hardware_factor:.3f}")
    print(f"  Range: [{min(hardware_factors):.3f}, {max(hardware_factors):.3f}]")
    
    # Check consistency
    if std_hardware_factor / avg_hardware_factor < 0.1:
        consistency = "✅ HIGH (baselines agree)"
    elif std_hardware_factor / avg_hardware_factor < 0.3:
        consistency = "⚠️  MEDIUM (some disagreement)"
    else:
        consistency = "❌ LOW (baselines strongly disagree)"
    
    print(f"  Consistency: {consistency}")
    
    # ==========================================
    # PREDICTION: Apply calibration to all models
    # ==========================================
    print(f"\n{'─'*80}")
    print("Predicting FRM on target device")
    print(f"{'─'*80}")
    
    predictions = {}
    
    for model in ref_agg['model_id'].unique():
        ref_frm = ref_agg[ref_agg['model_id'] == model]['frm'].values[0]
        
        # Method 1: Direct transfer (no calibration)
        pred_direct = ref_frm
        
        # Method 2: Average hardware factor
        pred_calibrated_avg = ref_frm * avg_hardware_factor
        
        # Method 3: Nearest baseline (find most similar calibration model)
        # For simplicity, use model architecture type
        if 'mobile' in model.lower() or model in ['MV2', 'MV3', 'MN']:
            nearest = 'MV3' if 'MV3' in proxy_frm else 'MV2'
        elif 'efficient' in model.lower() or model == 'EN':
            nearest = 'EN'
        else:
            nearest = 'R50'
        
        nearest_factor = proxy_frm.get(nearest, {}).get('hardware_factor', avg_hardware_factor)
        pred_calibrated_nearest = ref_frm * nearest_factor
        
        # Ground truth
        target_model_data = target_agg[target_agg['model_id'] == model]
        if len(target_model_data) > 0:
            actual_target_frm = target_model_data['frm'].values[0]
        else:
            actual_target_frm = np.nan
        
        predictions[model] = {
            'ref_frm': ref_frm,
            'pred_direct': pred_direct,
            'pred_calibrated_avg': pred_calibrated_avg,
            'pred_calibrated_nearest': pred_calibrated_nearest,
            'nearest_baseline': nearest,
            'actual_target_frm': actual_target_frm,
        }
    
    # ==========================================
    # EVALUATION: Compare predictions
    # ==========================================
    print(f"\n{'─'*80}")
    print("Prediction Results")
    print(f"{'─'*80}")
    print(f"{'Model':<10} {'Ref FRM':<10} {'Direct':<10} {'Calib':<10} {'Actual':<10} {'Error%':<10}")
    print(f"{'─'*80}")
    
    for model in sorted(predictions.keys()):
        p = predictions[model]
        if np.isnan(p['actual_target_frm']):
            continue
        
        error_direct = abs(p['pred_direct'] - p['actual_target_frm']) / p['actual_target_frm']
        error_calibrated = abs(p['pred_calibrated_avg'] - p['actual_target_frm']) / p['actual_target_frm']
        
        print(f"{model:<10} {p['ref_frm']:>9.3f} {p['pred_direct']:>9.3f} "
              f"{p['pred_calibrated_avg']:>9.3f} {p['actual_target_frm']:>9.3f} "
              f"{error_calibrated:>9.1%}")
    
    # Compute ranking correlations
    valid_models = [m for m in predictions.keys() if not np.isnan(predictions[m]['actual_target_frm'])]
    
    if len(valid_models) >= 5:
        ref_ranks = [predictions[m]['ref_frm'] for m in valid_models]
        pred_direct_ranks = [predictions[m]['pred_direct'] for m in valid_models]
        pred_calibrated_ranks = [predictions[m]['pred_calibrated_avg'] for m in valid_models]
        actual_ranks = [predictions[m]['actual_target_frm'] for m in valid_models]
        
        rho_direct = spearmanr(pred_direct_ranks, actual_ranks)[0]
        rho_calibrated = spearmanr(pred_calibrated_ranks, actual_ranks)[0]
        
        print(f"\n{'─'*80}")
        print("Ranking Correlations:")
        print(f"  Direct transfer: ρ={rho_direct:.3f}")
        print(f"  Calibrated transfer: ρ={rho_calibrated:.3f}")
        print(f"  Improvement: {rho_calibrated - rho_direct:+.3f}")
        
        results_all.append({
            'target_device': target_device,
            'hardware_factor_mean': avg_hardware_factor,
            'hardware_factor_std': std_hardware_factor,
            'consistency': consistency,
            'rho_direct': rho_direct,
            'rho_calibrated': rho_calibrated,
            'improvement': rho_calibrated - rho_direct,
            'n_models': len(valid_models),
        })

# ==========================================
# SUMMARY ANALYSIS
# ==========================================
if len(results_all) > 0:
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results_all)
    
    print(f"\nTested {len(results_df)} target devices")
    print(f"\nDirect transfer:")
    print(f"  Mean ρ: {results_df['rho_direct'].mean():.3f}")
    print(f"  Median ρ: {results_df['rho_direct'].median():.3f}")
    
    print(f"\nCalibrated transfer (proxy method):")
    print(f"  Mean ρ: {results_df['rho_calibrated'].mean():.3f}")
    print(f"  Median ρ: {results_df['rho_calibrated'].median():.3f}")
    
    improvements = results_df['improvement']
    print(f"\nImprovements:")
    print(f"  Mean: {improvements.mean():+.3f}")
    print(f"  Devices improved: {sum(improvements > 0)} / {len(improvements)}")
    print(f"  Devices worsened: {sum(improvements < 0)} / {len(improvements)}")
    
    print(f"\nHardware factor statistics:")
    print(f"  Mean: {results_df['hardware_factor_mean'].mean():.3f}")
    print(f"  Std across devices: {results_df['hardware_factor_mean'].std():.3f}")
    
    print(f"\nConsistency breakdown:")
    for level in ['✅ HIGH', '⚠️  MEDIUM', '❌ LOW']:
        count = sum(results_df['consistency'] == level)
        print(f"  {level}: {count} devices ({100*count/len(results_df):.1f}%)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Direct vs Calibrated correlation
    axes[0, 0].scatter(results_df['rho_direct'], results_df['rho_calibrated'], alpha=0.6)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[0, 0].set_xlabel('Direct Transfer ρ')
    axes[0, 0].set_ylabel('Calibrated Transfer ρ')
    axes[0, 0].set_title('Proxy Calibration Effect')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Improvement distribution
    axes[0, 1].hist(improvements, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Improvement in ρ')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Improvements')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Hardware factor vs improvement
    axes[1, 0].scatter(results_df['hardware_factor_std'], improvements, alpha=0.6)
    axes[1, 0].set_xlabel('Hardware Factor Std Dev (consistency)')
    axes[1, 0].set_ylabel('Improvement in ρ')
    axes[1, 0].set_title('Consistency vs Improvement')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Top/bottom performers
    sorted_results = results_df.sort_values('improvement', ascending=False)
    top5 = sorted_results.head(5)
    bottom5 = sorted_results.tail(5)
    
    axes[1, 1].barh(range(5), top5['improvement'].values, color='green', alpha=0.6, label='Top 5')
    axes[1, 1].barh(range(5, 10), bottom5['improvement'].values, color='red', alpha=0.6, label='Bottom 5')
    axes[1, 1].set_yticks(range(10))
    axes[1, 1].set_yticklabels(list(top5['target_device'].values) + list(bottom5['target_device'].values), fontsize=8)
    axes[1, 1].set_xlabel('Improvement in ρ')
    axes[1, 1].set_title('Best/Worst Performing Devices')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/proxy_calibration_summary.png", dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved: {OUTPUT_DIR}/plots/proxy_calibration_summary.png")
    plt.close()
    
    # Save results
    results_df.to_csv(f"{OUTPUT_DIR}/tables/proxy_calibration_results.csv", index=False)
    print(f"💾 Saved: {OUTPUT_DIR}/tables/proxy_calibration_results.csv")

print("\n" + "="*80)
print("✅ Analysis complete!")
print("="*80)
