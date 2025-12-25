"""
Simplified Proxy Calibration Test - Debugging Version
"""

import pandas as pd
import numpy as np
import json
from scipy.stats import spearmanr

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

print("="*80)
print("SIMPLIFIED PROXY CALIBRATION TEST")
print("="*80)

# Load baseline data
baseline = "R50"
df = pd.read_csv(f"{FRM_DIR}/frm_scores_{baseline}.csv")

for col in ['frm', 'ratio_flops', 'ratio_lat', 'ratio_mem', 's_latency_ms', 'memory_mib', 'acc_top1']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['group_id'] = df.apply(create_group_id, axis=1)
df['device_tier'] = df['device'].apply(get_device_tier)

# Add FLOPs
for model_id in df['model_id'].unique():
    if model_id in models_info:
        mask = df['model_id'] == model_id
        df.loc[mask, 'flops_g'] = models_info[model_id].get('flops_g', np.nan)
        df.loc[mask, 'params_m'] = models_info[model_id].get('params_m', np.nan)

print(f"\nLoaded {len(df)} rows")
print(f"Unique devices: {df['group_id'].nunique()}")
print(f"Unique models: {df['model_id'].nunique()}")

# Select reference and target
reference_device = 'h100_sxm|torch|none'  # Use valid GPU device
edge_devices = df[df['device_tier'] == 'edge']['group_id'].unique()

print(f"\nReference device: {reference_device}")
print(f"Available edge devices: {len(edge_devices)}")

if len(edge_devices) > 0:
    target_device = edge_devices[0]
    print(f"Testing with: {target_device}")
    
    # Get data
    ref_data = df[df['group_id'] == reference_device].copy()
    target_data = df[df['group_id'] == target_device].copy()
    
    print(f"\nReference data: {len(ref_data)} rows")
    print(f"Target data: {len(target_data)} rows")
    
    if len(ref_data) >= 5 and len(target_data) >= 5:
        # Aggregate
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
        
        print(f"\nAggregated:")
        print(f"  Reference: {len(ref_agg)} models")
        print(f"  Target: {len(target_agg)} models")
        
        # Get baseline (R50)
        if 'R50' in ref_agg['model_id'].values and 'R50' in target_agg['model_id'].values:
            ref_baseline_lat = ref_agg[ref_agg['model_id'] == 'R50']['s_latency_ms'].values[0]
            ref_baseline_mem = ref_agg[ref_agg['model_id'] == 'R50']['memory_mib'].values[0]
            
            target_baseline_lat = target_agg[target_agg['model_id'] == 'R50']['s_latency_ms'].values[0]
            target_baseline_mem = target_agg[target_agg['model_id'] == 'R50']['memory_mib'].values[0]
            
            print(f"\n{'='*80}")
            print("BASELINE MEASUREMENTS")
            print(f"{'='*80}")
            print(f"\nResNet50 on Reference ({reference_device}):")
            print(f"  Latency: {ref_baseline_lat:.2f} ms")
            print(f"  Memory: {ref_baseline_mem:.2f} MB")
            
            print(f"\nResNet50 on Target ({target_device}):")
            print(f"  Latency: {target_baseline_lat:.2f} ms")
            print(f"  Memory: {target_baseline_mem:.2f} MB")
            
            print(f"\nHardware speedup factors:")
            print(f"  Latency: {target_baseline_lat / ref_baseline_lat:.1f}x slower")
            print(f"  Memory: {target_baseline_mem / ref_baseline_mem:.1f}x more")
            
            # Create proxy FRM for R50 on target device
            # "What would R50's FRM be on reference device if it had target device's characteristics?"
            ref_baseline_flops = models_info.get('R50', {}).get('flops_g', 4.089)
            
            proxy_lat_ratio = target_baseline_lat / ref_baseline_lat
            proxy_mem_ratio = target_baseline_mem / ref_baseline_mem
            proxy_flops_ratio = ref_baseline_flops / ref_baseline_flops  # Always 1.0 for same model
            
            proxy_frm = (proxy_lat_ratio * proxy_mem_ratio * proxy_flops_ratio) ** (1/3)
            
            # Actual FRM on reference device
            actual_ref_frm = ref_agg[ref_agg['model_id'] == 'R50']['frm'].values[0]
            
            # Hardware calibration factor
            hardware_factor = proxy_frm / actual_ref_frm if actual_ref_frm > 0 else 1.0
            
            print(f"\n{'='*80}")
            print("PROXY CALIBRATION")
            print(f"{'='*80}")
            print(f"\nR50 on Reference device:")
            print(f"  Actual FRM: {actual_ref_frm:.3f}")
            
            print(f"\nProxy FRM (target→reference):")
            print(f"  Lat ratio: {proxy_lat_ratio:.3f}")
            print(f"  Mem ratio: {proxy_mem_ratio:.3f}")
            print(f"  FLOPs ratio: {proxy_flops_ratio:.3f}")
            print(f"  Proxy FRM: {proxy_frm:.3f}")
            
            print(f"\n🎯 Hardware Calibration Factor: {hardware_factor:.3f}")
            print(f"   Interpretation: Target device makes models {hardware_factor:.1f}x less efficient")
            
            # Test prediction on all models
            print(f"\n{'='*80}")
            print("PREDICTION TEST")
            print(f"{'='*80}")
            
            common_models = set(ref_agg['model_id']) & set(target_agg['model_id'])
            print(f"\nTesting on {len(common_models)} common models")
            
            predictions = []
            for model in sorted(common_models):
                ref_frm = ref_agg[ref_agg['model_id'] == model]['frm'].values[0]
                target_frm_actual = target_agg[target_agg['model_id'] == model]['frm'].values[0]
                
                # Method 1: Direct transfer (naive)
                pred_direct = ref_frm
                
                # Method 2: Calibrated with hardware factor
                pred_calibrated = ref_frm * hardware_factor
                
                error_direct = abs(pred_direct - target_frm_actual) / target_frm_actual
                error_calibrated = abs(pred_calibrated - target_frm_actual) / target_frm_actual
                
                predictions.append({
                    'model': model,
                    'ref_frm': ref_frm,
                    'pred_direct': pred_direct,
                    'pred_calibrated': pred_calibrated,
                    'actual': target_frm_actual,
                    'error_direct': error_direct,
                    'error_calibrated': error_calibrated,
                })
            
            pred_df = pd.DataFrame(predictions)
            
            print(f"\n{'Model':<10} {'Ref':<8} {'Direct':<8} {'Calib':<8} {'Actual':<8} {'Err(D)':<8} {'Err(C)':<8}")
            print("─"*80)
            for _, row in pred_df.iterrows():
                print(f"{row['model']:<10} {row['ref_frm']:>7.3f} {row['pred_direct']:>7.3f} "
                      f"{row['pred_calibrated']:>7.3f} {row['actual']:>7.3f} "
                      f"{row['error_direct']:>7.1%} {row['error_calibrated']:>7.1%}")
            
            print(f"\n{'='*80}")
            print("SUMMARY")
            print(f"{'='*80}")
            
            print(f"\nMean Absolute Percentage Error:")
            print(f"  Direct transfer: {pred_df['error_direct'].mean():.1%}")
            print(f"  Calibrated transfer: {pred_df['error_calibrated'].mean():.1%}")
            print(f"  Improvement: {pred_df['error_direct'].mean() - pred_df['error_calibrated'].mean():.1%}")
            
            # Ranking correlation
            rho_direct = spearmanr(pred_df['pred_direct'], pred_df['actual'])[0]
            rho_calibrated = spearmanr(pred_df['pred_calibrated'], pred_df['actual'])[0]
            
            print(f"\nRanking Correlation (Spearman ρ):")
            print(f"  Direct transfer: {rho_direct:.3f}")
            print(f"  Calibrated transfer: {rho_calibrated:.3f}")
            print(f"  Improvement: {rho_calibrated - rho_direct:+.3f}")
            
            # Check which method is better
            better_direct = sum(pred_df['error_direct'] < pred_df['error_calibrated'])
            better_calibrated = sum(pred_df['error_calibrated'] < pred_df['error_direct'])
            
            print(f"\nPer-model comparison:")
            print(f"  Direct better: {better_direct} models")
            print(f"  Calibrated better: {better_calibrated} models")
            
        else:
            print("\n❌ R50 not found in both devices")
    else:
        print("\n❌ Insufficient data")
else:
    print("\n❌ No edge devices found")

print("\n" + "="*80)
print("✅ Test complete!")
print("="*80)
