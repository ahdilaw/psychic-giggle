import pandas as pd

df = pd.read_csv('agg/frm/frm_scores_R50.csv')
df['group_id'] = df['device'] + '|' + df['framework'] + '|' + df['accelerator'].fillna('none')

print("GPU devices with data:")
gpu_devices = [g for g in df['group_id'].unique() if any(x in g.lower() for x in ['rtx', 'h100', 'a100', 'h200', 'l40'])]

for g in sorted(gpu_devices)[:10]:
    count = len(df[df['group_id'] == g])
    print(f"  {g}: {count} rows")
