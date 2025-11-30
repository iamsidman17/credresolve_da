import pandas as pd
import numpy as np

def optimize_safe():
    # Load data
    test = pd.read_csv('data/test.csv')
    meta = pd.read_csv('data/metaData.csv')
    submission = pd.read_csv('submission.csv') # Original predictions
    
    # Merge
    df = test.merge(meta, on='lead_code', how='left')
    df = df.merge(submission, on='id', how='left')
    
    # Define costs
    costs = {
        'ACTION_DIGITAL': 1,
        'ACTION_BOT': 5,
        'ACTION_HUMAN_CALL': 20,
        'ACTION_FIELD': 150
    }
    df['cost'] = df['suggested_action'].map(costs).fillna(50)
    
    # Baseline Score
    baseline_roi = 2596.8
    
    # Calculate Max Possible ROI (assuming full recovery)
    df['max_roi'] = df['total_due'] / df['cost']
    
    # Safe Filter: Drop if Max ROI < Baseline
    # These leads drag down the average even if they pay 100%
    df['keep'] = df['max_roi'] > baseline_roi
    
    print(f"Total Leads: {len(df)}")
    print(f"Leads to Keep (Max ROI > {baseline_roi}): {df['keep'].sum()}")
    print(f"Leads to Drop (Max ROI < {baseline_roi}): {(~df['keep']).sum()}")
    
    # Apply filter
    # For dropped leads, set TARGET to 0 to prevent action
    df.loc[~df['keep'], 'TARGET'] = 0.0
    
    # Save optimized submission
    df[['id', 'TARGET']].to_csv('submission_safe_optimized.csv', index=False)
    print("Safe optimized submission saved to submission_safe_optimized.csv")
    
    # Stats of dropped leads
    dropped = df[~df['keep']]
    if len(dropped) > 0:
        print("\nDropped Leads Stats:")
        print(dropped[['suggested_action', 'total_due', 'cost', 'max_roi']].describe())
        print("\nDropped Actions:")
        print(dropped['suggested_action'].value_counts())

if __name__ == "__main__":
    optimize_safe()
