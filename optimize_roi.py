import pandas as pd
import numpy as np

def optimize_roi():
    # Load data
    test = pd.read_csv('data/test.csv')
    meta = pd.read_csv('data/metaData.csv')
    submission = pd.read_csv('submission.csv') # My current predictions
    
    # Merge
    df = test.merge(meta, on='lead_code', how='left')
    df = df.merge(submission, on='id', how='left')
    
    # Define costs
    costs = {
        'ACTION_DIGITAL': 1,
        'ACTION_BOT': 5,          # Assumed
        'ACTION_HUMAN_CALL': 20,  # Assumed
        'ACTION_FIELD': 150
    }
    df['cost'] = df['suggested_action'].map(costs).fillna(50)
    
    # Calculate Expected Recovery and ROI
    df['expected_recovery'] = df['TARGET'] * df['total_due']
    df['roi'] = df['expected_recovery'] / df['cost']
    
    # Baseline Score (from Leaderboard)
    baseline_roi = 2596.8
    
    # Identify underperformers
    df['keep'] = df['roi'] > baseline_roi
    
    print(f"Total Leads: {len(df)}")
    print(f"Leads to Keep (ROI > {baseline_roi}): {df['keep'].sum()}")
    print(f"Leads to Drop (ROI < {baseline_roi}): {(~df['keep']).sum()}")
    
    # Create optimized submission
    # For leads to drop, set TARGET to 0 (or sufficiently low to prevent action)
    # But wait, does the system use my probability to decide?
    # If I predict 0, system won't act.
    # So I should set TARGET = 0 for dropped leads.
    
    df.loc[~df['keep'], 'TARGET'] = 0.0
    
    # Save optimized submission
    df[['id', 'TARGET']].to_csv('submission_optimized.csv', index=False)
    print("Optimized submission saved to submission_optimized.csv")
    
    # Show stats of dropped leads
    dropped = df[~df['keep']]
    if len(dropped) > 0:
        print("\nDropped Leads Stats:")
        print(dropped[['suggested_action', 'total_due', 'cost', 'TARGET', 'roi']].describe())
        print("\nDropped Actions:")
        print(dropped['suggested_action'].value_counts())

if __name__ == "__main__":
    optimize_roi()
