import pandas as pd
import numpy as np

def analyze_breakeven():
    # Load data
    test = pd.read_csv('data/test.csv')
    meta = pd.read_csv('data/metaData.csv')
    submission = pd.read_csv('submission.csv')
    
    # Merge
    df = test.merge(meta, on='lead_code', how='left')
    df = df.merge(submission, on='id', how='left')
    
    # Define costs (estimated)
    costs = {
        'ACTION_DIGITAL': 1,      # WhatsApp/SMS
        'ACTION_BOT': 5,          # Voice Bot (guess)
        'ACTION_HUMAN_CALL': 20,  # Human Call (guess)
        'ACTION_FIELD': 150       # Field Visit
    }
    
    # Calculate Break-Even Probability
    # P_be = Cost / Total_Due
    df['cost'] = df['suggested_action'].map(costs).fillna(50) # Default to 50 if unknown
    df['break_even_prob'] = df['cost'] / df['total_due']
    
    # Compare Prediction with Break-Even
    df['should_act_model'] = df['TARGET'] > df['break_even_prob']
    
    # Count how many we would NOT act on
    not_act_count = (~df['should_act_model']).sum()
    
    print(f"Total Test Leads: {len(df)}")
    print(f"Leads where Model Prob < Break-Even Prob: {not_act_count}")
    
    # Show some examples where we should NOT act
    if not_act_count > 0:
        print("\nExamples where we should NOT act:")
        print(df[~df['should_act_model']][['id', 'suggested_action', 'total_due', 'cost', 'break_even_prob', 'TARGET']].head())
    
    # Check minimum total_due
    print(f"\nMinimum Total Due: {df['total_due'].min()}")
    print(f"Maximum Break-Even Prob: {df['break_even_prob'].max()}")
    
    # Check distribution of predictions
    print("\nPrediction Stats:")
    print(df['TARGET'].describe())

if __name__ == "__main__":
    analyze_breakeven()
