import pandas as pd

def analyze_zeros():
    train = pd.read_csv('train_processed.csv')
    
    zeros = train[train['TARGET'] == 0]
    non_zeros = train[train['TARGET'] > 0]
    
    print(f"Zero Target Count: {len(zeros)}")
    
    # Compare means of numerical features
    print("\nFeature Means (Zeros vs Non-Zeros):")
    features = ['total_due', 'total_calls', 'avg_call_duration', 'answered_calls', 'no_answer_calls', 
                'total_sms', 'sms_delivered', 'total_wa', 'wa_read', 'total_visits', 'met_customer']
    
    comparison = pd.DataFrame({
        'Zero_Mean': zeros[features].mean(),
        'NonZero_Mean': non_zeros[features].mean()
    })
    print(comparison)
    
    # Compare categorical distributions
    print("\nCategorical Distributions (Zeros):")
    for col in ['dpd_bucket', 'state', 'suggested_action']:
        print(f"\n{col}:")
        print(zeros[col].value_counts(normalize=True).head())

if __name__ == "__main__":
    analyze_zeros()
