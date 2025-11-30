import pandas as pd
import joblib

def predict():
    print("Loading processed test data...")
    test_df = pd.read_csv('test_processed.csv')
    
    print("Loading model...")
    model = joblib.load('model.joblib')
    
    # Prepare features
    drop_cols = ['id', 'lead_code']
    features = [col for col in test_df.columns if col not in drop_cols]
    
    X_test = test_df[features]
    
    print("Generating predictions...")
    predictions = model.predict(X_test)
    
    # Clip predictions to [0, 1] as probabilities
    predictions = predictions.clip(0, 1)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'TARGET': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    predict()
