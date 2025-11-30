import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

def train_model():
    print("Loading processed data...")
    train_df = pd.read_csv('train_processed.csv')
    
    # Define features and target
    target = 'TARGET'
    drop_cols = ['id', 'lead_code', 'TARGET']
    features = [col for col in train_df.columns if col not in drop_cols]
    
    X = train_df[features]
    y = train_df[target]
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"Categorical features: {categorical_cols}")
    print(f"Numerical features: {numerical_cols}")
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )
    
    # Model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(random_state=42))
    ])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    
    # Save model
    joblib.dump(model, 'model.joblib')
    print("Model saved to model.joblib")

if __name__ == "__main__":
    train_model()
