from transformer import ChurnFeatureEngineer
import pandas as pd
import numpy as np
import os
import openpyxl
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression



if __name__ == "__main__":
    # 1. Calculate the absolute path of the 'src' directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Navigate up one level to the root directory to find the data
    DATA_PATH = os.path.join(BASE_DIR, '..', 'raw_data.xlsx')
    
    # Load your raw dataset dynamically
    df = pd.read_excel(DATA_PATH)
    
    # Define X and y
    # We explicitly drop 'customerID' and 'monthly' to prevent string errors in the model
    X = df.drop(['Churn', 'customerID', 'monthly'], axis=1, errors='ignore')
    
    # Churn is already int64, no need to map it
    y = df['Churn']
    
    # Define columns for the ColumnTransformer 
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = ['Contract', 'Tenure_Group', 'InternetService', 'PaymentMethod'] 
    
    # Setup preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ], 
        remainder='passthrough' # Passes the 1s and 0s directly to the model
    )
    
    # Setup the Logistic Regression model 
    model = LogisticRegression(C=0.001, class_weight='balanced', max_iter=100, solver='saga')
    
    # Assemble the Master Pipeline
    master_pipeline = Pipeline(steps=[
        ('engineer_features', ChurnFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train the pipeline
    master_pipeline.fit(X, y)
    
    # Calculate the absolute path of the folder where this script lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'churn_model_v1.pkl')
    
    # Export the pipeline explicitly to that path
    joblib.dump(master_pipeline, model_path)
    # print(f"Pipeline built and serialized successfully at: {model_path}")
    # print(f"Pipeline built and serialized successfully at: {model_path}")
    print("Pipeline built and serialized successfully. Ready for API deployment.")