# src/transformers.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_eng = X.copy()
        
        # 1. Map Yes/No to 1/0 
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
                       
        for col in binary_cols:
            if col in X_eng.columns:
                X_eng[col] = X_eng[col].map({
                    'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 
                    'No internet service': 0, 'No phone service': 0, 
                    1: 1, 0: 0
                }).fillna(0)

        # 2. Gender mapping 
        if 'gender' in X_eng.columns:
            X_eng['gender'] = X_eng['gender'].map({'Female': 1, 'Male': 0, 'female': 1, 'male': 0}).fillna(0)

        # 3. Force TotalCharges to numeric
        if 'TotalCharges' in X_eng.columns:
            X_eng['TotalCharges'] = pd.to_numeric(X_eng['TotalCharges'], errors='coerce').fillna(0)
        
        # 4. Tenure Binning
        if 'tenure' in X_eng.columns:
            bins = [0, 12, 48, 100]
            labels = ['New_Customer', 'Established_Customer', 'Loyal_Customer']
            X_eng['Tenure_Group'] = pd.cut(X_eng['tenure'], bins=bins, labels=labels, right=False)
        
        # 5. Risk Flag 
        if 'InternetService' in X_eng.columns and 'OnlineSecurity' in X_eng.columns and 'TechSupport' in X_eng.columns:
            condition = (X_eng['InternetService'] != 'No') & (X_eng['OnlineSecurity'] == 0) & (X_eng['TechSupport'] == 0)
            X_eng['No_Protection_Flag'] = np.where(condition, 1, 0)
        
        return X_eng