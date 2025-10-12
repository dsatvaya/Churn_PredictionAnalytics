import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset

file_path = 'Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

# target variable
Y = df['Churn']
# features
X = df.drop('Churn', axis=1)

# test train split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# scaling the numerical features
nemerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'totalSevice']
scaler = StandardScaler()
X_train[nemerical_cols] = scaler.fit_transform(X_train[nemerical_cols])
X_test[nemerical_cols] = scaler.fit_transform(X_test[nemerical_cols])



# logistic regression model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model = LogisticRegression(random_state=42,class_weight='balanced',C=0.01,max_iter=300, solver='liblinear')
model.fit(X_train, Y_train)
Y_pred = model.predict_proba(X_test)[:,1] >= 0.52


# evaluation metrics
cnf_matrix = confusion_matrix(Y_test, Y_pred)
print('Confusioin Matrix:')
print(cnf_matrix)

class_report = classification_report(Y_test, Y_pred)
print('\nClassification Report:')
print(class_report)

acc_score = accuracy_score(Y_test, Y_pred)
print('\nAccuracy Score:')
print(acc_score)



# feature analysis

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print('\nFeature Importance:')
print(feature_importance)
