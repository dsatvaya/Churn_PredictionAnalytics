# 📊 End-to-End Customer Churn Prediction & Strategy

## 🧠 Project Overview
This project is a comprehensive, end-to-end data science workflow designed to predict customer churn for a fictional telecommunications company. 

The primary goal is to move beyond simple predictions and develop a deep understanding of the key factors driving churn. The final model and its insights are then used to formulate actionable, data-driven strategies that the business can implement to improve customer retention.

The project demonstrates a complete process from data cleaning and feature engineering to model optimization, hyperparameter tuning, and final interpretation for business value.

## 🧰 Tech Stack

**Language:**  
Python 3.x  

**Libraries:**  
- Pandas & NumPy for data manipulation  
- Scikit-learn for modeling, preprocessing, and evaluation  

**Environment:**  
Jupyter Notebook (run in VS Code)


## 🔄 Project Workflow

The project followed a structured, iterative workflow:

### 🧹 Data Cleaning & Preprocessing
Handled missing values, corrected data types, and converted categorical variables into a machine-readable format using **One-Hot and Ordinal Encoding**.

### 🧩 Feature Engineering
Architected new, high-signal features to enhance the model's predictive power. This included:

- **Tenure_Group:** Binning customer tenure into 'New', 'Established', and 'Loyal' cohorts.  
- **No_Protection_Flag:** A derivative feature flagging high-risk customers with internet service but no essential protection add-ons (like OnlineSecurity or TechSupport).

### 🤖 Model Building & Iteration
- Established a **Baseline Logistic Regression** model to benchmark performance.  
- Experimented with more complex models (**Random Forest**, **LightGBM**) and identified Logistic Regression as the most effective for this dataset.

### ⚙️ Model Optimization & Tuning
- Addressed the class imbalance inherent in the dataset using **class_weight** to optimize for the primary business goal of identifying at-risk customers (Recall).  
- Performed an exhaustive **Hyperparameter Tuning** process using **GridSearchCV** to find the optimal model settings.

### 💡 Interpretation & Strategy Formulation
- Leveraged the final model's coefficients to extract key business insights and develop prescriptive analytics.


## 📊 Final Model Performance

An optimized **Logistic Regression** classifier achieved a strong and balanced performance on the unseen test data.

| **Metric** | **Score** |
|-------------|-----------|
| Accuracy | 80.1% |
| Precision (for Churn) | 60.0% |
| Recall (for Churn) | 73.0% |
| F1-Score (for Churn) | 66.0% |

This model represents an excellent balance, successfully identifying nearly 3 out of 4 actual churners while maintaining high overall accuracy and improved precision over baseline models.


## 🔍 Key Insights from the Final Model

Analysis of the model's feature importances revealed a clear profile of both at-risk and loyal customers.

### 🔺 Key Drivers of Churn (Positive Coefficients)

- **MonthlyCharges (Importance: 0.39):**  
  The single biggest factor. Customers with higher monthly bills are more likely to churn, indicating high price sensitivity.

- **No_Protection_Flag (Importance: 0.35):**  
  Our engineered feature proved to be the #2 predictor of churn. Customers with internet service but no security or tech support add-ons are at extremely high risk.

- **Internetservice_FiberOptics (Importance: 0.28):**  
  The premium fiber service is a strong churn indicator, likely due to its high cost and aggressive competitor offers targeting these high-value customers.

