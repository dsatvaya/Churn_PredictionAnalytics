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

#### The project followed a structured, iterative workflow:

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

```Python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model = LogisticRegression(random_state=42,class_weight={0:1,1:1.9},C=0.01,max_iter=100, solver='liblinear')
model.fit(X_train, Y_train)
Y_pred = model.predict_proba(X_test)[:,1] >= 0.52
```

### 💡 Interpretation & Strategy Formulation
- Leveraged the final model's coefficients to extract key business insights and develop prescriptive analytics.


```python
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print('\nFeature Importance:')
print(feature_importance)
```

## 📊 Final Model Performance

#### An optimized **Logistic Regression** classifier achieved a strong and balanced performance on the unseen test data.

| **Metric** | **Score** |
|-------------|-----------|
| Accuracy | 80.1% |
| Precision (for Churn) | 60.0% |
| Recall (for Churn) | 73.0% |
| F1-Score (for Churn) | 66.0% |

This model represents an excellent balance, successfully identifying nearly 3 out of 4 actual churners while maintaining high overall accuracy and improved precision over baseline models.


## 💡 Key Insights & Business Intelligence

Analysis of the final, optimized model's coefficients revealed clear, actionable personas for both at-risk and loyal customers. The model successfully moved beyond simple correlations to uncover the underlying business drivers of customer behavior.

---

### ⚠️ High-Risk Customer Profile: The Drivers of Churn
Our model identified several key indicators that point to a customer being at high risk of churning.

---

#### 💰 High MonthlyCharges & Internetservice_FiberOptics
**Insight:** The strongest predictors of churn are related to high cost. Customers with premium Fiber Optic service, and thus higher monthly bills, are the most likely to leave. This indicates extreme price sensitivity and vulnerability to competitor offers targeting this high-value segment.

---

#### 🛡️ The Engineered No_Protection_Flag
- **Insight:** Our custom-built feature proved to be the second most powerful churn predictor. Customers who subscribe to internet service but lack essential add-ons like OnlineSecurity or TechSupport are at extremely high risk. This flags a cohort that is under-invested in the ecosystem and more likely to have a negative experience when issues arise.

---

#### 💳 The "Digitally Engaged but Uncommitted" Cohort (PaperlessBilling + Payment_Echeck)
- **Insight:** The model identified a high-risk persona: customers who are digitally engaged (PaperlessBilling) but pay manually each month via Payment_Echeck. This combination creates a recurring monthly decision point, forcing a conscious re-evaluation of the service's cost, while their digital fluency makes switching to a competitor frictionless.

---

#### 🎬 Interaction Effect: StreamingMovies as a Churn Accelerant for Fiber Optic Customers
- **Insight:** A deeper analysis revealed a powerful interaction effect. While StreamingMovies is a general churn indicator, its negative impact is almost exclusively concentrated within the Fiber Optic cohort (21.9% churn rate) and is negligible for DSL users (5.6% churn rate). This is due to an "expectation gap": Fiber customers have sky-high expectations for a premium experience, and when the in-house streaming service fails to meet the standard set by market leaders like Netflix, it creates a profound sense of poor value that contaminates their perception of the core, high-cost fiber service.

---

#### 👴 Deep Dive: The SeniorCitizen Value Mismatch Cascade
- **Insight:** A root-cause analysis confirmed that high churn (30–47%) among senior citizens is driven by a severe product-value mismatch. The product catalog forces them to choose between high-cost internet tiers (DSL at ~₹55, Fiber at ~₹91) for what are often basic needs. Combined with over 50% of this cohort using manual Echeck payments, they are served a constant, conscious reminder that they are overpaying for underutilized services, inevitably leading them to seek simpler, more affordable alternatives.

---


## 💎 Loyal Customer Profile: The Drivers of Retention

The model clearly identified three distinct personas of a loyal customer. These insights provide a roadmap for the business on what behaviors and characteristics to encourage to build a stable, long-term customer base.

---

### 👨‍💼 Persona 1: The Committed Veteran
This is the cornerstone of the loyal customer base, defined by their long-term commitment and history with the company.

- **Key Indicators:** Contract (long-term), tenure, and our engineered Tenure_Group feature.  

- **Insight:** Unsurprisingly, customers locked into 1 or 2-year contracts and those who have been with the company for a long time are the least likely to churn. Our engineered Tenure_Group feature confirmed that loyalty increases significantly once a customer moves beyond the high-risk "New Customer" phase into an "Established" or "Loyal" cohort. This highlights the critical business need to survive the first year and lock customers into longer terms.

---

### 🛡️ Persona 2: The Invested Protector
This persona represents customers who are deeply integrated into the company's service ecosystem, using it for more than just a basic connection.

- **Key Indicators:** OnlineSecurity, OnlineBackup, Payment_Automatic.  

- **Insight:** These "sticky" features create significant loyalty. Protective add-ons like OnlineSecurity and OnlineBackup increase the "switching costs," as a customer would need to find and trust a new provider for these critical services. Payment_Automatic creates a frictionless relationship by removing the monthly conscious decision to pay, transforming the service into a passive utility and reducing churn opportunities.

---

### 📞 Persona 3: The Traditionalist
This persona is defined by their preference for simple, traditional services and interactions, making them naturally resistant to change.

- **Key Indicators:** Internetservice_No, Payment_Mcheck.  

- **Insight:** The model identified two key proxies for a change-averse customer. Customers with simple, phone-only plans (Internetservice_No) are a highly stable base, insulated from the intense competition in the broadband market. Similarly, those who pay by mailed check (Payment_Mcheck) represent a less digitally-savvy cohort for whom the effort of switching providers is a significant retention barrier.



## 🎯 Strategic Recommendations: A Data-Driven Action Plan

Based on the insights derived from the final predictive model, the following three-pronged strategy is recommended to reduce churn, enhance customer loyalty, and address critical gaps in the current product-market fit.

---

## 🧭 Action Plan: Data-Driven Recommendations

### 1️⃣ Mitigate High-Risk Segments
- **Launch a "Seniors Connect" Plan:** Create a new low-cost, basic-speed internet tier to resolve the product-value mismatch driving churn in the senior citizen cohort.  
- **Bundle Premium Streaming for Fiber Users:** Replace the in-house movie service with bundled 3rd-party services (e.g., *"Netflix on Us"*) to close the "expectation gap" for high-value fiber customers.  
- **Incentivize AutoPay Enrollment:** Offer a one-time credit to customers using E-check, converting this high-risk manual payment cohort to a "sticky" automatic relationship.  

---

### 2️⃣ Enhance Customer Loyalty
- **Aggressively Promote Long-Term Contracts:** Increase the value gap (discounts, exclusive perks) between monthly and yearly plans to drive customer commitment.  
- **Implement a "Smart Start" Onboarding:** Provide new customers with a free trial of the "sticky" security and backup service suite to increase ecosystem investment from day one.  

---

### 3️⃣ Protect the Stable Base
- **Maintain Simplicity for Traditionalists:** Avoid aggressive digital upselling to the highly loyal phone-only and mailed-check customer base to preserve their stability.


