# Customer Churn Prediction System

This project showcases a machine learning system designed to predict customer churn in a telecommunications company. The system is built as part of a data science workflow, involving data exploration, preprocessing, model training, and evaluation.

#Project Objective

To predict which customers are likely to discontinue their subscription (_churn_) based on their behavioral and demographic attributes, enabling the business to take proactive retention actions.

#Data Source

- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Description: Contains information about 7,043 customers including service usage, contract details, and payment behavior.
- Features include: tenure, MonthlyCharges, TotalCharges, contract type, payment method, internet service, etc.

#Approach as a Data Scientist

As a Data Scientist, the approach includes:

- 📊 **Exploratory Data Analysis (EDA)**  
  Understand distribution, class imbalance, and correlation between features and churn.

-  **Data Preprocessing**  
  - Handling missing values  
  - Encoding categorical variables  
  - Feature scaling and selection

- **Model Development**  
  - Trained multiple models: CatBoost, XGBoost, Logistic Regression  
  - Combined using **StackingClassifier** to enhance performance

- **Evaluation Metrics**  
  - Accuracy, Precision, Recall, F1-score  
  - Confusion Matrix and ROC Curve for model comparison

-  **Model Interpretability** (optional if done)  
  - Feature importance to understand key drivers of churn

##  Final Model

The final system uses a **Stacking Ensemble Classifier** combining:

- CatBoostClassifier  
- XGBoostClassifier  
- LogisticRegression  

This ensemble improves generalization and stability by leveraging the strengths of each base model.

##  Results

- Achieved accuracy: **[fill in your actual score]**  
- Key drivers of churn: contract type, tenure, monthly charges

##  Business Insight

This model can help telecom companies:

- Identify high-risk customers before they churn  
- Design targeted retention campaigns  
- Improve long-term customer value

##  Technologies Used

- Python, Pandas, Scikit-Learn  
- XGBoost, CatBoost, Logistic Regression  
- Matplotlib, Seaborn (for EDA)

---

