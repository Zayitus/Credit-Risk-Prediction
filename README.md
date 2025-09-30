Credit Risk Prediction
Predictive model for credit default risk using machine learning classification techniques.
Overview
This project builds a binary classification model to predict credit default risk using historical loan data. The model achieves 0.95 AUC-ROC score and uses SHAP values for model interpretability, making predictions explainable for business stakeholders.
Dataset

Source: Kaggle Credit Risk Dataset
Size: 32,581 loan records
Features: 12 variables including demographics, loan characteristics, and credit history
Target: Binary classification (default vs. non-default)
Class distribution: 21.8% default rate (imbalanced)

Key Results
MetricLogistic RegressionXGBoostAUC-ROC0.870.95Accuracy0.860.92Precision (Default)0.750.83Recall (Default)0.550.79
Methodology
1. Data Preprocessing

Handled missing values using median imputation
Applied One-Hot Encoding for categorical variables
Standardized numerical features using StandardScaler
Created income brackets for analysis

2. Feature Engineering

Encoded categorical variables (home ownership, loan intent, loan grade)
Scaled numerical features to comparable ranges
Addressed class imbalance using scale_pos_weight parameter

3. Model Development

Baseline: Logistic Regression for comparison
Main Model: XGBoost Classifier optimized for imbalanced data
Validation: 80/20 train-test split with stratification

4. Model Interpretability

SHAP (SHapley Additive exPlanations) for feature importance
Global interpretation: Which features matter most overall
Local interpretation: Why individual predictions were made

Key Findings
Most Important Features for Default Prediction:

person_income - Higher income strongly reduces default risk
loan_percent_income - Higher loan-to-income ratio increases risk
loan_int_rate - Higher interest rates correlate with defaults
loan_intent_VENTURE - Venture loans show higher risk
person_home_ownership - Ownership status affects risk profile

Business Insights:

Low-income applicants with high loan-to-income ratios are highest risk
Interest rate is both a predictor and consequence of risk
Loan grade F/G customers have significantly elevated default probability
Age shows concentration of defaults in 20-30 year range with high loan amounts

Technologies Used

Python 3.8+
Data Processing: pandas, numpy
Visualization: matplotlib, seaborn
Machine Learning: scikit-learn, XGBoost
Interpretability: SHAP
Environment: Google Colab

Installation & Usage
bash# Clone repository
git clone [your-repo-url]

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook Credit_Risk_Analysis.ipynb
Project Structure
credit-risk-prediction/
├── Credit_Risk_Analysis.ipynb    # Main analysis notebook
├── credit_risk_dataset.csv       # Dataset
├── README.md                      # This file
└── requirements.txt               # Dependencies
Requirements
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
shap>=0.40.0
Future Improvements

Hyperparameter tuning using GridSearchCV
Test ensemble methods (Random Forest, LightGBM)
Implement cost-sensitive learning
Deploy model as REST API
Add feature selection analysis

License
MIT License
Contact
Gastón Schvartz
Email: schvartz.g@gmail.com
LinkedIn: gaston-schvartz

Credit risk analysis project demonstrating end-to-end machine learning workflow from data preprocessing to model interpretability.
