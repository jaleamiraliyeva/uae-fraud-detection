# UAE E-Commerce Fraud Detection System Documentation

## Project Overview

### Business Problem

Online e-commerce platforms and digital payment systems face significant financial losses due to fraudulent transactions. Fraudulent payment activity can result in:

* Financial losses from chargebacks
* Reputation damage
* Increased operational investigation costs
* Customer trust issues
* Regulatory compliance risks

This project was developed to detect potentially fraudulent e-commerce transactions in the UAE market using machine learning techniques.

The system predicts whether a transaction is likely to be fraudulent based on transactional behavior, customer activity patterns, payment characteristics, velocity signals, and risk indicators.

---

# Project Objective

The primary objective of this project is:

* Detect fraudulent transactions in real time
* Minimize financial losses caused by fraud
* Reduce false positive fraud alerts
* Improve fraud investigation efficiency
* Provide interpretable fraud risk predictions

The solution includes:

* End-to-end machine learning pipeline
* Advanced feature engineering
* Fraud risk scoring
* Streamlit-based prediction dashboard
* Model evaluation and visualization tools

---

# Dataset Overview

## Dataset Name

`uae_ecom_fraud_100k.csv`

## Dataset Characteristics

| Attribute       | Value                       |
| --------------- | --------------------------- |
| Domain          | UAE E-Commerce Transactions |
| Problem Type    | Binary Classification       |
| Target Variable | `is_fraud`                  |
| Dataset Size    | 100,000 transactions        |
| Fraud Class     | Fraudulent Transaction      |
| Non-Fraud Class | Legitimate Transaction      |

---

# Target Variable

| Value | Meaning                |
| ----- | ---------------------- |
| 0     | Legitimate Transaction |
| 1     | Fraudulent Transaction |

---

# Business Understanding of Fraud Indicators

The dataset contains transactional and behavioral indicators commonly associated with fraudulent activity.

Examples include:

* High transaction amounts
* Rapid transaction frequency
* Suspicious IP risk scores
* Newly created accounts
* Chargeback history
* Transactions during unusual hours
* Temporary email domains
* Device anomalies
* Geographic mismatches

---

# Exploratory Data Analysis (EDA)

## Objectives of EDA

EDA was performed to:

* Understand transaction behavior
* Identify fraud patterns
* Detect data quality issues
* Analyze feature distributions
* Discover feature relationships
* Identify predictive fraud indicators

---

## Data Quality Analysis

### Issues Identified

| Issue                   | Description                                   |
| ----------------------- | --------------------------------------------- |
| Missing Values          | Some columns contained missing information    |
| Skewed Distributions    | Financial variables were highly skewed        |
| Outliers                | Extreme transaction amounts existed           |
| Categorical Variability | Multiple categorical payment-related features |
| Behavioral Noise        | Transaction velocity fluctuations             |
| Risk Signal Imbalance   | Fraud-related flags unevenly distributed      |

---

## Key EDA Findings

### Fraud Patterns Observed

Fraudulent transactions showed stronger association with:

* High IP risk scores
* High transaction velocity
* New customer accounts
* Odd-hour transactions
* Repeated chargeback behavior
* Higher transaction values
* Disposable email domains

### Customer Behavior Insights

Fraudulent users often demonstrated:

* Abnormal spending behavior
* Short account lifetime
* Increased activity within short time windows
* Risky behavioral combinations

---

# Machine Learning Pipeline Architecture

The project uses a structured machine learning pipeline to ensure reproducibility and prevent data leakage.

Pipeline Order:

```text
Raw Data
   ↓
AbsTransformer
   ↓
Feature Engineering
   ↓
Outlier Treatment
   ↓
Skewness Transformation
   ↓
Scaling
   ↓
Logistic Regression Model
```

---

# Custom Feature Engineering

A custom transformer named `FraudFeatureEngineer` was implemented.

Purpose:

* Create fraud-related behavioral features
* Build interaction variables
* Generate risk indicators
* Encode categorical variables
* Prevent training leakage

---

# FraudFeatureEngineer Details

## Feature Engineering Logic

### 1. Ratio Features

#### amount_per_item_price

Measures transaction amount relative to average product price.

```python
amount_aed / (avg_item_price + 1)
```

Business Meaning:

* Detects suspiciously expensive purchases
* Detects quantity manipulation behavior

---

#### amount_per_card_age

Measures spending relative to card age.

```python
amount_aed / (card_age_days + 1)
```

Business Meaning:

* Detects risky behavior on newly created cards

---

### 2. Interaction Features

#### amount_x_risk

Combines transaction amount with IP risk.

```python
amount_aed * ip_risk_score
```

Business Meaning:

* High-value risky transactions become more suspicious

---

### 3. High-Risk Fraud Signals

#### fresh_high_value

```python
(user_account_age_days < 14) AND (amount_aed > 1000)
```

Business Meaning:

* New users making expensive purchases are high risk

---

#### high_amount

```python
amount_aed > 2000
```

Business Meaning:

* Very large transactions may indicate fraud attempts

---

#### risk_velocity

```python
ip_risk_score * transactions_last_1h
```

Business Meaning:

* High transaction activity combined with risky IP behavior

---

#### night_high_risk

```python
fraud_flag_odd_hour == 1 AND ip_risk_score > 70
```

Business Meaning:

* Suspicious late-night activity from risky IPs

---

### 4. Behavioral Features

#### velocity_ratio

```python
transactions_last_1h / (transactions_last_24h + 1)
```

Business Meaning:

* Detects transaction bursts

---

### 5. User Historical Statistics

The model learns customer behavior from training data.

Generated Features:

* user_mean_amount
* user_std_amount
* user_max_amount
* user_txn_count

Business Meaning:

* Detects deviation from normal customer behavior

---

### 6. Time-Based Features

Generated Features:

* day_of_week
* is_weekend
* hour_bin

Business Meaning:

* Fraud often occurs during unusual time periods

---

### 7. Aggregated Risk Features

#### combined_risk_score

```python
Sum of fraud-related flags
```

Business Meaning:

* Aggregates multiple fraud warning signals

---

### 8. Disposable Email Detection

#### is_disposable_email

Checks for temporary email providers.

Examples:

* mailinator
* guerrillamail
* tempmail

Business Meaning:

* Temporary emails are commonly used in fraud attempts

---

# Categorical Encoding Strategy

Label Encoding was used for categorical features.

Encoded Columns:

* payment_method
* device_type
* browser
* merchant_category
* shipping_city
* billing_city
* bin_country

Unknown categories are encoded as:

```python
-1
```

This prevents inference failures during prediction.

---

# Columns Removed From Model

The following columns were removed because they:

* introduced leakage
* were identifiers
* duplicated information
* were not useful for prediction

Dropped Examples:

* user_id
* transaction_id
* timestamp_utc
* ip_address
* email_domain
* currency

---

# Outlier Treatment

Custom transformer:

```python
OutlierCapper
```

Method:

* Caps values between 1st and 99th percentile

Purpose:

* Prevent extreme transaction amounts from destabilizing the model

---

# Skewness Transformation

Custom transformer:

```python
SkewnessTransformer
```

Method:

```python
log1p()
```

Applied to:

* highly skewed positive numerical features

Purpose:

* Normalize feature distributions
* Improve Logistic Regression performance

---

# Model Selection

## Models Considered

Multiple approaches were evaluated.

Final Selected Model:

```python
LogisticRegression
```

Reason for Selection:

* Interpretable predictions
* Fast inference speed
* Stable performance
* Suitable for real-time fraud scoring
* Lower deployment complexity

---

# Hyperparameter Tuning

GridSearchCV was used.

Parameters Tested:

```python
param_grid_lr = {
    'model__C': [0.01, 0.1, 1, 10, 50],
    'model__solver': ['liblinear', 'lbfgs']
}
```

Optimization Metric:

```python
ROC-AUC
```

Cross Validation:

```python
5-Fold Stratified Cross Validation
```

---

# Model Evaluation Metrics

## Metrics Used

| Metric           | Purpose                              |
| ---------------- | ------------------------------------ |
| ROC-AUC          | Overall classification performance   |
| PR-AUC           | Performance on imbalanced data       |
| Precision        | Accuracy of fraud predictions        |
| Recall           | Ability to detect fraud              |
| F1-Score         | Balance between precision and recall |
| Confusion Matrix | Error analysis                       |

---

# Business Interpretation of Metrics

## Precision

High precision means:

* fewer false fraud alerts
* lower investigation costs
* better customer experience

---

## Recall

High recall means:

* fewer fraudulent transactions missed
* reduced financial losses

In fraud systems, recall is extremely important.

---

# Streamlit Application

The project includes a Streamlit-based fraud prediction dashboard.

File:

```text
deployment/app.py
```

---

# Streamlit Features

## 1. Fraud Prediction Interface

Allows users to:

* enter transaction information
* simulate fraud scenarios
* generate fraud probability predictions

---

## 2. Model Metrics Dashboard

Displays:

* ROC-AUC
* F1-score
* Precision
* Recall

---

## 3. ROC and Precision-Recall Curves

Provides visual evaluation of model performance.

---

## 4. Confusion Matrix Visualization

Helps analyze:

* false positives
* false negatives
* fraud detection success

---

## 5. Pipeline Information Page

Displays:

* pipeline structure
* feature engineering details
* model configuration

---

# Prediction Workflow

```text
User Input
   ↓
AbsTransformer
   ↓
Feature Engineering
   ↓
Encoding
   ↓
Outlier Processing
   ↓
Skewness Transformation
   ↓
Scaling
   ↓
Fraud Probability Prediction
```

---

# Project Structure

```text
project/
│
├── data/
│   └── uae_ecom_fraud_100k.csv
│
├── deployment/
│   └── app.py
│
├── documentation/
│   └── documentation.md
│
├── model/
│   └── fraud_model8.pkl
│
├── notebooks/
│   └── EDAt2.ipynb
│   └── feature_engineering.ipynb
│   └── modeling.ipynb
│
├── presentation/
│   └── fraud_detection_dashbord.pbix
│   └── UAE Fraud Detection Modeling Pipeline.pdf
│
└── requirements/
    └── requirements.txt

```

---

# Model Serialization

The trained pipeline is stored using:

```python
joblib.dump()
```

Saved Model:

```text
fraud_model7.pkl
```

The saved object contains:

* feature engineering pipeline
* preprocessing logic
* trained Logistic Regression model

---

# Technology Stack

## Programming Language

* Python 3.13.6

---

## Core Libraries

| Library              | Purpose                                                |
| -------------------  | ------------------------------------------------------ |
| pandas               | Data processing                                        |
| numpy                | Numerical operations                                   |
| datetime             | Manipulating dates and times                           |
| matplotlib           | Visualization                                          |
| seaborn              | Statistical plots                                      |
| scikit-learn         | Machine learning pipeline                              |
| xgboost / lightgbm   | Advanced gradient boosting                             |
| optuna               | Hyperparameter tuning                                  |
| streamlit            | Web application                                        |
| joblib               | Model serialization                                    |
| psycopg2             | Database connection                                    |
| cloudpickle          | Extended serialization for complex objects/functions   |
| os                   | Operating system interface and file management         |
---

# Deployment Requirements

## Required Packages

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

# Business Value Delivered

## Operational Benefits

* Faster fraud detection through automated transaction risk scoring
* Reduced manual fraud investigation workload
* Lower financial losses caused by fraudulent transactions
* Improved operational efficiency for fraud monitoring teams
* Faster response time for suspicious activities
* Scalable fraud detection process for high transaction volumes

---

## Financial Impact

The fraud detection system helps reduce direct and indirect financial risks by:

* Preventing chargeback losses
* Reducing fraudulent transaction approvals
* Minimizing operational investigation costs
* Lowering customer compensation expenses
* Reducing reputational damage caused by fraud incidents

Even small improvements in fraud recall can significantly reduce annual financial losses in high-volume e-commerce systems.

---

## Customer Experience Benefits

An effective fraud detection system improves customer trust and platform reliability.

Benefits include:

* Better account protection
* Safer online payment experience
* Faster fraud prevention response
* Reduced unauthorized transaction activity
* Improved trust in the e-commerce platform

Balancing fraud detection recall and precision also helps minimize false fraud alerts that may negatively impact legitimate customers.

---

## Risk Management Benefits

The system provides additional support for enterprise risk management by:

* Identifying high-risk transaction behavior patterns
* Monitoring suspicious customer activity
* Detecting abnormal transaction velocity
* Highlighting risky geographic or behavioral patterns
* Supporting fraud investigation prioritization

---

## Business Intelligence Value

The engineered fraud indicators provide useful analytical insights for business teams.

Examples include:

* Customer behavioral anomalies
* Transaction timing trends
* Fraud concentration patterns
* Risky payment methods
* High-risk customer segments

These insights can support strategic decision-making and fraud prevention policy improvements.

---

## Scalability Advantages

The machine learning pipeline was designed to support scalable deployment.

Advantages include:

* Automated preprocessing pipeline
* Reusable feature engineering logic
* Fast inference capability
* Easy integration into production systems
* Real-time prediction support

---

## Regulatory and Compliance Support

Fraud monitoring systems help organizations strengthen compliance and transaction security processes.

Potential compliance-related benefits include:

* Improved transaction monitoring
* Better fraud audit capabilities
* Enhanced financial risk tracking
* Support for internal security controls

---

## Long-Term Strategic Value

As more transaction data becomes available, the system can continuously improve through retraining and monitoring.

Long-term benefits include:

* Adaptive fraud detection improvement
* Better fraud pattern recognition
* Improved operational intelligence
* Stronger fraud prevention capabilities over time

---

## Overall Business Outcome

The solution provides a balance between:

* fraud detection performance,
* operational efficiency,
* customer experience,
* and business risk reduction.

By combining machine learning with behavioral fraud analytics, the system creates a scalable and intelligent fraud prevention framework suitable for modern e-commerce environments.

---

# Model Risks and Limitations

## Potential Limitations

| Risk              | Description                         |
| ----------------- | ----------------------------------- |
| Concept Drift     | Fraud patterns may change over time |
| False Positives   | Legitimate users may be flagged     |
| Data Drift        | Input data distributions may shift  |
| Adversarial Fraud | Fraudsters adapt behavior           |

---

# Monitoring Recommendations

## Recommended Monitoring Metrics

* prediction probability distribution
* fraud detection rate
* false positive rate
* recall degradation
* transaction distribution changes

---

# Retraining Recommendations

Retraining should be considered when:

* fraud behavior changes significantly
* recall decreases substantially
* data drift is detected
* new fraud attack patterns emerge

---

# Future Improvements

Potential enhancements:

* ensemble models
* anomaly detection systems
* graph-based fraud detection
* real-time streaming architecture
* explainable AI integration
* adaptive threshold optimization

---

# Security Considerations

## Sensitive Data Handling

The system processes transactional and behavioral data.

Recommendations:

* encrypt sensitive customer data
* anonymize personally identifiable information
* secure model artifacts

---

# Conclusion

This project delivers a complete machine learning fraud detection system for UAE e-commerce transactions.

The solution combines:

* advanced feature engineering
* behavioral fraud analytics
* machine learning classification
* business-oriented fraud interpretation
* interactive prediction dashboards

The system is designed to support both technical teams and business stakeholders by providing interpretable fraud risk analysis and operational fraud detection capabilities.

# Business Value Delivered

## Operational Benefits

- Faster fraud detection through automated transaction risk scoring
- Reduced manual fraud investigation workload
- Lower financial losses caused by fraudulent transactions
- Improved operational efficiency for fraud monitoring teams
- Faster response time for suspicious activities
- Scalable fraud detection process for high transaction volumes

---

## Financial Impact

The fraud detection system helps reduce direct and indirect financial risks by:

- Preventing chargeback losses
- Reducing fraudulent transaction approvals
- Minimizing operational investigation costs
- Lowering customer compensation expenses
- Reducing reputational damage caused by fraud incidents

Even small improvements in fraud recall can significantly reduce annual financial losses in high-volume e-commerce systems.

---

## Customer Experience Benefits

An effective fraud detection system improves customer trust and platform reliability.

Benefits include:

- Better account protection
- Safer online payment experience
- Faster fraud prevention response
- Reduced unauthorized transaction activity
- Improved trust in the e-commerce platform

Balancing fraud detection recall and precision also helps minimize false fraud alerts that may negatively impact legitimate customers.

---

## Risk Management Benefits

The system provides additional support for enterprise risk management by:

- Identifying high-risk transaction behavior patterns
- Monitoring suspicious customer activity
- Detecting abnormal transaction velocity
- Highlighting risky geographic or behavioral patterns
- Supporting fraud investigation prioritization

---

## Business Intelligence Value

The engineered fraud indicators provide useful analytical insights for business teams.

Examples include:

- Customer behavioral anomalies
- Transaction timing trends
- Fraud concentration patterns
- Risky payment methods
- High-risk customer segments

These insights can support strategic decision-making and fraud prevention policy improvements.

---

## Scalability Advantages

The machine learning pipeline was designed to support scalable deployment.

Advantages include:

- Automated preprocessing pipeline
- Reusable feature engineering logic
- Fast inference capability
- Easy integration into production systems
- Real-time prediction support

---

## Regulatory and Compliance Support

Fraud monitoring systems help organizations strengthen compliance and transaction security processes.

Potential compliance-related benefits include:

- Improved transaction monitoring
- Better fraud audit capabilities
- Enhanced financial risk tracking
- Support for internal security controls

---

## Long-Term Strategic Value

As more transaction data becomes available, the system can continuously improve through retraining and monitoring.

Long-term benefits include:

- Adaptive fraud detection improvement
- Better fraud pattern recognition
- Improved operational intelligence
- Stronger fraud prevention capabilities over time

---

## Overall Business Outcome

The solution provides a balance between:

- fraud detection performance
- operational efficiency
- customer experience
- business risk reduction

By combining machine learning with behavioral fraud analytics, the system creates a scalable and intelligent fraud prevention framework suitable for modern e-commerce environments.