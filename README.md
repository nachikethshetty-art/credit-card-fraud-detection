# 💳 Credit Card Fraud Detection System

## 🔍 Project Overview
This project presents an end-to-end machine learning pipeline for detecting fraudulent credit card transactions from highly imbalanced financial data. The objective is to accurately identify rare fraud events while minimizing false alarms that negatively impact customer experience. The solution emphasizes rigorous evaluation, business-aware threshold tuning, model explainability, and production-ready batch inference.

---

## 📊 Dataset
The model is trained on the Credit Card Fraud Detection dataset containing **284,807 transactions** with approximately **0.17% fraud cases**, making it an extremely imbalanced classification problem.

**Key characteristics:**
- PCA-transformed features: `V1–V28`
- Additional features: `Time`, `Amount`
- Target variable: `Class` (1 = Fraud, 0 = Legitimate)
- No missing values
- Severe class imbalance

---

## ⚙️ Modeling Approach

### 1️⃣ Data Preparation
- Performed exploratory data analysis (EDA)
- Used stratified train-test split to preserve class distribution
- Established Logistic Regression baseline

### 2️⃣ Imbalance Handling
Due to extreme skew (~0.17% fraud), **SMOTE** was applied **only on the training set** to avoid data leakage.

### 3️⃣ Model Training & Comparison
Models evaluated:
- Logistic Regression (baseline)
- Random Forest ⭐ (selected)
- XGBoost

Random Forest provided the best recall–precision balance for fraud detection.

### 4️⃣ Threshold Optimization
Instead of using the default 0.5 threshold, tuning was performed to optimize business trade-offs.

**Final operating threshold:** `0.3`

This improved fraud recall significantly while maintaining acceptable precision.

---

## 📈 Key Results

| Metric | Baseline | Final Model |
|--------|----------|------------|
| Fraud Recall | ~0.68 | **~0.93** |
| ROC-AUC | — | **~0.96+** |
| False Negatives | — | **12** |
| False Positives | — | **40** |

**Business interpretation:**
- High fraud capture rate  
- Very low missed fraud cases  
- Controlled customer friction  

Precision–Recall analysis was emphasized due to extreme class imbalance.

---

## 🧠 Model Explainability (SHAP)
To improve model transparency, **SHAP (SHapley Additive exPlanations)** was applied.

### 🔹 Global Insights
Top fraud-driving components:

- **V14**
- **V4**
- **V12**

These PCA components capture latent transaction patterns strongly associated with fraudulent behavior.

### 🔹 Local Explanation
Waterfall analysis was used to interpret individual high-risk transactions, demonstrating how feature contributions push predictions toward fraud.

---

## 🚀 Batch Inference Pipeline
To simulate production deployment, the trained model was serialized using **joblib** and integrated into a reusable batch scoring pipeline.

**Pipeline capabilities:**
- Loads trained model  
- Reads new transactions from CSV  
- Generates fraud probabilities  
- Applies optimized threshold  
- Outputs scored results to new CSV  

This demonstrates real-world ML lifecycle readiness beyond notebook experimentation.

---

## 📸 Key Visualizations

### Confusion Matrix
![Confusion Matrix](reports/confusion_matrix)

### Precision–Recall Curve
![PR Curve](reports/pr_curve)

### SHAP Summary (Beeswarm)
![SHAP Summary](reports/shap_summary)

---

## ▶️ How to Run

### 1️⃣ Clone repository
```bash
git clone <https://github.com/nachikethshetty-art/credit-card-fraud-detection>
cd fraud-detection