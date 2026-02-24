# 🎯 Campus Placement Prediction using Machine Learning

An end-to-end Machine Learning project that predicts student placement outcomes using academic and aptitude indicators.  
The project includes an interactive **Streamlit dashboard** for visualization and real-time prediction.

---

## 🚀 Overview

Campus placements are important indicators of student readiness and institutional performance.  
This project uses Machine Learning classification algorithms to predict whether a student is likely to be placed based on:

- CGPA
- IQ Score

Multiple ML models were evaluated, and the best-performing model was deployed through a user-friendly dashboard.

---

## 🧠 Problem Statement

Institutions often lack early predictive insights into placement readiness.  
This system helps analyze academic data and provides placement predictions using data-driven techniques.

---

## ⚙️ Tech Stack

- **Python**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Streamlit**
- **Matplotlib**
- **Pickle**

---

## 📊 Machine Learning Pipeline

1. Data Collection from dataset (`placement.csv`)
2. Data Cleaning & Preprocessing
3. Feature Scaling using `StandardScaler`
4. Train-Test Split
5. Model Training & Comparison

### Models Evaluated
- Logistic Regression (~82%)
- Support Vector Machine (~84%)
- Decision Tree (~86%)
- ✅ K-Nearest Neighbors (~90%) *(Selected Model)*

The trained model is saved using **Pickle** and integrated into the Streamlit dashboard.

---

## 📈 Dashboard Features

- Real-time placement prediction
- Interactive CGPA & IQ input sliders
- Dataset visualization
- Accuracy & confusion matrix display
- Decision boundary visualization
- Model performance comparison

---


