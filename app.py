"""
Streamlit Placement Prediction Dashboard
Files included (this single file contains the whole app using tabs):
- app (this file)

This version is updated to load the machine learning Pipeline,
which includes Polynomial Features and Standardization, ensuring
correct prediction when only CGPA and IQ are provided.

Presented by: Yash Rawat
Enrollment: SS/BCA/2301/293
PRF No.: Niu-23-17281

To run:
1. pip install -r requirements.txt
2. streamlit run streamlit_placement_dashboard.py

Requirements (requirements.txt):
streamlit
pandas
numpy
scikit-learn
matplotlib
plotly
seaborn

"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Use the paths derived from your uploaded files
# Note: Since your model file was named 'modle.pkl' and the prompt used 'model.pkl', 
# I will use 'modle.pkl' to match your uploaded file and ensure compatibility.
MODEL_PATH = "model.pkl" 
DATA_PATH = "placement.csv"

# -------------------- Utility functions --------------------
@st.cache_data
def load_data(path=DATA_PATH):
    """Loads the dataset and ensures clean column names."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: Dataset not found at {path}. Please check file location.")
        return pd.DataFrame() # Return empty DataFrame on failure

    # Drop unnamed index if exists
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Normalize column names: strip whitespace and convert to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Ensure expected names (note: 'placement' is used in your CSV, so we check for it)
    if 'cgpa' not in df.columns or 'iq' not in df.columns or 'placement' not in df.columns:
        st.warning('Dataset does not have expected columns (cgpa, iq, placement).')
    
    return df

@st.cache_resource
def load_model(path=MODEL_PATH):
    """Loads the machine learning pipeline (model + scaler + poly features)."""
    try:
        with open(path, 'rb') as f:
            # We assume this loaded object is the full scikit-learn Pipeline
            model_pipeline = pickle.load(f) 
        return model_pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. Please check file location.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# CRITICAL FIX: The prediction function is simplified to rely on the loaded Pipeline.
# If you updated your notebook correctly, the Pipeline handles the PolynomialFeatures 
# and StandardScaler steps internally, using the raw CGPA and IQ input.
def predict(model_pipeline, cgpa, iq):
    """
    Predicts using the loaded pipeline. The pipeline handles the necessary 
    PolynomialFeatures and StandardScaler transformations internally.
    """
    # CRITICAL: Input must be a 2D array with the two raw features: [[CGPA, IQ]]
    # This matches the input format of the original training data columns.
    X_raw = np.array([[cgpa, iq]])
    
    # Predict the class (0 or 1)
    pred = model_pipeline.predict(X_raw)[0]
    
    # Predict the probability for each class [Prob_Not_Placed, Prob_Placed]
    proba = model_pipeline.predict_proba(X_raw)[0]
    
    # The probability of the positive class (Placed, class 1)
    prob_positive = float(proba[1])
    
    return int(pred), prob_positive

# -------------------- App layout --------------------
st.set_page_config(page_title='Placement Predictor — Yash Rawat', layout='wide')

st.title('Placement Prediction Dashboard')
st.caption('Professional dashboard for predicting student placement from CGPA and IQ')

# Load data and model once for the whole app
df = load_data()
model_pipeline = load_model()

# Top Tabs (user chose Top Tabs)
tabs = st.tabs(["Home", "Make a Prediction", "Dataset Explorer", "Visualizations", "Model Performance", "About"])

# -------------------- Home --------------------
with tabs[0]:
    st.header('Overview')
    st.markdown(
        """
        **Problem statement:** Predict whether a student will be placed (1) or not placed (0) based on two features: CGPA and IQ.

        **Model:** **Enhanced Logistic Regression Pipeline** (includes Polynomial Features and Standardization for non-linear fitting) trained on the provided dataset.

        **Data:** Based on the uploaded `placement.csv`.
        """
    )
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Dataset snapshot')
        if not df.empty:
            st.dataframe(df.head(10))
    with c2:
        st.subheader('Basic statistics')
        if not df.empty:
            st.table(df[['cgpa','iq','placement']].describe())

# -------------------- Prediction --------------------
with tabs[1]:
    st.header('Make a Prediction')
    
    if model_pipeline is None:
        st.error("Cannot make predictions. Model failed to load.")
    else:
        left, right = st.columns([1,1])
        with left:
            # Use a slightly higher max value for CGPA if your data has high values
            cgpa = st.number_input('CGPA', min_value=0.0, max_value=10.0, value=6.5, step=0.1, format="%.2f")
        with right:
            iq = st.number_input('IQ', min_value=30.0, max_value=250.0, value=120.0, step=1.0)

        if st.button('Predict Placement'):
            pred, prob = predict(model_pipeline, cgpa, iq)
            label = 'Placed' if pred==1 else 'Not Placed'
            
            st.metric('Prediction', f'{label} (class {pred})', delta=f'Probability: {prob:.2f}')

            st.markdown('**Interpretation**')
            if prob >= 0.75:
                st.success(f'The model is confident ({prob:.0%}) that the student will be placed.')
            elif prob >= 0.5:
                st.info(f'The model leans towards placement ({prob:.0%}) but not strongly confident.')
            else:
                st.warning(f'The model predicts not placed ({(1-prob):.0%} confidence on negative).')

            # Show where the point lies on feature scatter
            if not df.empty:
                fig = px.scatter(df, x='cgpa', y='iq', color=df['placement'].astype(str), 
                                 title='CGPA vs IQ (colored by historical placement)')
                # Add the input point marker
                fig.add_scatter(x=[cgpa], y=[iq], mode='markers', name='Input Student', 
                                marker=dict(size=15, symbol='star', color='red'))
                st.plotly_chart(fig, use_container_width=True)

# -------------------- Dataset Explorer --------------------
with tabs[2]:
    st.header('Dataset Explorer')
    st.subheader('Full dataset')
    if not df.empty:
        st.dataframe(df)

    st.subheader('Upload your own CSV (optional)')
    uploaded = st.file_uploader('Upload CSV with columns cgpa, iq', type=['csv'])
    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded)
            st.success('Uploaded file loaded')
            st.dataframe(user_df.head())
        except Exception as e:
            st.error('Could not read uploaded file: ' + str(e))

# -------------------- Visualizations --------------------
with tabs[3]:
    st.header('Visualizations')
    if not df.empty:
        st.subheader('Scatter: CGPA vs IQ')
        fig = px.scatter(df, x='cgpa', y='iq', color=df['placement'].astype(str), 
                         title="Placement Status by Features")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Distributions')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('CGPA distribution')
            fig2 = px.histogram(df, x='cgpa', nbins=20)
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            st.markdown('IQ distribution')
            fig3 = px.histogram(df, x='iq', nbins=20)
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader('Correlation heatmap')
        corr = df[['cgpa','iq','placement']].corr()
        fig4, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig4)
    else:
        st.warning("Cannot display visualizations. Data failed to load.")

# -------------------- Model Performance --------------------
with tabs[4]:
    st.header('Model Performance')
    
    if df.empty or model_pipeline is None:
        st.error("Cannot calculate performance metrics. Data or Model failed to load.")
    else:
        # Prepare X and y using ONLY the two features the model expects
        X = df[['cgpa','iq']].values
        y = df['placement'].values

        # Predictions using the full pipeline
        y_pred = model_pipeline.predict(X)
        
        # Use predict_proba from the pipeline
        if hasattr(model_pipeline, 'predict_proba'):
             y_scores = model_pipeline.predict_proba(X)[:,1]
        else:
             y_scores = None # Should not happen with Pipeline containing LogReg

        # Calculate metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Accuracy', f'{acc:.3f}')
        c2.metric('Precision', f'{prec:.3f}')
        c3.metric('Recall', f'{rec:.3f}')
        c4.metric('F1 Score', f'{f1:.3f}')

        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y, y_pred)
        fig5, ax5 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax5)
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')
        ax5.set_title("Confusion Matrix")
        st.pyplot(fig5) 

        if y_scores is not None:
            fpr, tpr, _ = roc_curve(y, y_scores)
            roc_auc = auc(fpr, tpr)
            st.subheader('ROC Curve')
            fig6, ax6 = plt.subplots()
            ax6.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
            ax6.plot([0, 1], [0, 1], 'r--')
            ax6.set_xlabel('False Positive Rate')
            ax6.set_ylabel('True Positive Rate')
            ax6.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax6.legend(loc="lower right")
            st.pyplot(fig6) 

        st.markdown('**Short analysis:**')
        st.write('The model is an enhanced Logistic Regression Pipeline. The metrics above reflect its performance. The **F1 Score** and **ROC AUC** are the best indicators of its effectiveness on this binary classification task, showing its ability to separate placed from non-placed students.')

# -------------------- About --------------------
with tabs[5]:
    st.header('About this Project')
    st.markdown(f"""
    **Project:** Student Placement Predictor (Enhanced Model)

    **Presented by:** Yash Rawat  
    **Enrollment:** SS/BCA/2301/293  
    **PRF No.:** Niu-23-17281

    **Model used:** **scikit-learn Pipeline** (`PolynomialFeatures` + `StandardScaler` + `LogisticRegression`)
    **Reason for Pipeline:** To capture non-linear relationships between CGPA and IQ while adhering to the two-feature constraint.

    **Files used:** `{DATA_PATH}`, `{MODEL_PATH}`

    **How to run:**
    1. Ensure the model file (`modle.pkl` - which must be the Pipeline) and data file (`placement.csv`) exist in the same directory.  
    2. Install requirements: `pip install -r requirements.txt`  
    3. Run: `streamlit run streamlit_placement_dashboard.py`
    """)

    st.markdown('---')
    st.markdown('**Next Steps / Possible Enhancements:**')
    st.write('- If performance is still poor, consider trying the **Random Forest** algorithm within the Pipeline.')
    st.write('- Create a downloadable PPT summary for your viva.')
    st.write('- Add export (CSV) of all predictions.')

# -------------------- End --------------------

if __name__ == '__main__':
    pass