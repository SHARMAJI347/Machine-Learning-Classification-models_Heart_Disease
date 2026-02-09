# This file will host the Streamlit application.
import joblib

# Load the models
log_reg_model = joblib.load('model/log_reg_model.pkl')
dt_model = joblib.load('model/dt_model.pkl')
knn_model = joblib.load('model/knn_model.pkl')
nb_model = joblib.load('model/nb_model.pkl')
rf_model = joblib.load('model/rf_model.pkl')
xgb_model = joblib.load('model/xgb_model.pkl')

print("All models loaded successfully.")

import streamlit as st

st.title('Heart Disease Prediction App')
st.write('Please input the patient\'s information below:')

# Input widgets for features
age = st.slider('Age', min_value=20, max_value=80, value=50)
sex = st.selectbox('Sex', options=[(0, 'Female'), (1, 'Male')], format_func=lambda x: x[1])
cp = st.selectbox('Chest Pain Type (cp)', options=[0, 1, 2, 3])
trestbps = st.slider('Resting Blood Pressure (trestbps)', min_value=90, max_value=200, value=120)
chol = st.slider('Serum Cholestoral (chol)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[(0, 'False'), (1, 'True')], format_func=lambda x: x[1])
restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', options=[0, 1, 2])
thalach = st.slider('Maximum Heart Rate Achieved (thalach)', min_value=70, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1])
oldpeak = st.slider('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', options=[0, 1, 2])
ca = st.selectbox('Number of Major Vessels (ca)', options=[0, 1, 2, 3])
thal = st.selectbox('Thalassemia (thal)', options=[(0, 'Normal'), (1, 'Fixed Defect'), (2, 'Reversible Defect')], format_func=lambda x: x[1])

# Convert selectbox outputs to raw values for prediction
sex_val = sex[0]
fbs_val = fbs[0]
exang_val = exang[0]
thal_val = thal[0]

# Create a dictionary for the input features
input_data = {
    'age': age,
    'sex': sex_val,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs_val,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang_val,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal_val
}

st.write('Input Data:')
st.write(input_data)
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Re-create preprocessing steps for consistent data transformation ---
# Define features and target (re-defining for Streamlit app context)
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Convert input_data to DataFrame for preprocessing
input_df = pd.DataFrame([input_data])

# Make predictions
if st.button('Predict'):
    # The loaded models are pipelines and should handle preprocessing internally.
    # No need to manually preprocess with 'preprocessor' before passing to model.predict().

    # For pipelines, the preprocessing is done automatically when calling .predict()
    log_reg_pred = log_reg_model.predict(input_df)[0]
    dt_pred = dt_model.predict(input_df)[0]
    knn_pred = knn_model.predict(input_df)[0]
    nb_pred = nb_model.predict(input_df)[0]
    rf_pred = rf_model.predict(input_df)[0]
    xgb_pred = xgb_model.predict(input_df)[0]

    st.subheader('Prediction Results:')
    st.write(f'Logistic Regression Prediction: {'Positive' if log_reg_pred == 1 else 'Negative'}')
    st.write(f'Decision Tree Prediction: {'Positive' if dt_pred == 1 else 'Negative'}')
    st.write(f'K-Nearest Neighbor Prediction: {'Positive' if knn_pred == 1 else 'Negative'}')
    st.write(f'Naive Bayes Prediction: {'Positive' if nb_pred == 1 else 'Negative'}')
    st.write(f'Random Forest Prediction: {'Positive' if rf_pred == 1 else 'Negative'}')
    st.write(f'XGBoost Prediction: {'Positive' if xgb_pred == 1 else 'Negative'}')

st.subheader('How to Run the App Locally:')
st.write('1. Make sure you have Streamlit installed (`pip install streamlit`).')
st.write('2. Save the `streamlit_app.py` file to your local machine.')
st.write('3. Open your terminal or command prompt, navigate to the directory where you saved the file, and run the following command:')
st.code('streamlit run streamlit_app.py')
st.write('4. This will open the Streamlit application in your web browser.')

    # Store predictions in a DataFrame
    prediction_results = pd.DataFrame({
        'Model': [
            'Logistic Regression',
            'Decision Tree',
            'K-Nearest Neighbor',
            'Naive Bayes',
            'Random Forest',
            'XGBoost'
        ],
        'Prediction': [
            'Positive' if log_reg_pred == 1 else 'Negative',
            'Positive' if dt_pred == 1 else 'Negative',
            'Positive' if knn_pred == 1 else 'Negative',
            'Positive' if nb_pred == 1 else 'Negative',
            'Positive' if rf_pred == 1 else 'Negative',
            'Positive' if xgb_pred == 1 else 'Negative'
        ]
    })

    st.subheader('Prediction Results:')
    st.dataframe(prediction_results)
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Re-create preprocessing steps for consistent data transformation ---
# Define features and target (re-defining for Streamlit app context)
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Convert input_data to DataFrame for preprocessing
input_df = pd.DataFrame([input_data])

# Make predictions
if st.button('Predict'):
    # The loaded models are pipelines and should handle preprocessing internally.
    # No need to manually preprocess with 'preprocessor' before passing to model.predict().

    # For pipelines, the preprocessing is done automatically when calling .predict()
    log_reg_pred = log_reg_model.predict(input_df)[0]
    dt_pred = dt_model.predict(input_df)[0]
    knn_pred = knn_model.predict(input_df)[0]
    nb_pred = nb_model.predict(input_df)[0]
    rf_pred = rf_model.predict(input_df)[0]
    xgb_pred = xgb_model.predict(input_df)[0]

    st.subheader('Prediction Results:')
    st.write(f'Logistic Regression Prediction: {'Positive' if log_reg_pred == 1 else 'Negative'}')
    st.write(f'Decision Tree Prediction: {'Positive' if dt_pred == 1 else 'Negative'}')
    st.write(f'K-Nearest Neighbor Prediction: {'Positive' if knn_pred == 1 else 'Negative'}')
    st.write(f'Naive Bayes Prediction: {'Positive' if nb_pred == 1 else 'Negative'}')
    st.write(f'Random Forest Prediction: {'Positive' if rf_pred == 1 else 'Negative'}')
    st.write(f'XGBoost Prediction: {'Positive' if xgb_pred == 1 else 'Negative'}')

    # Store predictions in a DataFrame
    prediction_results = pd.DataFrame({
        'Model': [
            'Logistic Regression',
            'Decision Tree',
            'K-Nearest Neighbor',
            'Naive Bayes',
            'Random Forest',
            'XGBoost'
        ],
        'Prediction': [
            'Positive' if log_reg_pred == 1 else 'Negative',
            'Positive' if dt_pred == 1 else 'Negative',
            'Positive' if knn_pred == 1 else 'Negative',
            'Positive' if nb_pred == 1 else 'Negative',
            'Positive' if rf_pred == 1 else 'Negative',
            'Positive' if xgb_pred == 1 else 'Negative'
        ]
    })

    st.subheader('Prediction Results:')
    st.dataframe(prediction_results)

st.subheader('How to Run the App Locally:')
st.write('1. Make sure you have Streamlit installed (`pip install streamlit`).')
st.write('2. Save the `streamlit_app.py` file to your local machine.')
st.write('3. Open your terminal or command prompt, navigate to the directory where you saved the file, and run the following command:')
st.code('streamlit run streamlit_app.py')
st.write('4. This will open the Streamlit application in your web browser.')
