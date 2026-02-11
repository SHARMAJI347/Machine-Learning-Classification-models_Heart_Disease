import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. Load Models and Evaluation Metrics ---

# Create a dictionary to store models
models = {
    "Logistic Regression": joblib.load('model/log_reg_model.pkl'),
    "Decision Tree": joblib.load('model/dt_model.pkl'),
    "K-Nearest Neighbor": joblib.load('model/knn_model.pkl'),
    "Naive Bayes": joblib.load('model/nb_model.pkl'),
    "Random Forest": joblib.load('model/rf_model.pkl'),
    "XGBoost": joblib.load('model/xgb_model.pkl')
}

# Load evaluation metrics
try:
    evaluation_metrics_df = pd.read_csv('model/evaluation_metrics.csv')
except FileNotFoundError:
    st.error("Evaluation metrics file not found. Please ensure 'evaluation_metrics.csv' is in the 'model' directory.")
    evaluation_metrics_df = pd.DataFrame() # Empty DataFrame if file not found

# Load X_test and y_test for showing evaluation reports for selected model
try:
    X_test_loaded = joblib.load('model/X_test.pkl')
    y_test_loaded = joblib.load('model/y_test.pkl')
except FileNotFoundError:
    st.error("X_test or y_test not found. Cannot display classification report/confusion matrix for selected model.")
    X_test_loaded = pd.DataFrame() # Empty DataFrame if file not found
    y_test_loaded = pd.Series() # Empty Series if file not found

st.set_page_config(layout="wide")
st.title('Heart Disease Prediction App')
st.write('Predict the presence of heart disease based on patient medical parameters.')

# --- 2. Model Selection ---

st.sidebar.header('Model Selection')
selected_model_name = st.sidebar.selectbox(
    'Choose a Classification Model',
    list(models.keys())
)
selected_model = models[selected_model_name]
st.sidebar.write(f'**Selected Model:** {selected_model_name}')

# --- 3. Individual Patient Prediction ---

st.header('Predict for a Single Patient')
st.write('Please input the patient\'s information below:')

# Input widgets for features
with st.expander("Input Patient Data"): # Use an expander for input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider('Age', min_value=20, max_value=80, value=50)
        sex = st.selectbox('Sex', options=[(0, 'Female'), (1, 'Male')], format_func=lambda x: x[1])
        cp = st.selectbox('Chest Pain Type (cp)', options=[0, 1, 2, 3])
        trestbps = st.slider('Resting Blood Pressure (trestbps)', min_value=90, max_value=200, value=120)
        chol = st.slider('Serum Cholestoral (chol)', min_value=100, max_value=600, value=200)
    with col2:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[(0, 'False'), (1, 'True')], format_func=lambda x: x[1])
        restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', options=[0, 1, 2])
        thalach = st.slider('Maximum Heart Rate Achieved (thalach)', min_value=70, max_value=220, value=150)
        exang = st.selectbox('Exercise Induced Angina (exang)', options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1])
        oldpeak = st.slider('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    with col3:
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

# Convert input_data to DataFrame for prediction
single_input_df = pd.DataFrame([input_data])

if st.button(f'Predict Heart Disease for this Patient'):
    prediction = selected_model.predict(single_input_df)[0]
    prediction_proba = selected_model.predict_proba(single_input_df)[:, 1][0]

    st.subheader('Single Patient Prediction Result:')
    if prediction == 1:
        st.error(f'**Result: Positive for Heart Disease** (Probability: {prediction_proba:.2f})')
    else:
        st.success(f'**Result: Negative for Heart Disease** (Probability: {prediction_proba:.2f})')

# --- 4. Batch Prediction (CSV Upload) ---

st.header('Batch Prediction (Upload CSV)')
st.write('Upload a CSV file containing multiple patient records for prediction.')
st.info('Note: For Streamlit free tier, please keep the CSV file size small and ensure it contains the same columns as the training data except for the target column.')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        batch_input_df = pd.read_csv(uploaded_file)
        st.write('Uploaded Data Head:')
        st.dataframe(batch_input_df.head())

        # Check if 'target' column exists in uploaded data, and drop it if so (for prediction)
        if 'target' in batch_input_df.columns:
            # If target column is present, we can compute metrics
            st.success("Target column found in uploaded CSV. Classification Report and Confusion Matrix will be generated.")
            y_true_batch = batch_input_df['target']
            X_batch_for_pred = batch_input_df.drop(columns=['target'])
        else:
            st.warning("No 'target' column found in uploaded CSV. Only predictions will be made.")
            X_batch_for_pred = batch_input_df.copy()

        if st.button(f'Predict on Uploaded Data with {selected_model_name}'):
            if X_batch_for_pred.empty:
                st.warning("Uploaded data is empty or contains no features for prediction.")
            else:
                try:
                    batch_predictions = selected_model.predict(X_batch_for_pred)
                    batch_predictions_proba = selected_model.predict_proba(X_batch_for_pred)[:, 1]

                    prediction_results_df = batch_input_df.copy()
                    prediction_results_df['Predicted_Target'] = batch_predictions
                    prediction_results_df['Prediction_Probability'] = batch_predictions_proba.round(2)
                    prediction_results_df['Predicted_Diagnosis'] = prediction_results_df['Predicted_Target'].apply(lambda x: 'Positive' if x == 1 else 'Negative')

                    st.subheader('Batch Prediction Results:')
                    st.dataframe(prediction_results_df[['age', 'sex', 'cp', 'trestbps', 'chol', 'Predicted_Diagnosis', 'Prediction_Probability']])

                    csv = prediction_results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Prediction Results as CSV",
                        data=csv,
                        file_name=f'heart_disease_predictions_{selected_model_name.replace(" ", "_")}.csv',
                        mime='text/csv',
                    )

                    # Display Classification Report and Confusion Matrix if target is present
                    if 'target' in batch_input_df.columns:
                        st.subheader('Batch Evaluation Metrics (for uploaded data):')
                        st.text('Classification Report:')
                        st.code(classification_report(y_true_batch, batch_predictions))
                        st.text('Confusion Matrix:')
                        st.code(confusion_matrix(y_true_batch, batch_predictions))

                except Exception as pred_e:
                    st.error(f"Error during prediction: {pred_e}. Please ensure the uploaded CSV format matches the expected features.")

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

# --- 5. Display Evaluation Metrics ---

st.header('Model Evaluation Metrics')
st.write('Performance of all trained models on the test dataset:')
if not evaluation_metrics_df.empty:
    st.dataframe(evaluation_metrics_df.round(4))

    # Display Classification Report and Confusion Matrix for the selected model on original X_test
    if not X_test_loaded.empty and not y_test_loaded.empty:
        st.subheader(f'Detailed Evaluation for {selected_model_name} on Test Data:')
        y_pred_selected_model = selected_model.predict(X_test_loaded)

        st.text('Classification Report:')
        st.code(classification_report(y_test_loaded, y_pred_selected_model))

        st.text('Confusion Matrix:')
        st.code(confusion_matrix(y_test_loaded, y_pred_selected_model))

else:
    st.info('Evaluation metrics could not be loaded.')

st.subheader('How to Run the App Locally:')
st.write('1. Make sure you have Streamlit installed (`pip install streamlit`).')
st.write('2. Save the `streamlit_app.py` file to your local machine.')
st.write('3. Open your terminal or command prompt, navigate to the directory where you saved the file, and run the following command:')
st.code('streamlit run streamlit_app.py')
st.write('4. This will open the Streamlit application in your web browser.')
