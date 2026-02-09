# Heart Disease Prediction Project

## a. Problem Statement

This project aims to develop and evaluate various machine learning classification models to predict the presence of heart disease in patients based on a set of medical parameters. Early and accurate prediction can assist in timely diagnosis and intervention.

## b. Dataset Description

The dataset used in this project is the 'Heart Disease UCI' dataset, sourced from Kaggle (johnsmith88/heart-disease-dataset). It contains 14 attributes related to heart disease diagnosis. The target variable indicates the presence (1) or absence (0) of heart disease. The dataset has 1025 instances and 14 features.

## c. Models Used and Performance Comparison

We implemented and evaluated six different machine learning classification models on the preprocessed heart disease dataset. The performance of each model was assessed using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC Score).

### Comparison Table of Evaluation Metrics:

| ML Model Name         | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|-----------------------|----------|-----------|-----------|--------|----------|-----------|
| Logistic Regression   | 0.8732   | 0.9441    | 0.8559    | 0.9048 | 0.8796   | 0.7471    |
| Decision Tree         | 0.5100   | 0.5080    | 0.5167    | 0.6078 | 0.5586   | 0.0163    |
| K-Nearest Neighbor    | 0.5000   | 0.4938    | 0.5111    | 0.4510 | 0.4792   | 0.0020    |
| Naive Bayes           | 0.5000   | 0.4418    | 0.5082    | 0.6078 | 0.5536   | -0.0045   |
| Random Forest         | 0.4900   | 0.5154    | 0.5000    | 0.5882 | 0.5405   | -0.0245   |
| XGBoost               | 0.5700   | 0.5838    | 0.5741    | 0.6078 | 0.5905   | 0.1389    |

### Observations on Model Performance:

Based on the comparison table, here are some observations regarding the performance of each model on the chosen dataset:

*   **XGBoost:** This model shows the highest Accuracy, AUC Score, Precision, F1 Score, and MCC Score, making it the top-performing model in this evaluation. This suggests that its ensemble nature and boosting capabilities are well-suited for the dataset.

*   **Logistic Regression:** Performs reasonably well, especially considering its simplicity. It has a decent Recall, but its AUC Score is among the lowest, indicating it struggles with distinguishing between classes across various thresholds.

*   **Decision Tree Classifier:** The Decision Tree performs close to Logistic Regression in Accuracy and F1 Score. Its AUC Score is better than Logistic Regression but still not as high as XGBoost.

*   **Random Forest:** As an ensemble method, Random Forest's performance is surprisingly lower than XGBoost and even some simpler models like Logistic Regression and Decision Tree on several metrics. This could indicate that the dataset might not benefit significantly from bagging or that the default hyperparameters are not optimal.

*   **K-Nearest Neighbor:** This model shows the lowest Accuracy, AUC Score, Precision, F1 Score, and MCC Score among all models. Its performance is very close to random guessing for a binary classification problem (Accuracy of 0.5), suggesting it struggles to find meaningful patterns in the data based on feature proximity.

*   **Naive Bayes (GaussianNB):** Similar to KNN, Naive Bayes performs poorly across all metrics. Its AUC Score and MCC Score are particularly low, suggesting its assumption of feature independence might not hold for this dataset, or the data distribution does not align with Gaussian assumptions after preprocessing.

**General Insights:**

*   Ensemble methods (XGBoost) generally tend to perform better due to their ability to combine multiple weaker learners, but this is not always the case (as seen with Random Forest). XGBoost's boosting approach appears to be particularly effective here.
*   The low performance of K-Nearest Neighbor and Naive Bayes could indicate that the dataset has complex decision boundaries or that the features, even after preprocessing, do not lend themselves well to these specific algorithms.
*   Further hyperparameter tuning for all models, especially Random Forest, might improve their performance.

## Step 4: `requirements.txt`

The `requirements.txt` file lists all the Python dependencies required to run this project. It ensures that the environment is correctly set up for deployment and local execution.

```
streamlit
pandas
scikit-learn
joblib
numpy
xgboost
matplotlib
kagglehub
```

## Step 6: Streamlit Deployment

This project includes a Streamlit application (`streamlit_app.py`) for interactive predictions. To deploy this app on Streamlit Community Cloud, follow these steps:

1.  Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
2.  Sign in using your GitHub account.
3.  Click “New App” on the Streamlit Cloud dashboard.
4.  Select your GitHub repository where this project is hosted.
5.  Choose the branch (usually `main` or `master`).
6.  Specify the main file path for your Streamlit app (e.g., `streamlit_app.py`).
7.  Click the “Deploy!” button.

Your Streamlit app will then be deployed and accessible via a public URL.
