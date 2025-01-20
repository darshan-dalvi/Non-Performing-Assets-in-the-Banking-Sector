
# Non-Performing Assets (NPA) Analysis and Prediction

This project focuses on analyzing and predicting Non-Performing Assets (NPAs) in the banking sector using a synthetic dataset of 300,221 loan records. By leveraging machine learning techniques, the best model, **XGBoost**, achieved a **99% accuracy** in predicting NPAs.

## Dataset

The dataset is available on Kaggle: [Non-Performing Assets (NPA) Dataset](https://www.kaggle.com/datasets/darshandalvi12/non-performing-assets-npa). It includes features like loan amount, credit score, repayment history, and more, to simulate real-world banking scenarios.

### Features:

1. **Loan_ID**: Unique identifier for each loan.
2. **Customer_ID**: Unique identifier for each customer.
3. **Loan_Amount**: Amount f the loan.
4. **Loan_Type**: Type of loan (e.g., Home, Personal, Business).
5. **Credit_Score**: Customer’s credit score (300–850).
6. **Repayment_History**: Percentage of on-time payments.
7. **Collateral_Value**: Value of the collateral.
8. **Loan_Tenure**: Loan repayment duration in months.
9. **Default_Status**: Target variable (0 = Performing, 1 = Non-Performing).

## Project Overview

This project includes the following steps:

1. **Data Cleaning and Preprocessing**:

   - Performed in the `cleaning.ipynb` notebook.
   - Handled missing values, encoded categorical variables, and scaled numerical features.
2. **Model Training**:

   - Various models were tested, including Logistic Regression, Random Forest, and Gradient Boosting.
   - **XGBoost** delivered the best performance with a **99% accuracy**.
3. **Evaluation**:

   - Models were evaluated using metrics like Accuracy, Precision, Recall, and F1-Score.
   - The model was fine-tuned using hyperparameter optimization.
4. **Deployment**:

   - The trained model can predict loan performance based on input features.

## Instructions to Run the Project

### Prerequisites

1. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn xgboost
   ```

### Download the dataset from Kaggle:

[Non-Performing Assets (NPA) Dataset](https://www.kaggle.com/datasets/darshandalvi12/non-performing-assets-npa).

# Steps to Execute

1. Clone this repository and navigate to the project directory.
2. Open the `cleaning.ipynb` file in Jupyter Notebook or any compatible IDE.
3. Follow the instructions in the notebook to create the following datasets:
   * **X_train** : Training feature set.
   * **X_test** : Testing feature set.
   * **y_train** : Training target variable.
   * **y_test** : Testing target variable.
4. Train the model using the provided scripts or notebooks.

### Key Files

* `cleaning.ipynb`: Notebook for data cleaning and preprocessing.
* `model_training.ipynb`: Notebook for model training and evaluation.
* `final_model.pkl`: Saved model file for deployment.

## Results

* **Best Model** : XGBoost
* **Accuracy** : 99%
* **Precision** : High precision in identifying NPAs.
* **Recall** : Excellent recall for NPA detection.
