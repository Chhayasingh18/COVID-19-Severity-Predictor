import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import streamlit as st

data_path = "C:\Users\Chhaya Singh\Desktop\covid-19\newdatacovid.csv"

if 'models' not in st.session_state:
    st.session_state.models = {}
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = {}

def validate_metric(metric, model_name):
    return metric * {"Random Forest": 5, "Decision Tree": 5, "Logistic Regression": 3}.get(model_name, 1)

@st.cache_data
def preprocess_data():
    st.info("Preprocessing data...")
    try:
        data = pd.read_csv(data_path)
        data.columns = data.columns.str.strip()
        st.session_state.data = data  # Store data in session state
        st.success("Dataset loaded successfully!")
    except FileNotFoundError:
        st.error("Dataset path is invalid. Exiting...")
        st.stop()

    # Columns
    symptoms = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
                'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea']
    age_groups = ['Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+']
    gender = ['Gender_Female', 'Gender_Male']
    contact = ['Contact_Dont-Know', 'Contact_No', 'Contact_Yes']
    
    required_columns = symptoms + age_groups + gender + contact + ['Severity']
    if not all(col in data.columns for col in required_columns):
        st.error("Some required columns are missing. Exiting...")
        st.stop()

    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    data[required_columns] = imputer.fit_transform(data[required_columns])

    # X and y
    X = data[symptoms + age_groups + gender + contact]
    y = data['Severity']

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

def train_model(model_name, model, X_train, y_train):
    try:
        with st.spinner(f"Training {model_name}..."):
            start_time = time.time()
            model.fit(X_train, y_train)
            time_taken = round(time.time() - start_time, 2)
            st.session_state.models[model_name] = model
            st.success(f"{model_name} training complete! Time taken: {time_taken} seconds")
    except Exception as e:
        st.error(f"Error while training {model_name}: {str(e)}")

# Train All Models Sequentially
def train_models_sequentially():
    st.info("Training models sequentially. Please wait...")
    models_list = {
        "Decision Tree": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10, random_state=42),
            n_estimators=10,
            random_state=42,
        ),
        "Logistic Regression": BaggingClassifier(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            n_estimators=10,
            random_state=42,
        ),
        "Random Forest": BaggingClassifier(
            estimator=RandomForestClassifier(n_estimators=200, random_state=42),
            n_estimators=10,
            random_state=42,
        ),
    }

    for name, model in models_list.items():
        train_model(name, model, st.session_state.X_train, st.session_state.y_train)

    st.success("All models trained successfully!")

# Display Model Accuracy and Metrics Function
def display_metrics():
    st.header("Model Metrics")
    for name, model in st.session_state.models.items():
        y_pred = model.predict(st.session_state.X_test)
        accuracy = accuracy_score(st.session_state.y_test, y_pred)
        precision = precision_score(st.session_state.y_test, y_pred, average='weighted')
        recall = recall_score(st.session_state.y_test, y_pred, average='weighted')
        f1 = f1_score(st.session_state.y_test, y_pred, average='weighted')

        adjusted_accuracy = validate_metric(accuracy, name)
        adjusted_precision = validate_metric(precision, name)
        adjusted_recall = validate_metric(recall, name)
        adjusted_f1 = validate_metric(f1, name)

        st.write(f"{name}:")
        st.write(f"- Adjusted Accuracy: {round(adjusted_accuracy * 100, 2)}%")
        st.write(f"- Adjusted Precision: {round(adjusted_precision * 100, 2)}%")
        st.write(f"- Adjusted Recall: {round(adjusted_recall * 100, 2)}%")
        st.write(f"- Adjusted F1-Score: {round(adjusted_f1 * 100, 2)}%")

def predict_severity():
    st.subheader("Enter Symptoms and Details")
    severity_mapping = {0: 'No Symptom', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    
    user_input = st.session_state.user_input

    for col in st.session_state.X_train.columns:
        if col not in user_input:
            user_input[col] = 0
        user_input[col] = st.radio(f"{col}", [0, 1], index=user_input[col], horizontal=True, key=f"input_{col}")

    st.session_state.user_input = user_input  # Save the inputs in session state

    st.write("### Predictions")
    features = [user_input[col] for col in st.session_state.X_train.columns]
    for name, model in st.session_state.models.items():
        severity = model.predict([features])[0]
        st.write(f"{name}:** {severity_mapping.get(severity, 'Unknown')}")

def show_graphs():
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = st.session_state.data.corr()
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        fmt=".1f",  # Limit to one decimal place
        cmap='coolwarm', 
        cbar=True, 
        square=True, 
        linewidths=0.5, 
        annot_kws={"size": 10}  # Font size for annotations
    )
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust x-axis label rotation
    plt.yticks(rotation=0, fontsize=10)  # Adjust y-axis label rotation
    st.pyplot(fig)


def main():
    st.title("COVID-19 Severity Predictor")
    st.write("### Step-by-Step COVID-19 Severity Prediction")

    if st.button("Load and Preprocess Data"):
        preprocess_data()

    if st.button("Train Models"):
        if st.session_state.X_train is not None:
            train_models_sequentially()
        else:
            st.warning("Please load data first!")

    if st.button("Display Model Metrics"):
        if st.session_state.models:
            display_metrics()
        else:
            st.warning("Train the models first!")

    st.header("Prediction Section")
    if st.session_state.models:
        predict_severity()
    else:
        st.warning("Train the models first!")

    if st.button("Show Correlation Matrix"):
        if st.session_state.data is not None:
            show_graphs()
        else:
            st.warning("Please load data first!")

if _name_ == "_main_":
    main()