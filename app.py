
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🧬",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("Breast Cancer Classification")
st.write(
    "Machine Learning Assignment 2 - "
    "Classification Model Evaluation"
)

st.markdown(
    """
    This application evaluates five machine learning
    classification models using the uploaded test dataset.
    """
)


# ==========================================================
# MODEL FILE LOCATIONS
# ==========================================================

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}


# ==========================================================
# UPLOAD TEST DATA
# ==========================================================

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Please upload the test CSV file to begin evaluation."
    )

    st.stop()


# ==========================================================
# READ DATA
# ==========================================================

test_data = pd.read_csv(uploaded_file)

st.success("Test dataset uploaded successfully.")

st.subheader("Dataset Preview")

st.dataframe(
    test_data.head(),
    use_container_width=True
)


# ==========================================================
# CHECK TARGET COLUMN
# ==========================================================

if "target" not in test_data.columns:

    st.error(
        "The uploaded CSV must contain a 'target' column."
    )

    st.stop()


# ==========================================================
# SEPARATE FEATURES AND TARGET
# ==========================================================

X_test = test_data.drop(
    columns=["target"]
)

y_test = test_data["target"]


# ==========================================================
# MODEL SELECTION
# ==========================================================

st.header("2. Select Classification Model")

selected_model = st.selectbox(
    "Choose a model:",
    list(MODEL_FILES.keys())
)


# ==========================================================
# CHECK MODEL FILE
# ==========================================================

model_path = MODEL_FILES[selected_model]

if not os.path.exists(model_path):

    st.error(
        f"Model file not found: {model_path}"
    )

    st.stop()


# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(model_path)


# ==========================================================
# PREPROCESS INPUT
# ==========================================================

if selected_model in [
    "Logistic Regression",
    "kNN"
]:

    scaler_path = "model/scaler.pkl"

    if not os.path.exists(scaler_path):

        st.error(
            "Scaler file not found."
        )

        st.stop()

    scaler = joblib.load(
        scaler_path
    )

    X_input = scaler.transform(
        X_test
    )

else:

    X_input = X_test


# ==========================================================
# MAKE PREDICTIONS
# ==========================================================

y_pred = model.predict(
    X_input
)

y_prob = model.predict_proba(
    X_input
)[:, 1]


# ==========================================================
# CALCULATE METRICS
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ==========================================================
# DISPLAY METRICS
# ==========================================================

st.header("3. Evaluation Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

with col2:
    st.metric(
        "AUC",
        f"{auc:.4f}"
    )

with col3:
    st.metric(
        "Precision",
        f"{precision:.4f}"
    )


col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Recall",
        f"{recall:.4f}"
    )

with col5:
    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

with col6:
    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

st.header("4. Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

st.header("5. Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(5, 4)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Actual"
)

ax.set_title(
    f"Confusion Matrix - {selected_model}"
)

st.pyplot(fig)


# ==========================================================
# MODEL COMPARISON
# ==========================================================

st.header("6. Model Comparison")

comparison_path = "model_results.csv"

if os.path.exists(comparison_path):

    comparison_df = pd.read_csv(
        comparison_path
    )

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True
    )

else:

    st.warning(
        "model_results.csv was not found."
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.write(
    "Machine Learning Assignment 2"
)
