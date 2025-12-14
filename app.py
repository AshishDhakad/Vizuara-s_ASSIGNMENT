import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

import matplotlib.pyplot as plt

st.set_page_config(page_title="No-Code ML Pipeline Builder", layout="wide")

st.markdown("""
<style>
.stButton>button {
    background-color:#4CAF50;
    color:white;
    border-radius:8px;
    padding:8px 18px;
}
.step-box {
    background:#f6f8fc;
    padding:15px;
    border-radius:10px;
    margin-bottom:15px;
}
</style>
""", unsafe_allow_html=True)

st.title(" No-Code ML Pipeline Builder")


if "df" not in st.session_state:
    st.session_state.df = None

if "dataset_done" not in st.session_state:
    st.session_state.dataset_done = False

if "feature_done" not in st.session_state:
    st.session_state.feature_done = False

if "preprocess_done" not in st.session_state:
    st.session_state.preprocess_done = False

if "model_done" not in st.session_state:
    st.session_state.model_done = False

# -----------------------------
# Sidebar (STATUS + ACTIONS)
# -----------------------------
st.sidebar.header(" Pipeline Status")


# -----------------------------
# STEP 1: Upload Dataset
# -----------------------------
st.markdown("<div class='step-box'>", unsafe_allow_html=True)
st.header("Step 1️ Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.session_state.df = df
    st.session_state.dataset_done = True

    st.success("Dataset uploaded successfully")
    st.dataframe(df.head())
    st.sidebar.write("Dataset Uploaded:", "uploaded" if st.session_state.dataset_done else "failed")

st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.dataset_done:
    st.stop()

df = st.session_state.df

st.markdown("<div class='step-box'>", unsafe_allow_html=True)
st.header("Step 2️ Select Target & Features")

categorical_columns = []
reasons = {}
for col in df.columns:
    unique_values = df[col].nunique()
    if df[col].dtype == "object" or str(df[col].dtype) == "category":
        categorical_columns.append(col)
        reasons[col] = "Text-based column"
    elif np.issubdtype(df[col].dtype, np.number) and unique_values <= 10 and unique_values > 1:
        d = df[col] <= 10
        s = d.sum()
        if s == len(df[col]):
            categorical_columns.append(col)
            reasons[col] = f"Numeric column with {df[col].nunique()} unique values"

if len(categorical_columns) == 0:
    st.error("No valid categorical column found for target")
    st.stop()

for col in categorical_columns:
    with st.expander("Why is '" + col + "' categorical?"):
        st.info(reasons[col])
        st.write("Sample values:", df[col].unique()[:5])

selected_target = st.selectbox("Select Target Column", categorical_columns, index=None)

selected_features = []

if selected_target is not None:
    for col in df.columns:
        if col != selected_target:
            selected_features.append(col)

    selected_features = st.multiselect("Select Feature Columns", selected_features)

if selected_target is not None and len(selected_features) > 0:
    st.session_state.feature_done = True
    st.sidebar.write("Features Selected:", "done" if st.session_state.feature_done else "failed")
    

st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.feature_done:
    st.stop()

st.markdown("<div class='step-box'>", unsafe_allow_html=True)
st.header("Step 3️ Preprocessing")

preprocess_choice = st.radio(
    "Select preprocessing method",
    ["Standardization", "Normalization"],
    index=None
)

if preprocess_choice is not None:
    X = df[selected_features].copy()
    y = df[selected_target].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    for col in X.columns:
        if X[col].dtype == "object":
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])

    if preprocess_choice == "Standardization":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    st.session_state.X = X_scaled
    st.session_state.y = y
    st.session_state.preprocess_done = True
    st.sidebar.write("Preprocessing Done:", "done" if st.session_state.preprocess_done else "fail")

    st.success("Preprocessing completed successfully")

st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.preprocess_done:
    st.stop()

st.markdown("<div class='step-box'>", unsafe_allow_html=True)
st.header("Step 4️ Train–Test Split")

split_option = st.selectbox("Select split ratio", ["70 / 30", "80 / 20"])

if split_option == "70 / 30":
    test_size = 0.3
else:
    test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    st.session_state.X,
    st.session_state.y,
    test_size=test_size,
    random_state=42
)
if len(selected_features) == 2:
        st.subheader("Plot of features")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(x=X_train[:,0],y=X_train[:,1],hue=y_train)
        st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='step-box'>", unsafe_allow_html=True)
st.header("Step 5️ Model Selection")



model_choice = st.selectbox(
    "Choose algorithm",
    ["Logistic Regression", "Decision Tree"],
    index=None
)

st.markdown("</div>", unsafe_allow_html=True)

if model_choice is None:
    st.stop()

st.markdown("<div class='step-box'>", unsafe_allow_html=True)
st.header("Step 6️ Train Model & Results")

if st.button(" Train Model"):

    if model_choice == "Logistic Regression":
        if df[selected_target].nunique() > 2:
            st.error("Logistic Regression can be used only for 2 classes. for more than 2 classes please select another classification method")
            st.stop()
        else:
           model = LogisticRegression(max_iter=1000)
    else:
        model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train)) * 100
    test_accuracy = accuracy_score(y_test, model.predict(X_test)) * 100

    st.metric("Training Accuracy (%)", f"{train_accuracy:.2f}")
    st.metric("Testing Accuracy (%)", f"{test_accuracy:.2f}")

    if model_choice == "Decision Tree":
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(selected_features, model.feature_importances_)
        ax.set_xticklabels(selected_features, rotation=45, ha="right")
        st.pyplot(fig)
    
    
    
    if train_accuracy - test_accuracy > 10:
        st.warning(" Model is Overfitting")
    elif train_accuracy < 60 and test_accuracy < 60:
        st.warning(" Model is Underfitting")
    else:
        st.success(" Model is Well-Fitted")

    st.session_state.model_done = True
    st.sidebar.write("Model Trained:", "done" if st.session_state.model_done else "fail")

    st.sidebar.markdown("---")
    st.sidebar.subheader(" Summary")
    st.sidebar.write("Target:", selected_target)
    st.sidebar.write("Features:", selected_features)
    st.sidebar.write("Preprocessing:", preprocess_choice)
    st.sidebar.write("Split:", split_option)
    st.sidebar.write("Train Acc (%):", round(train_accuracy, 2))
    st.sidebar.write("Test Acc (%):", round(test_accuracy, 2))


st.markdown("</div>", unsafe_allow_html=True)


