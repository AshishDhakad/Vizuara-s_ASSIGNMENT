# Vizuara-s_ASSIGNMENT


.

** Project Overview – No-Code ML Pipeline Builder**

This Streamlit app allows users to build and train a Machine Learning classification model without writing code.
Users can upload a dataset, select features, apply preprocessing, choose a model, and see results step-by-step through an interactive UI.

** How This App Works (Step-by-Step)**

 **How to Run the App**
 pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib
 streamlit run app.py

**Step 1: Upload Dataset**

Upload a CSV or Excel file

The dataset preview is shown

App moves forward only after upload

**Step 2: Select Target & Features**

The app automatically detects categorical columns suitable for prediction

You select:

Target column (what you want to predict)

Feature columns (inputs for the model)

**Step 3: Preprocessing**

Choose one preprocessing method:

Standardization (mean = 0, std = 1)

Normalization (values between 0 and 1)

Text columns are encoded automatically

Target column is label-encoded
**
**Step 4: Train–Test Split

Choose dataset split:

70 / 30 or 80 / 20

If exactly 2 features are selected, a scatter plot is shown

**Step 5: Model Selection**

Choose a classification model:

Logistic Regression (only for binary classification)

Decision Tree Classifier

Step 6: Train Model & View Results

Click Train Model

Displays:

Training Accuracy

Testing Accuracy

Shows:

Feature importance (Decision Tree only)

Overfitting / Underfitting warnings

Full summary is shown in the sidebar

 **Tech Stack Used**

Streamlit – UI & workflow

Pandas / NumPy – Data handling

Scikit-Learn – ML models & preprocessing

Matplotlib / Seaborn – Visualizations

