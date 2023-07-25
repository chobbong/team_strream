import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.write("""
# Multi-classification Analysis
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    n_estimators = st.sidebar.slider('n_estimators', 100, 200, 10)
    learning_rate = st.sidebar.slider('learning_rate', 0.1, 0.5, 0.1)
    data = {'n_estimators': n_estimators,
            'learning_rate': learning_rate,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = pd.read_csv('./csv/multi_classification_data.csv')

# Define the feature set 'X' and the target set 'y'
X = data.iloc[:, :-7]
y = data.iloc[:, -7:]

# Define the target set without the 'Other_Faults' column
y_without_other = y.drop('Other_Faults', axis=1)

# Split the data without 'Other_Faults' into training and validation sets
X_train_without, X_val_without, y_train_without, y_val_without = train_test_split(
    X, y_without_other, test_size=0.2, random_state=42, stratify=y_without_other)

# Train the Gradient Boosting Classifier with the best hyperparameters
best_gb = GradientBoostingClassifier(n_estimators=df['n_estimators'][0], learning_rate=df['learning_rate'][0], random_state=42)
best_gb.fit(X_train_without, y_train_without.idxmax(axis=1))

# Predict the classes on validation data
y_pred_without_best_gb = best_gb.predict(X_val_without)

# Compute the classification report
classification_report_dict = classification_report(y_val_without.idxmax(axis=1), y_pred_without_best_gb, output_dict=True)

# Convert the classification report to a DataFrame
classification_report_df = pd.DataFrame(classification_report_dict).transpose()

st.write('Classification report:\n')
st.write(classification_report_df)
         

