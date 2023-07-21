import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import streamlit as st

st.set_page_config(
    page_icon=":goose:",
    page_title="부지런한 거위들",
    layout="wide",
)
st.write("""
# 부지런한 거위들의 데이터 분석 프로젝트
""")

data_reg = pd.read_csv('Regression_data.csv')
# Separate the features and the target
X_reg = data_reg.drop("Rings", axis=1)
y_reg = data_reg["Rings"]

# Apply one-hot encoding to the 'Sex' feature
encoder = OneHotEncoder(sparse=False, drop='first')
sex_encoded = encoder.fit_transform(X_reg[['Sex']])
sex_encoded_df = pd.DataFrame(sex_encoded, columns=['Sex_Female', 'Sex_Infant'])


# sex_encoded_df = pd.DataFrame(sex_encoded, columns=encoder.get_feature_names(['Sex']))

# Replace the 'Sex' column with the encoded columns
X_reg = X_reg.drop("Sex", axis=1)
X_reg = pd.concat([X_reg, sex_encoded_df], axis=1)

# Split the data into training and test sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

st.write("""SEX column을 one-hot encoding으로 변환하고 Sex_Female, Sex_Infant로 나눔""")
st.dataframe(X_train_reg)

# Compute the correlation matrix
corr = X_reg.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
heatmap_1 = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

st.write("""Heatmap 1""")
st.pyplot(heatmap_1.figure)

# Draw the heatmap with the mask and correct aspect ratio
# Here, we will not use the mask, and add the annotation to display the correlation scores
heatmap_2 = sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

st.write("""Heatmap 2""")
st.pyplot(heatmap_2.figure)


# Initialize a linear regression model
model_reg = LinearRegression()

# Fit the model to the training data
model_reg.fit(X_train_reg, y_train_reg)

# Use the model to make predictions on the test data
y_pred_reg = model_reg.predict(X_test_reg)

# Calculate the performance metrics
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

if st.button("Linear Regression code 보기"):
    code = """
    
# Initialize a linear regression model
model_reg = LinearRegression()

# Fit the model to the training data
model_reg.fit(X_train_reg, y_train_reg)

# Use the model to make predictions on the test data
y_pred_reg = model_reg.predict(X_test_reg)

# Calculate the performance metrics
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
"""
    st.code(code, language='python')

st.subheader("""Linear Regression""")
st.write("평균 제곱 오차 (Mean Squared Error, MSE):", mse)  
st.write("평균 제곱근 오차 (Root Mean Squared Error, RMSE):", rmse)
st.write("평균 절대 오차 (Mean Absolute Error, MAE):", mae)
st.write("결정 계수 (R^2 Score):", r2)

# Initialize a lasso regression model
lasso_reg = Lasso()

parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


# Perform grid search
grid_search = GridSearchCV(lasso_reg, parameters, cv=5)
grid_search.fit(X_train_reg, y_train_reg)

# Get the best model
best_model = grid_search.best_estimator_

# Use the model to make predictions on the test data
y_pred_reg = best_model.predict(X_test_reg)

# Calculate the performance metrics
lasso_mse = mean_squared_error(y_test_reg, y_pred_reg)
lasso_rmse = np.sqrt(mse)
lasso_mae = mean_absolute_error(y_test_reg, y_pred_reg)
lasso_r2 = r2_score(y_test_reg, y_pred_reg)

if st.button("Lasso Regression code 보기"):
    code_lasso = """
    # Initialize a lasso regression model
lasso_reg = Lasso()

parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


# Perform grid search
grid_search = GridSearchCV(lasso_reg, parameters, cv=5)
grid_search.fit(X_train_reg, y_train_reg)

# Get the best model
best_model = grid_search.best_estimator_

# Use the model to make predictions on the test data
y_pred_reg = best_model.predict(X_test_reg)

# Calculate the performance metrics
lasso_mse = mean_squared_error(y_test_reg, y_pred_reg)
lasso_rmse = np.sqrt(mse)
lasso_mae = mean_absolute_error(y_test_reg, y_pred_reg)
lasso_r2 = r2_score(y_test_reg, y_pred_reg)
"""
    st.code(code_lasso, language='python')

st.subheader("""Lasso Regression""")
st.write("평균 제곱 오차 (Mean Squared Error, MSE):", lasso_mse)
st.write("평균 제곱근 오차 (Root Mean Squared Error, RMSE):", lasso_rmse)
st.write("평균 절대 오차 (Mean Absolute Error, MAE):", lasso_mae)
st.write("결정 계수 (R^2 Score):", lasso_r2)

# Initialize a standard scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform the data
X_train_scaled = scaler.fit_transform(X_train_reg)

# Transform the test data
X_test_scaled = scaler.transform(X_test_reg)

# Perform grid search with the scaled data
grid_search = GridSearchCV(lasso_reg, parameters, cv=5)
grid_search.fit(X_train_scaled, y_train_reg)

# Get the best model
best_model = grid_search.best_estimator_

# Use the model to make predictions on the test data
y_pred_reg = best_model.predict(X_test_scaled)

# Calculate the performance metrics
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

if st.button("Lasso Regression with Standard Scaler code 보기"):
    code_lasso_scaler = """
    # Initialize a standard scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform the data
X_train_scaled = scaler.fit_transform(X_train_reg)

# Transform the test data
X_test_scaled = scaler.transform(X_test_reg)

# Perform grid search with the scaled data
grid_search = GridSearchCV(lasso_reg, parameters, cv=5)
grid_search.fit(X_train_scaled, y_train_reg)

# Get the best model
best_model = grid_search.best_estimator_

# Use the model to make predictions on the test data
y_pred_reg = best_model.predict(X_test_scaled)

# Calculate the performance metrics
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
"""
    st.code(code_lasso_scaler, language='python')

st.subheader("""Lasso Regression with Standard Scaler""")
st.write("평균 제곱 오차 (Mean Squared Error, MSE):", mse)
st.write("평균 제곱근 오차 (Root Mean Squared Error, RMSE):", rmse)
st.write("평균 절대 오차 (Mean Absolute Error, MAE):", mae)
st.write("결정 계수 (R^2 Score):", r2)
