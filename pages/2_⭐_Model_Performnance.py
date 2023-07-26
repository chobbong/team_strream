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
    page_title="부지런한 거위",
    layout="wide",
)

st.header("""
 Model Perfortmance
""")

st.sidebar.subheader("""
 Model Perfortmance
""")


data_reg = pd.read_csv('./csv/Regression_data.csv')
data_bin = pd.read_csv('./csv/binary_classification_data.csv')
data_mul = pd.read_csv('./csv/multi_classification_data.csv')

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

# st.write("""SEX column을 one-hot encoding으로 변환하고 Sex_Female, Sex_Infant로 나눔""")
# st.dataframe(X_train_reg)

# # Compute the correlation matrix
# corr = X_reg.corr()

# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# heatmap_1 = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})

# st.write("""Heatmap 1""")
# st.pyplot(heatmap_1.figure)

# # Draw the heatmap with the mask and correct aspect ratio
# # Here, we will not use the mask, and add the annotation to display the correlation scores
# heatmap_2 = sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

# st.write("""Heatmap 2""")
# st.pyplot(heatmap_2.figure)

select_dataset = st.sidebar.selectbox('select a data-set', ['data-1', 'data-2','data-3'])

if select_dataset == 'data-1':

   st.write('### data-1 (Regression_data)')
   select_model = st.selectbox('Select a model', ['XGBoost','LinearRegression', 'Lasso','StandardScaler+GridSearchCV'])

   if select_model == 'XGBoost':
        
      XGB_rmse = 2.124
      XGB_acc = 0.852
      st.write("""
      ### XGBoost
      """)
      st.write("""
      **best params**   
      'learning_rate': 0   
      'n_estimators': 0
      'max_depth': 0   
      'colsample_bytree': 0  
      'l2': 0  
      'l1':0    
      'gamma': 0 
      """)
      st.write('#### rmse :', XGB_rmse) 
      st.write('#### acc :', XGB_acc) 
      st.image('./img/xgb.png')
   
      if st.button("XGBoost code 보기"):
         code = """
      
         # XGBRegressor 튜닝

         def reg_xgb_model(trial, X_train, X_test, y_train, y_test):
            model_params = {
               'learning_rate':trial.suggest_float('learning_rate', 0.0001, 1.0, log=True),
               'n_estimators':trial.suggest_int('n_estimators', 100, 2000, step=50),
               'max_depth':trial.suggest_int('max_depth', 4, 20),
               'colsample_bytree':trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1),
               'reg_lambda':trial.suggest_float('l2', 0.001, 100.0, log=True),
               'reg_alpha':trial.suggest_float('l1', 0.001, 100.0, log=True),
               'gamma':trial.suggest_float('gamma', 0.001, 100.0, log=True),
            }

            model = XGBRegressor(random_state=42,
                                 objective='reg:squarederror',
                                 eval_metric='rmse',
                                 tree_method='gpu_hist',
                                 gpu_id=0,
                                 early_stopping_rounds=30,
                                 **model_params)
            
            model.fit(X_train, y_train,
                     eval_set=[(X_test, y_test)],
                     verbose=0)
            
            return model

         def reg_xgb_objective(trial, X_train, X_test, y_train, y_test):
            
            model = reg_xgb_model(trial, X_train, X_test, y_train, y_test)
            rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)

            return rmse

         reg_xgb_best = reg_xgb_model(reg_xgb_study.best_trial, X1_train, X1_test, y1_train, y1_test)

         def reg_performance(model, X_test, y_test=None, plot=False):
            y_pred = pd.Series(model.predict(X_test), index=y_test.index)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            acc = np.mean(1-abs((y_pred - y_test) / y_test))
            r2 = r2_score()
            
            if plot==True:
               # y_test의 target 크기별 예측값이 어떤지 확인
               y_test = y_test.sort_values()
               y_pred = y_pred[y_test.index]

               print(f'rmse : {rmse:.3f}\nacc : {acc:.3f}')
               plt.figure(figsize=(12,6))
               plt.plot(y_test.reset_index(drop=True), alpha=0.7, label='y_true')
               plt.plot(y_pred.reset_index(drop=True), alpha=0.7, label='y_pred')
               plt.ylabel('Rings', fontsize=14)
               plt.xlabel('scale order(ascending)', fontsize=14)
               plt.title('True vs Prediction\n(by ascending target value)', fontsize=14)
               plt.legend()
               plt.show()
            else:
               return rmse, acc

         reg_performance(reg_xgb_best, X1_test, y1_test, plot=True)
         """
         st.code(code, language='python')

   elif select_model == 'LinearRegression':
         
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

         st.subheader("""Linear Regression""")
         st.write("평균 제곱 오차 (Mean Squared Error, MSE):", mse)  
         st.write("평균 제곱근 오차 (Root Mean Squared Error, RMSE):", rmse)
         st.write("평균 절대 오차 (Mean Absolute Error, MAE):", mae)
         st.write("결정 계수 (R^2 Score):", r2)

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

   elif select_model == 'Lasso':
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
        lasso_rmse = np.sqrt(lasso_mse)
        lasso_mae = mean_absolute_error(y_test_reg, y_pred_reg)
        lasso_r2 = r2_score(y_test_reg, y_pred_reg)

        st.subheader("""Lasso Regression""")
        st.write("평균 제곱 오차 (Mean Squared Error, MSE):", lasso_mse)
        st.write("평균 제곱근 오차 (Root Mean Squared Error, RMSE):", lasso_rmse)
        st.write("평균 절대 오차 (Mean Absolute Error, MAE):", lasso_mae)
        st.write("결정 계수 (R^2 Score):", lasso_r2)

        if st.button("Lasso code 보기"):
            code = """
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
            lasso_rmse = np.sqrt(lasso_mse)
            lasso_mae = mean_absolute_error(y_test_reg, y_pred_reg)
            lasso_r2 = r2_score(y_test_reg, y_pred_reg)
            """
            st.code(code, language='python')


   elif select_model == 'StandardScaler+GridSearchCV':
      # Initialize a standard scaler
      scaler = StandardScaler()
      lasso_reg = Lasso()
      # Fit the scaler to the training data and transform the data
      X_train_scaled = scaler.fit_transform(X_train_reg)

      # Transform the test data
      X_test_scaled = scaler.transform(X_test_reg)

      parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

      # Perform grid search with the scaled data
      grid_search = GridSearchCV(lasso_reg, parameters, cv=5)
      grid_search.fit(X_train_scaled, y_train_reg)

      # Get the best model
      best_model = grid_search.best_estimator_

      # Use the model to make predictions on the test data
      y_pred_reg = best_model.predict(X_test_scaled)

      # Calculate the performance metrics
      scaler_mse = mean_squared_error(y_test_reg, y_pred_reg)
      scaler_rmse = np.sqrt(scaler_mse)
      scaler_mae = mean_absolute_error(y_test_reg, y_pred_reg)
      scaler_r2 = r2_score(y_test_reg, y_pred_reg)

      st.subheader("""Lasso Regression with Standard Scaler""")
      st.write("평균 제곱 오차 (Mean Squared Error, MSE):", scaler_mse)
      st.write("평균 제곱근 오차 (Root Mean Squared Error, RMSE):", scaler_rmse)
      st.write("평균 절대 오차 (Mean Absolute Error, MAE):", scaler_mae)
      st.write("결정 계수 (R^2 Score):", scaler_r2)

      if st.button("Lasso Regression with Standard Scaler code 보기"):
         code = """
         scaler = StandardScaler()
         lasso_reg = Lasso()
         # Fit the scaler to the training data and transform the data
         X_train_scaled = scaler.fit_transform(X_train_reg)

         # Transform the test data
         X_test_scaled = scaler.transform(X_test_reg)

         parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

         # Perform grid search with the scaled data
         grid_search = GridSearchCV(lasso_reg, parameters, cv=5)
         grid_search.fit(X_train_scaled, y_train_reg)

         # Get the best model
         best_model = grid_search.best_estimator_

         # Use the model to make predictions on the test data
         y_pred_reg = best_model.predict(X_test_scaled)

         # Calculate the performance metrics
         scaler_mse = mean_squared_error(y_test_reg, y_pred_reg)
         scaler_rmse = np.sqrt(scaler_mse)
         scaler_mae = mean_absolute_error(y_test_reg, y_pred_reg)
         scaler_r2 = r2_score(y_test_reg, y_pred_reg)
         """
         st.code(code, language='python')

if select_dataset == 'data-2':

   st.write('### data-2 (Binary Classification data)')
   st.write('select_model == "XGBoost"')
   st.write("""
   #### best params :      
   smote_k': 0,       
   'enn_k': 0,     
   'learning_rate': 0.000,     
   'n_estimators': 000000,     
   'max_depth': 00,     
   'colsample_bytree': 000,     
   'l2': 0.00,     
   'l1': 0.0,     
   'gamma': 0.03,     
   'scale_pos_weight': 00    
   """)
   accuracy = 0.9752
   st.write('#### accuracy :', accuracy) 
   
if select_dataset == 'data-3':
   st.write('### data-3 (Multi Classification data)')
   st.write('select_model == "XGBoost"')
   st.write("""
   #### best params :        
  'learning_rate': 0,   
  'n_estimators': 0,   
  'max_depth': 0,   
  'colsample_bytree': 0,   
  'l2': 0,   
  'l1': 0,   
  'gamma': 0  
   """)
   bin_accuracy = 0.844
   multi_accuracy = 0.934
   st.write('#### binary accuracy :', bin_accuracy)
   st.write('#### multi accuracy :', multi_accuracy) 
      
