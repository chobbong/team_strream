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

tab1, tab2, tab3 = st.tabs(["전복나이예측", "펄서여부예측", "스테인레스결함예측"])

with tab1:

   st.write('### 전복나이예측 (Regression_data)')
   select_model = st.selectbox('Select a model', ['regression_nn','XGBoost','LinearRegression', 'Lasso','StandardScaler+GridSearchCV'])

   if select_model == 'regression_nn':
      st.subheader("""regression_nn""")
      accuracy_nn = 0.859
      st.write('#### acc :', accuracy_nn) 
      st.write('#### the performance metrics')
      st.write("""
      MAE: 1.4769553656213021
      MSE: 4.476184649680055
      RMSE: 2.1156995650800834
      R2 Score: 0.5865036677494322
       """)

   if select_model == 'XGBoost':
        
      XGB_rmse = 2.124
      XGB_acc = 0.852
      st.write("""
      ### XGBoost
      """)
      st.write("""
      **best params**   
      'learning_rate': 0.02087425763287998   
      'n_estimators': 1550  
      'max_depth': 17    
      'colsample_bytree': 0.5    
      'l2': 10.670146505870857    
      'l1': 0.0663394675391197     
      'gamma': 9.015017136084957  
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

with tab2:

   st.write('### 펄서여부 예측 (Binary Classification data)')
   st.write('select_model == "XGBoost"')
   st.write("""
   #### best params :      
   smote_k': 2,       
   'enn_k': 6,     
   'learning_rate': 0.03233685808565227,     
   'n_estimators': 1200,     
   'max_depth': 20,     
   'colsample_bytree': 0.5,     
   'l2': 0.004666963217784473,     
   'l1': 0.002792083422830542,     
   'gamma': 0.036934880241175236,     
   'scale_pos_weight': 7.0    
   """)
   accuracy = 0.981
   st.write('#### accuracy :', accuracy) 
   
with tab3:
   st.write('### 스테인레스 결함 예측 (Multi Classification data)')
   st.write('select_model == "XGBoost"')
   st.write("""
   #### best params :        
  'learning_rate': 0.07522487380833985,   
  'n_estimators': 250,   
  'max_depth': 4,   
  'colsample_bytree': 0.6000000000000001,   
  'l2': 0.001648272236870337,   
  'l1': 0.01657588037413299,   
  'gamma': 0.002792373320363197  
   """)
   bin_accuracy = 0.844
   multi_accuracy = 0.934
   st.write('#### binary accuracy :', bin_accuracy)
   st.write('#### multi accuracy :', multi_accuracy) 
      