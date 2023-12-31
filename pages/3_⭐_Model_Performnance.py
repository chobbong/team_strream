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

   if select_model == 'regression_nn':
      st.subheader("""regression_nn""")
      accuracy_nn = 0.862
      st.write('#### acc :', accuracy_nn) 
      st.write('#### the performance metrics')
      st.write("""
      MAE: 1.4418711713626624  
      MSE: 4.2280861108604055  
      RMSE: 2.0562310451066548   
      R2 Score: 0.609422256652145    
       """)
      
      if st.button("regression_nn code 보기"):
         code = """
         # One-hot encode the 'Sex' column
         ohe = OneHotEncoder(sparse=False)
         sex_encoded = ohe.fit_transform(data[['Sex']])
         sex_encoded_df = pd.DataFrame(sex_encoded, columns=ohe.categories_[0])

         # Concatenate the one-hot encoded columns to the original data frame
         data_encoded = pd.concat([data.drop('Sex', axis=1), sex_encoded_df], axis=1)

         # Separate the features from the target
         X = data_encoded.drop('Rings', axis=1)
         y = data_encoded['Rings']

         # Split the data into training and test sets
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

         # Create TensorFlow datasets for training and validation
         train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
         valid_ds = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))

         # Cache the datasets and batch them
         train_ds = train_ds.cache().shuffle(3500).batch(32)
         valid_ds = valid_ds.cache().shuffle(1000).batch(32)

         # Define the model
         nn = keras.models.Sequential([
            keras.layers.Dense(64, input_shape=[X_train.shape[1],], kernel_regularizer=l1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(32, kernel_regularizer=l1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(16, kernel_regularizer=l1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(8, kernel_regularizer=l1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(1),
         ])

         # Compile the model
         nn.compile(optimizer=keras.optimizers.Adam(0.01), loss='mse')

         # Define the callbacks
         reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=0, mode='auto')
         e_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

         # Train the model
         hist = nn.fit(train_ds, validation_data=valid_ds, epochs=200, callbacks=[reduce_lr, e_stop], verbose=0)


         # # Plot the loss
         # plt.plot(hist.history['loss'], label='train')
         # plt.plot(hist.history['val_loss'], label='val')
         # plt.xlabel('epoch')
         # plt.ylabel('loss')
         # plt.legend()
         # plt.show()

         from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

         # Use the model to make predictions on the test set
         y_pred = nn.predict(X_test)

         # Flatten the arrays (because the output of the model is a 2D array)
         y_test_flat = y_test.values.flatten()
         y_pred_flat = y_pred.flatten()

         # Calculate the performance metrics
         mae = mean_absolute_error(y_test_flat, y_pred_flat)
         mse = mean_squared_error(y_test_flat, y_pred_flat)
         rmse = np.sqrt(mse)  # or mse**0.5
         r2 = r2_score(y_test_flat, y_pred_flat)

         # Print the performance metrics
         print(f"MAE: {mae}")
         print(f"MSE: {mse}")
         print(f"RMSE: {rmse}")
         print(f"R2 Score: {r2}")
         
         # Predict on the test data
         y_pred = nn.predict(X_test).flatten()

         # Calculate the MAPE
         mape = np.mean(np.abs((y_test - y_pred) / y_test))

         # Calculate 1 - MAPE
         accuracy = 1 - mape
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

   select_model_2 = st.selectbox('Select a model', ['XGBoost','LinearRegression'])

   if select_model_2 == 'XGBoost':
      st.write('select_model == "XGBoost"')
      st.write("""
      #### best params :      
      'learning_rate': 0.07599786434716908  
      'n_estimators': 1400  
      'max_depth': 19  
      'colsample_bytree': 0.5  
      'l2': 0.007915306338055306   
      'l1': 0.007964684761239524  
      'gamma': 0.008710642511159737  
      'scale_pos_weight': 0.9  
      """)
      accuracy = 0.981
      st.write('#### accuracy :', accuracy) 
   
   elif select_model_2 == 'LinearRegression':
      st.write('select_model == "LinearRegression"')
      accuracy_li = 0.979
      precision_li = 0.9359
      recall_li = 0.8193
      f1_li = 0.8738
      st.write('#### accuracy :', accuracy_li)
      st.write('#### precision :', precision_li)
      st.write('#### recall :', recall_li)
      st.write('#### f1 :', f1_li)
   
      if st.button("LinearRegression code 보기"):
         code = """
         # Check for missing values
         missing_values = data_bin.isnull().sum()

         # Display the number of missing values per column
         missing_values

         # Define the features and the target
         X = data_bin.drop("target_class", axis=1)
         y = data_bin["target_class"]

         # Initialize a new StandardScaler instance
         scaler = StandardScaler()

         # Fit the scaler to the features and transform
         X_scaled = scaler.fit_transform(X)

         # Convert the scaled features into a DataFrame
         X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

         # Split the data into a training set and a test set
         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

         # Check the shapes of the resulting datasets
         X_train.shape, X_test.shape, y_train.shape, y_test.shape

         # Initialize a new logistic regression model
         model = LogisticRegression(random_state=42)

         # Fit the model to the training data
         model.fit(X_train, y_train)

         # Use the model to make predictions on the test data
         y_pred = model.predict(X_test)

         # Calculate the metrics
         accuracy = accuracy_score(y_test, y_pred)
         precision = precision_score(y_test, y_pred)
         recall = recall_score(y_test, y_pred)
         f1 = f1_score(y_test, y_pred)
         roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # Compute ROC AUC from prediction scores
         """
         st.code(code, language='python')

      
   
with tab3:

   st.write('### 스테인레스 결함 예측 (Multi Classification data)')

   select_model_3 = st.selectbox('Select a model', ['XGBoost_model3','LinearRegression_model3'])

   if select_model_3 == 'XGBoost_model3':

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

   elif select_model_3 == 'LinearRegression_model3':

      st.write('select_model == "LinearRegression"')
      st.write("""
      #### best params :
      "learning_rate": 0.2  
      "min_samples_split": 0.25  
      "min_samples_leaf": 0.1     
      "max_depth":5  
      "max_features":"sqrt"    
      "criterion":"friedman_mse"  
      "subsample":0.95   
      "n_estimators": 10  
      """)
      mo3_li_accuracy = 0.76
      st.write('#### accuracy :', mo3_li_accuracy)
