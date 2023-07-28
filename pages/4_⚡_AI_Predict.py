
import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV

import seaborn as sns
import matplotlib.pyplot as plt
import joblib

import streamlit as st

st.set_page_config(
    page_icon=":goose:",
    page_title="부지런한 거위",
    layout="wide",
)

st.header("""
예측 모델 
""")

class Model1:
    def __init__(self):
        # 데이터 로드
        data_reg = pd.read_csv('./csv/Regression_data.csv')

        # Separate the features and the target
        X_reg = data_reg.drop("Rings", axis=1)
        y_reg = data_reg["Rings"]

        # Apply one-hot encoding to the 'Sex' feature
        self.encoder = OneHotEncoder(sparse=False, drop='first')
        sex_encoded = self.encoder.fit_transform(X_reg[['Sex']])
        sex_encoded_df = pd.DataFrame(sex_encoded, columns=['Sex_Female', 'Sex_Infant'])

        # Replace the 'Sex' column with the encoded columns
        X_reg = X_reg.drop("Sex", axis=1)
        X_reg = pd.concat([X_reg, sex_encoded_df], axis=1)

        # Split the data into training and test sets
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

        # Initialize a linear regression model
        self.model = LinearRegression()

        # Fit the model to the training data
        self.model.fit(X_train_reg, y_train_reg)

    def predict(self, inputs):
        # 입력 데이터를 변환하고 모델을 사용하여 예측하는 코드를 작성해야 합니다.
        inputs = pd.DataFrame([inputs], columns=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight'])
        inputs[['Sex_Female', 'Sex_Infant']] = self.encoder.transform(inputs[['Sex']])
        inputs = inputs.drop(columns=['Sex'])
        prediction = self.model.predict(inputs)
        return prediction[0]
    
class PulsarPredictor:
    def __init__(self):
        # Load data
        data_bin = pd.read_csv('./csv/binary_classification_data.csv')

        # Separate the features and the target
        X = data_bin.drop("target_class", axis=1)
        y = data_bin["target_class"]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Initialize a new StandardScaler instance
        self.scaler = StandardScaler()

        # Fit the scaler to the training data
        self.scaler.fit(X_train)

        # Scale the training data
        X_train = self.scaler.transform(X_train)

        # Initialize an RFECV instance
        selector = RFECV(LogisticRegression(random_state=42), step=1, cv=5)

        # Fit the selector to the data
        selector.fit(X_train, y_train)

        # Get the mask of selected features
        selected_features_mask = selector.support_

        # Select the features from the training and test sets
        X_train_selected = X_train[:, selected_features_mask]

        # Initialize a new LogisticRegression instance
        self.model = LogisticRegression(random_state=42)

        # Fit the model to the training data
        self.model.fit(X_train_selected, y_train)

        # Save the selected features mask for prediction
        self.selected_features_mask = selected_features_mask

    def predict(self, inputs):
        # Scale the inputs using the trained scaler
        inputs = self.scaler.transform([inputs])

        # Select the features using the saved mask
        inputs_selected = inputs[:, self.selected_features_mask]

        # Use the model to predict the outcome
        prediction = self.model.predict(inputs_selected)

        return prediction[0]
    
class DefectPredictor:
    def __init__(self):
        self.clf = None
    
    def preprocess_data(self, data_multi):
        # Drop the 'TypeOfSteel_A400' column
        data_multi = pd.read_csv('./csv/multi_classification_data.csv')
        # Drop the 'TypeOfSteel_A400' column
        data_multi = data_multi.drop(['TypeOfSteel_A400'], axis=1)
        data_multi = data_multi.drop(['Outside_Global_Index'], axis=1)

        # Handle outliers by clipping to the 1.5 IQR range
        for Feature in data_multi.columns:
            Q1 = data_multi[Feature].quantile(0.25)
            Q3 = data_multi[Feature].quantile(0.75)
            IQR = Q3 - Q1
            line_down = data_multi[Feature].quantile(0.25) - IQR * 1.5
            line_up = data_multi[Feature].quantile(0.75) + IQR * 1.5
            data_multi[Feature] = data_multi[Feature].clip(line_down, line_up)

        return data_multi

    def train_model(self, data_multi):
        # Split the data into features and targets
        features = data_multi.drop(['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'], axis=1)
        targets_with_other = data_multi[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, targets_with_other.idxmax(axis=1), test_size=0.2, random_state=42, stratify=targets_with_other.idxmax(axis=1))

        # Define a Gradient Boosting classifier with optimal parameters (Replace with your optimal parameters)
        self.clf = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=0.3, min_samples_leaf=0.1, max_depth=5, max_features='sqrt', criterion='friedman_mse', subsample=0.9, n_estimators=100)

        # Train the model
        self.clf.fit(X_train, y_train)

        # Make predictions
        y_pred = self.clf.predict(X_test)

        # Save the trained model for future use
        joblib.dump(self.clf, 'clf_model.pkl')

    def predict_defect(self, input_data):
        # Ensure the input is array-like and reshape it for the model
        input_data = np.array(input_data).reshape(1, -1)
        
        # Make the prediction
        prediction = self.clf.predict(input_data)
        
        # Return the predicted defect
        return prediction
    

tab1, tab2, tab3 = st.tabs(["전복나이예측", "펄서여부예측", "스테인레스결함예측"])

with tab1:

    st.subheader('전복 나이 예측')

    # 모델 로드
    model1 = Model1()

    col1, col2 = st.columns(2)

    with col1: 
        # 사용자 입력 양식
        input_sex = st.selectbox('성별', ['F', 'M','I'])
        input_length = st.slider('Length',0.01, 1.0, 0.01)
        input_diameter = st.slider('Diameter', 0.01, 1.0, 0.01)
        input_height = st.slider('Height', 0.01, 1.0, 0.01)
        input_wholeWeight = st.slider('Whole weight', 0.01, 1.0, 0.01)
        input_shuckedWeight = st.slider('Shucked weight', 0.01, 1.0, 0.01)
        input_visceraWeight = st.slider('Viscera weight', 0.01, 1.0, 0.01)
        input_shellWeight = st.slider('Shell weight', 0.01, 1.0, 0.01)

        inputs = [input_sex, 
                input_length, 
                input_diameter, 
                input_height, 
                input_wholeWeight, 
                input_shuckedWeight, 
                input_visceraWeight, 
                input_shellWeight
                ]

    # 예측
    predicted_size = model1.predict(inputs)
 
    with col2:
        # 결과 표시
        st.write( '    ')
        st.write( '    ')
        st.write( '    ')
        st.write( '    ')
        st.write( '    ')
        st.write( '    ')
        st.write( '    ')
        st.write( '    ')
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write('### 전복 나이 예측:', predicted_size)

with tab2:
    model2 = PulsarPredictor()

    st.subheader('Pulsar Predictor')

    col1, col2 = st.columns(2)

    with col1:

        input_0 = st.slider('Mean of the integrated profile', 6.0, 190.0, 1.0)
        input_1 = st.slider('Standard deviation of the integrated profile',24.0, 100.0, 1.0)
        input_2 = st.slider('Excess kurtosis of the integrated profile', 0.0, 10.0, 0.1)
        input_3 = st.slider('Skewness of the integrated profile', 0.0, 70.0, 1.0)
        input_4 = st.slider('Mean of the DM-SNR curve ', 0.0, 220.0, 1.0)
        input_5 = st.slider('Standard deviation of the DM-SNR curve', 7.0, 110.0, 1.0)
        input_6 = st.slider('Excess kurtosis of the DM-SNR curve', 0.0, 40.0, 1.0)
        input_7 = st.slider('Skewness of the DM-SNR curve', 0.0, 122.0, 10.0)

    inputs = [input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7]
    result = model2.predict(inputs)

    with col2:
        st.write( '    ')
        st.write( '    ')
        st.write( '    ')
        st.write( '    ')
        st.write( '    ')
        st.write( '    ')
        st.write( '    ')
        st.write( '    ')
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        if result == 1:
            st.write('### 펄서 여부 예측 : 펄서임')
        else:
            st.write('### 펄서 여부 예측 : 펄서 아님')

with tab3:

    model3 = DefectPredictor()
    data_multi = model3.preprocess_data('./csv/multi_classification_data.csv')
    model3.train_model(data_multi)

    st.subheader('스테인레스결함 예측')

    col1, col2 = st.columns(2)

    with col1:
        # Create sliders for each feature
        input_0 = st.slider('X_Minimum', 0.0, 100.0, 1.0) #1
        input_1 = st.slider('X_Maximum', 0.0, 100.0, 1.0) #2
        input_2 = st.slider('Y_Minimum', 0.0, 100.0, 1.0) #3
        input_3 = st.slider('Y_Maximum', 0.0, 100.0, 1.0) #4
        input_4 = st.slider('Pixels_Areas', 0.0, 100.0, 1.0) #5
        input_5 = st.slider('X_Perimeter', 0.0, 100.0, 1.0) #6
        input_6 = st.slider('Y_Perimeter', 0.0, 100.0, 1.0) #7
        input_7 = st.slider('Sum_of_Luminosity', 0.0, 100.0, 1.0) #8
        input_8 = st.slider('Minimum_of_Luminosity', 0.0, 100.0, 1.0) #9
        input_9 = st.slider('Maximum_of_Luminosity', 0.0, 100.0, 1.0) #10
        input_10 = st.slider('Length_of_Conveyer', 0.0, 100.0, 1.0) #11
        input_11 = st.slider('TypeOfSteel_A300', 0.0, 1.0, 1.0) #12
        input_12 = st.slider('Steel_Plate_Thickness', 0.0, 100.0, 1.0) #13
        input_13 = st.slider('Edges_Index', 0.0, 1.0, 0.01) #14
        input_14 = st.slider('Empty_Index', 0.0, 1.0, 0.01) #15
        input_15 = st.slider('Square_Index', 0.0, 1.0, 0.01) #16
        input_16 = st.slider('Outside_X_Index', 0.0, 1.0, 0.01) #17
        input_17 = st.slider('Edges_X_Index', 0.0, 1.0, 0.01) #18
        input_18 = st.slider('Edges_Y_Index', 0.0, 1.0, 0.01) #19
        input_19 = st.slider('LogOfAreas', 0.0, 100.0, 1.0) #20
        input_20 = st.slider('Log_X_Index', 0.0, 100.0, 1.0) #21
        input_21 = st.slider('Log_Y_Index', 0.0, 100.0, 1.0) #22
        input_22 = st.slider('Orientation_Index', 0.0, 1.0, 0.01) #23
        input_23 = st.slider('Luminosity_Index', 0.0, 1.0, 0.01) #24
        input_24 = st.slider('SigmoidOfAreas', 0.0, 1.0, 0.01) #25

        # Gather all inputs
        inputs = [input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24]

    with col2:
  
        predicted_defect = model3.predict_defect(inputs)
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write( '    ') 
        st.write('### 스테인레스 결함:', predicted_defect)
        

