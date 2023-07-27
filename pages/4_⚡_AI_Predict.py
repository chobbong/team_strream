import time
import joblib
import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.utils import compute_sample_weight
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score, recall_score, precision_score
# from sklearn.metrics import precision_recall_curve

from xgboost import XGBRegressor
from xgboost import XGBClassifier

import streamlit as st

st.set_page_config(
    page_icon=":goose:",
    page_title="부지런한 거위",
    layout="wide",
)



# *중요! 각 csv파일은 무조건 data폴더 안에 있어야합니다.
def load_data1(df1=None, split=True):
   
    if df1 is None: 
        df1 = pd.read_csv('./data/Regression_data.csv')

    num_cols = list(df1.drop(columns=['Sex', 'Rings']))
    cat_cols = ['Sex']
    X = df1.drop('Rings', axis=1)
    y = df1.Rings

    if split:
        data1_ct = make_column_transformer(
            (OneHotEncoder(sparse_output=False), cat_cols),
            (StandardScaler(), num_cols),
            remainder='passthrough'
        )    
        data1_ct.set_output(transform='pandas')
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=.25,
                                                            random_state=42)
        
        X_train = data1_ct.fit_transform(X_train)
        X_test = data1_ct.transform(X_test)
        return X_train, X_test, y_train, y_test, data1_ct
    else:
        return X, y, df1
    

def load_data2(df2=None, split=True):
   
    if df2 is None:
        df2 = pd.read_csv('./data/binary_classification_data.csv')

    data2_ct = StandardScaler()
    data2_ct.set_output(transform='pandas')

    X = df2.drop(columns='target_class')
    y = df2.target_class

    if split:
        X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                            test_size=.25,
                                                            stratify=y,
                                                            random_state=42)
        
        X_train = data2_ct.fit_transform(X_train)
        X_test = data2_ct.transform(X_test)
        return X_train, X_test, y_train, y_test, data2_ct
    else:
        return X, y, df2

def load_data3(df3=None, split=True):
   
    if df3 is None:
        df3 = pd.read_csv('./data/mulit_classification_data.csv')

    
    y = df3.loc[:,'Pastry':]
    X = df3.drop(columns=y.columns, axis=1)
    y['label'] = np.argmax(y.values, axis=1) # label구분을 위한 컬럼추가
    if not split:
        return X, y, df3
    
    X = X.drop(columns='TypeOfSteel_A400')

    # 범주형 : TypeOfSteel_A300, Outside_Global_Index
    cat_cols = ['TypeOfSteel_A300', 'Outside_Global_Index']
    num_cols = list(X.drop(columns=cat_cols))

    # 범주형 ohe, 나머지 표준화
    data3_ct = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(sparse_output=False, drop='if_binary'), cat_cols),
        remainder='passthrough'
    )
    data3_ct.set_output(transform='pandas')

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        stratify=y['label'],
                                                        random_state=42)
    # 독립변수 28개, 종속변수 7개
    X_train = data3_ct.fit_transform(X_train)
    X_test = data3_ct.transform(X_test)
    return X_train, X_test, y_train, y_test, data3_ct


# 각 모델 클래스는 위에서 선언한 model을 상속받아 필요한것만 오버라이딩
class Model:
    model_num = None # load_data호출시 모델번호. 
    
    def __init__(self):
      
        X, y, _ = self.load_data(split=False)
        self.params = {}
        self.model = None
        self.ct = None
        self.features = list(X)
        self.target = [y.name] if isinstance(y, pd.Series) else list(y)

        if self.model_num == 1:
            name = 'reg'
        elif self.model_num == 2:
            name = 'bin'
        elif self.model_num == 3:
            name = 'multi'
        else:
            raise ValueError(f'model_num 이상. 입력값 : {self.model_num}')
            
        # 먼저 전처리기 불러오기
        file_path_1 = './data/'+name+'_ct.pkl'
        try:
            self.ct = joblib.load(file_path_1)
            print('전처리기 불러오기 성공')
        except:
            print('전처리기 불러오기 실패.. 학습진행')
            self.ct = self.load_data()[-1]

        # 모델 불러오기
        file_path_model = './data/'+name+'_BestModel.pkl'
        try:
            self.model = joblib.load(file_path_model)
            self.params = self.model.get_params()
            print('모델 불러오기 성공')
        except:
            print('모델 불러오기 실패.. 학습진행')
            self.model, train_time = self.train()
            self.params = self.model.get_params()

            # 학습 소요시간 print. 0.1초보다 작으면 ms단위로
            if train_time < 0.1:
                print(f'모델학습 완료. 소요시간 {train_time:.2f}')
            else:
                print(f'모델학습 완료. 소요시간 {train_time*1000:.2f}ms')




    # 각 모델에 맞는 데이터를 불러오는 함수
    @classmethod
    def load_data(cls, df=None, split=True):
        
        if cls.model_num == 1:
            return load_data1(df1=df, split=split)
        elif cls.model_num == 2:
            return load_data2(df2=df, split=split)
        elif cls.model_num == 3:
            return load_data3(df3=df, split=split)
        else:
            raise ValueError(f'model_num 이상. 입력값 : {cls.model_num}')
            

    # 아예 파라미터를 새로 설정하려면 self.params = {..}
    # 기존의 파라미터를 업데이트 하려면 self.update_params(param1=값1, param2=값2..)
    def update_params(self, **params):
        self.params.update(params)
        self.model.set_params(**self.params)
    

    def _to_frame(self, data, target=False):
       
    
        # 이미 DataFrame이면 그대로 반환
        if isinstance(data, pd.DataFrame):
            return data

        def transform(data):
            # X가 DataFrame이 아니면 DataFrame으로 최종변환
            if not isinstance(data, pd.DataFrame):
                # 리스트나 튜플이면 np.ndarray로 변환
                if isinstance(data, (list|tuple)):
                    data = np.array(data)

                if isinstance(data, np.ndarray):
                    # [data1, data2, ..] 이면 [[data1, data2.. ]]로 변환
                    if data.ndim == 1:
                        data = data.reshape(1,-1)
                    elif data.ndim !=2:
                        print(f'입력데이터의 차원이 이상합니다. {data.ndim}차원이 아닌 1차원 혹은 2차원이 되어야합니다.')
                        return None
                    # DataFrame으로 변환
                    data = pd.DataFrame(data)

                if isinstance(data, pd.Series):
                    data = data.to_frame().T
            return data

        data = transform(data=data)
        if data is None:
            return None
       
        if target: 
            # 컬럼숫자가 안맞으면 에러메시지 출력하고 None 리턴
            if data.shape[1] != len(self.target):
                print(f'타겟데이터의 컬럼수가 안맞습니다. {data.shape[1]}개가 아닌 {len(self.features)}개가 되어야합니다.')
                return None
            data.columns = self.target
            return data # y
        else:
            if data.shape[1] != len(self.features):
                print(f'피쳐데이터의 컬럼수가 안맞습니다. {data.shape[1]}개가 아닌 {len(self.features)}개가 되어야합니다.')
                return None
            data.columns = self.features
            return data # X


    def train(self, params=None, df=None):
       

        model = XGBRegressor(random_state=42,
                             objective='reg:squarederror',
                             tree_method='hist',
                             gpu_id=0,
                             eval_metric='rmse',
                             early_stopping_rounds=30)
        
        model.set_params(**params)
        
        # 데이터셋 준비
        if df is None:
            X_train, X_test, y_train, y_test = self.load_data()[:4]
        else:
            X_train, X_test, y_train, y_test = self.load_data(df=df)[:4]
    
        # 학습
        t1 = time.time()
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=0)
        t2 = time.time()
        train_time = t2-t1
        
        # 모델과 훈련 소요시간 반환
        return model, train_time
    

    def predict(self, X):
        
        if not isinstance(X, pd.DataFrame):
            X = self._to_frame(X, target=False)
            if X is None:
                return None
        
        X = self.ct.transform(X)
        pred = self.model.predict(X)

        return pred
    

    def performance(self, df=None):
       
        if df:
            # array-like이면 dataframe으로 변환
            if isinstance(df, (np.ndarray|list|tuple)):
                df = pd.DataFrame(df, columns=self.features+self.target)
            
            X_test = df.drop(columns=self.target)
            y_true = df[self.target]
        else:
            _, X_test, _, y_true, _ = self.load_data()
        
        y_pred = self.model.predict(X_test)

        if isinstance(self.model, XGBRegressor):
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            acc = np.mean(1-abs((y_pred - y_true) / y_true))
            r2 = r2_score(y_true, y_pred)
            return rmse, acc, r2
        elif isinstance(self.model, XGBClassifier):
            precision = precision_score(y_true, y_pred, pos_label=1)
            recall = recall_score(y_true, y_pred, pos_label=1)
            f1 = f1_score(y_true, y_pred, pos_label=1)
            acc = accuracy_score(y_true, y_pred)
            return precision, recall, f1, acc


    def save_model(self):
        """
        모델을 저장하는 함수
        """
        
        if self.model_num == 1:
            file_path_2 = './data/reg_BestModel.pkl'
        elif self.model_num == 2:
            file_path_2 = './data/bin_BestModel.pkl'
        elif self.model_num == 3:
            file_path_2 = './data/multi_BestModel.pkl'
        else:
            raise ValueError(f'model_num 이상. 값 {self.model_num}')
            
        joblib.dump(self.model, file_path_2)


    def make_val_data(self, n_samples:int=100, seed:int=None):
       
        val_df = self.load_data(split=False)[2] # df
        
        if seed:
            val_df = val_df.sample(n_samples, random_state=seed)
        else:
            val_df = val_df.sample(n_samples)
        
        return val_df
        
        
class Model1(Model):
    model_num = 1
    
    # train만 오버라이딩
    def train(self, params=None, df=None):
        # params 들어온게 없으면, 최적파라미터로 초기설정 (rmse ~ 2.12)
        if params is None:
            params = {
                'learning_rate': 0.02087425763287998,
                'n_estimators': 1550,
                'max_depth': 17,
                'colsample_bytree': 0.5,
                'reg_lambda': 10.670146505870857,
                'reg_alpha': 0.0663394675391197,
                'gamma': 9.015017136084957
            }
        return super().train(params, df)


class Model2(Model):
    model_num = 2

    # train만 오버라이딩
    def train(self, params=None, df=None):
        # params 들어온게 없으면, 최적파라미터로 초기설정 (f1 ~ 0.871)
        # 현재 튜닝성능이 안나와서 파라미터 개선 예정입니다!
        if params is None:
            params = {
                'learning_rate': 0.03233685808565227,
                'n_estimators': 1200,
                'max_depth': 20,
                'colsample_bytree': 0.5,
                'reg_lambda': 0.004666963217784473,
                'reg_alpha': 0.002792083422830542,
                'gamma': 0.036934880241175236,
                'scale_pos_weight': 7.0
            }
        return super().train(params, df)
    

class Model3(Model):
    model_num = 3

    def __init__(self):
        super().__init__()
        # self.model = Other_Faults 이진분류 모델
        # self.model2 = 나머지 6개 다중분류 모델
        self.params2 = None
        self.model2 = None

        # 모델2 불러오기
        file_path_model2 = './data/multi_BestModel2.pkl'
        try:
            self.model2 = joblib.load(file_path_model2)
            self.params2 = self.model2.get_params()
            print('모델2 불러오기 성공')
        except:
            print('모델2 불러오기 실패.. 학습진행')
            self.model2, train_time = self.train(case=2)
            self.params2 = self.model2.get_params()

            # 학습 소요시간 print. 0.1초보다 작으면 ms단위로
            if train_time < 0.1:
                print(f'모델학습 완료. 소요시간 {train_time:.2f}')
            else:
                print(f'모델학습 완료. 소요시간 {train_time*1000:.2f}ms')


    def train(self, case=1, params=None, df=None):
        """
        case = 1이면 1번모델 학습 후 반환, 2이면 2번모델 반환
        """
        # 데이터셋 준비
        if df is None:
            X_train, X_test, y_train, y_test = self.load_data()[:4]
        else:
            X_train, X_test, y_train, y_test = self.load_data(df=df)[:4]
        
        # 이진분류 모델
        if case == 1:
            if params is None:
                params = {
                    'learning_rate': 0.012202469692864067,
                    'n_estimators': 700,
                    'max_depth': 11,
                    'colsample_bytree': 0.7,
                    'reg_lambda': 0.6998936289657887,
                    'reg_alpha': 0.07423629936782049,
                    'gamma': 0.04343495370664839,
                    'scale_pos_weight': 1.2
                }
            X_trainA = X_train
            X_testA = X_test
            y_trainA = y_train.Other_Faults
            y_testA = y_test.Other_Faults

            model = XGBClassifier(random_state=42,
                                objective='binary:logistic',
                                eval_metric='error',
                                tree_method='gpu_hist',
                                gpu_id=0,
                                early_stopping_rounds=30)
            model.set_params(**params)

            t1 = time.time()
            model.fit(X_trainA, y_trainA,
                      eval_set=[(X_testA, y_testA)],
                      verbose=0)
            t2 = time.time()
            train_time = t2-t1
            return model, train_time
        
        # 다중분류 모델
        elif case == 2:
            if params is None:
                params = {
                    'learning_rate': 0.07522487380833985,
                    'n_estimators': 250,
                    'max_depth': 4,
                    'colsample_bytree':0.6,
                    'reg_lambda': 0.001648272236870337,
                    'reg_alpha': 0.01657588037413299,
                    'gamma': 0.002792373320363197
                }
            y_trainB = y_train[y_train.Other_Faults == 0].label
            y_testB = y_test[y_test.Other_Faults == 0].label
            X_trainB = X_train.loc[y_trainB.index]
            X_testB = X_test.loc[y_testB.index]

            # 다중분류는 scale_pos_weight 설정이 불가함으로 sklearn을 활용,
            # sample weight로 weight 설정
            weight = compute_sample_weight(class_weight='balanced', y=y_trainB)

            model2 = XGBClassifier(random_state=42,
                                   objective='multi:softprob',
                                   eval_metric='aucpr',
                                   tree_method='gpu_hist',
                                   gpu_id=0,
                                   early_stopping_rounds=30,
                                   num_class=6)
            model2.set_params(**params)

            t1 = time.time()
            model2.fit(X_trainB, y_trainB,
                       eval_set=[(X_testB, y_testB)],
                       sample_weight=weight,
                       verbose=0)
            t2 = time.time()
            train_time = t2-t1
            return model2, train_time
    

    def predict(self, X, th=0.496, name_out=True):
        """
        기존 함수와 차이

        th : 이진분류 threshold (기본값 최적)

        name_out : 이름으로 출력할지 라벨로 출력할지
        """
        

        if not isinstance(X, pd.DataFrame):
            X = self._to_frame(X, target=False)
            if X is None:
                return None
        else:
            X = X.copy()

        if 'TypeOfSteel_A400' in X:
            X.drop(columns='TypeOfSteel_A400')
        for col in self.target:
            if col in X:
                X.drop(columns=col, inplace=True)

        try:
            X = self.ct.transform(X)
        except:
            pass
        # 먼저 Other_Faults인지 아닌지 이진분류
        pred = np.where(self.model.predict_proba(X)[:,1] >= th, 6, 0)
        pred = pd.Series(pred, index=X.index, name='label')

        # Other_Faults가 아니라고 예측한 항목들 다중분류
        idx = pred[pred == 0].index
        pred2 = self.model2.predict(X.loc[idx])
        pred2 = pd.Series(pred2, index=idx, name='label')

        # 다중분류한 라벨 원래 pred에 업데이트
        pred.update(pred2)

        if name_out:
            # 라벨을 결함이름으로 변환
            pred = pred.map(dict(zip([0,1,2,3,4,5,6], self.target)))

        return pred
        
    
    def performance(self, df=None):
        """
        모델의 성능을 평가하는 함수.  

        df가 있으면 해당데이터로, 없으면 기본 검증데이터로 평가
        
        args:
            df : 검증 데이터프레임
        
        returns:
            macro precision, macro recall, macro f1, acc : 이진분류의 경우
        """
        if df is not None:
            # array-like이면 dataframe으로 변환
            if isinstance(df, (np.ndarray|list|tuple)):
                df = pd.DataFrame(df, columns=self.features+self.target)
            
            print(list(df))
            X_test, y_true, _ = self.load_data(df=df, split=False)
            y_true = y_true.label
        else:
            _, X_test, _, y_true, _ = self.load_data()
            y_true = y_true.label
        
        y_pred = self.predict(X_test, name_out=False)

        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        return precision, recall, f1, acc
    

    def save_model(self):
        super().save_model()
        joblib.dump(self.model2, './data/multi_BestModel2.pkl')
        

# # 웹과의 상호작용을 하는 함수
# def main():
#     """동작순서 간단히
#     1. 데이터 불러오기
#     2. 모델 인스턴스화
#     3. 루프로 상호작용 반복
#     4. 특정 조건이 되면 종료
#     """
    
#### 사용예시 ####
# 모델 인스턴스 생성

st.subheader("""
예측 모델 
""")

tab1, tab2 = st.tabs(["전복나이예측", "펄서여부예측"])

with tab1:

    st.subheader('전복크기예측')
    model1 = Model1()
    # 검증용 데이터 생성
    val_df1 = model1.make_val_data(10)
    # 검증데이터로 예측해보기
    # model1.predict(val_df1.drop(columns='Rings'))
    # 입력 데이터로 예측해보기
    col1, col2 = st.columns(2)
    with col1:
        input_sex = st.selectbox('성별', ['F', 'M','I'])
        input_length = st.slider('Length',0.01, 1.0, 0.01)
        input_diameter = st.slider('Diameter', 0.01, 1.0, 0.01)
        input_height = st.slider('Height', 0.01, 1.0, 0.01)
        input_wholeWeight = st.slider('Whole weight', 0.01, 1.0, 0.01)
        input_shuckedWeight = st.slider('Shucked weight', 0.01, 1.0, 0.01)
        input_visceraWeight = st.slider('Viscera weight', 0.01, 1.0, 0.01)
        input_shellWehgit = st.slider('Shell weight', 0.01, 1.0, 0.01)
    
    inputs = [input_sex, 
              input_length, 
              input_diameter, 
              input_height, 
              input_wholeWeight, 
              input_shuckedWeight, 
              input_visceraWeight, 
              input_shellWehgit
              ]
    result = model1.predict(inputs)

    with col2:
        st.write('   ')
        st.write('   ')
        st.write('#### 예측한 크기 :', result[0])
    
    # 성능확인
    rmse, acc, r2 = model1.performance()
    st.write(f'rmse : {rmse:.3f}\tacc : {acc:.3f}\tr2 : {rmse:.3f}')


with tab2:

    model2 = Model2()
    val_df2 = model2.make_val_data(50)

    st.subheader('펄서여부 예측')

    col1, col2 = st.columns(2)
    with col1:
        input_0 = st.slider('Mean of the integrated profile', 6.0, 190.0, 1.0)
        input_1 = st.slider('Standard deviation of the integrated profile',24.0, 100.0, 1.0)
        input_2 = st.slider('Excess kurtosis of the integrated profile', -2.0, 10.0, 0.1)
        input_3 = st.slider('Skewness of the integrated profile', -2.0, 70.0, 1.0)
        input_4 = st.slider('Mean of the DM-SNR curve ', 0.0, 220.0, 1.0)
        input_5 = st.slider('Standard deviation of the DM-SNR curve', 7.0, 110.0, 1.0)
        input_6 = st.slider('Excess kurtosis of the DM-SNR curve', -4.0, 40.0, 1.0)
        input_7 = st.slider('Skewness of the DM-SNR curve', 0.0, 122.0, 10.0)

    inputs = [input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7]
    result = model2.predict(inputs)

    with col2:
        st.write('#### 펄서 여부 (1 = 펄서) :', result[0])

# with tab3:
#     model3 = Model3()
#     val_df3 = model3.make_val_data(50)
#     # st.write('모델3 검증세트 추론 :',list(model3.predict(val_df3)))

#     st.subheader('스테인레스 결함 예측')
#     st.write("""
#     ### 모델3 기본성능 확인
#     """)
#     precision, recall, f1, acc = model3.performance()
#     st.write(f'precision : {precision:.3f}\trecall : {recall:.3f}\tf1 : {f1:.3f}\tacc : {acc:.3f}')

#     st.write("""
#     ### 모델3 랜덤검증성능 확인
#     """)
#     precision, recall, f1, acc = model3.performance(df=val_df3)
#     st.write(f'precision : {precision:.3f}\trecall : {recall:.3f}\tf1 : {f1:.3f}\tacc : {acc:.3f}')


    
# # py 실행하면 main()실행
# if __name__ == '__main__':
#     main()
#     # main() 종료 이후 서버에 로그 남기기? (여유되면)