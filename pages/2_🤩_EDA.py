import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import sklearn


# data
df = pd.read_csv('./csv/Regression_data.csv')
df2 = pd.read_csv('./csv/binary_classification_data.csv')
df3 = pd.read_csv('./csv/multi_classification_data.csv')

st.set_page_config(
    page_icon=":goose:",
    page_title="부지런한 거위들",
    layout="wide",
)

st.header("""
 데이터 셋 EDA
""")


tab1, tab2, tab3 = st.tabs(["전복나이예측", "펄서여부예측", "스테인레스결함예측"])

with tab1:

    st.write('### 전복나이예측 (Regression_data)')
    st.write("""
    - 4177개의 데이터, 결측치, 중복값 없음, ‘sex’ 컬럼이 범주형 데이터
    - 데이터 분포 히스토그램 : 대부분 정규 분포, 약하게 skewed 분포된 특성들 있음
    - 산점도 : 범주형 데이터인 성별을 제외하고, 모두 타겟 변수에 양의 영향
    - 상관관계분석 : 다중공선성 문제 심각.
    - 다중공선성(VIF) 검증 : Height 특성은 3.6, 그 외 특성은 17.5~109.8의 높은 다중공선성을 보임
    - 박스플롯 : 데이터 이상치 문제 거의 없음
    """)
    select_graph = st.selectbox('Select a graph', ['hist', 'box plot','산점도','상관관계'])
    st.write('graph type : ', select_graph)
    
    if select_graph == 'hist':
        def hist(column):
            plt.hist(column, bins=50)
            plt.title(f'{column.name} hist')
            plt.show()

        df.select_dtypes(include=[float, int]).hist(bins=30, figsize=(14,8), grid=False)
        plt.tight_layout()
        st.pyplot(plt)

    elif select_graph == 'box plot':
        plt.figure(figsize=(12,6))
        df.drop(columns='Rings').boxplot()
        plt.title('boxplot of regression data', fontsize=14)
        st.pyplot(plt)

    elif select_graph == '산점도':
        # 산점도
        plt.subplots(2,4, figsize=(15,8))
        for idx, col in enumerate(df.drop(columns='Rings').columns):
            plt.subplot(2, 4, idx+1)
            plt.scatter(df[col], df.Rings)
            plt.title(col)
            plt.ylabel('Rings')
        st.pyplot(plt)

    elif select_graph == '상관관계':
        # df.Height.sort_values(ascending=False)[:5]
        sns.heatmap(df.iloc[:,1:].corr().abs(), annot=True, cmap='Blues')
        plt.title('Pearson Corr', fontsize=15)
        st.pyplot(plt)
    
    

with tab2:

    st.write('### 펄서여부예측 (Binary_classification_data)')
    st.write("""
    - row 17898개, 범주형 데이터, 중복, 결측치 없음
    - 이상치 있는 컬럼 3개
    - 타겟 클래스 비율이 9:1로 매우 심각함
    - 데이터 분포 히스토그램 : 전체적으로 skewed data
    - 상관관계분석 : 2개 세트의 특성들이 상관계수가 높으나, 나머지는 높지 않음
    - 바이올린 플롯 : 특성별 클래스간 차이가 유의미하게 나타남 => 이상치가 오히려 클래스 간의 유의미한 차이를 만들고 있음
    """)
    select_graph_2 = st.selectbox('select a graph', ['hist', '바이올린 플롯','상관관계'])
    
    st.write('graph type : ', select_graph_2)

    if select_graph_2 == 'hist':
        df2.iloc[:,:-1].hist(bins=30, figsize=(14,8), grid=False)
        st.pyplot(plt)
        st.write('3개 컬럼 이상치 제거 이후 분포 확인')
        cols = df2.iloc[[0],[2,3,4]].columns
        df2[(df2[cols[0]] < 2) & (df2[cols[1]] < 15) & (df2[cols[2]] < 50)][cols].hist(bins=30, figsize=(10,6), grid=False)
        st.pyplot(plt)

    if select_graph_2 == '바이올린 플롯':
        plt.figure(figsize=(14,12))
        for idx, col in enumerate(df2.drop(columns='target_class')):
            plt.subplot(3,3,idx+1)
            sns.violinplot(x=df2['target_class'], y=df2[col], palette=['red', 'lime'], )
            plt.axhline(df2[col].mean(), linestyle='dashed', label=f'mean {df2[col].mean():.2f}')
            plt.ylabel('')
            plt.xlabel('')
            plt.legend(loc='best')
            plt.title(col, fontsize=10)
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        st.pyplot(plt)

    if select_graph_2 == '상관관계':
        sns.heatmap(df2.corr().abs(), annot=True, cmap='Blues', fmt='.2f')
        plt.title('Pearson Corr', fontsize=15)
        st.pyplot(plt)

    
with tab3:

    st.write('### 스테인레스결함예측 (Multi_classification_data)')
    st.write("""
    - 데이터 1941개, 중복, 결측치 없음
    - target변수 7개이며, class 비율이 2.8% 부터 34.7% 까지 존재
    - 데이터 분포 히스토그램 : skewed data가 꽤 많음
    - 상관관계 분석 : 상관관계 1인 컬럼들 존재
    - 바이올린 플롯 : 클래스간 유의마한 데이터 분포 차이 발견 X, 이상치가 다수 존재
    """)
    select_graph_3 = st.selectbox('select a graph', ['target_class rate', 'hist', '바이올린 플롯'])
    st.write('graph type : ', select_graph_3)

    if select_graph_3 == 'target_class rate':
        fig = plt.figure(figsize=(14,8))
        x = 0
        for i in df3.iloc[:,27:].columns:
            x += 1
            temp = df3[i].value_counts()
            ax = fig.add_subplot(240+x)
            ax.pie(temp, labels=[0,1], autopct='%1.1f%%')
            ax.set_title(i, fontsize=10)
        plt.tight_layout()
        st.pyplot(plt)

    if select_graph_3 == 'hist':
        df3.iloc[:,:27].hist(bins=30, figsize=(14,20), grid=False)
        st.pyplot(plt)

    if select_graph_3 == '바이올린 플롯':
        targets = df3.loc[:,'Pastry':'Other_Faults']
        dic = dict(zip([0,1,2,3,4,5,6], targets.columns))
        temp = df3.iloc[:,:27]
        temp['label'] = targets.apply(lambda x: np.argmax(x), axis=1)
        temp.label = temp.label.map(dic)

        plt.figure(figsize=(14,30))
        for idx, col in enumerate(temp.drop(columns='label')):
            plt.subplot(9,3,idx+1)
            sns.violinplot(x=temp.label, y=temp[col])
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('')
            plt.ylabel('')
            plt.title(col, fontsize=10)
            
        plt.subplots_adjust(wspace=0.2, hspace=0.7)
        st.pyplot(plt)
