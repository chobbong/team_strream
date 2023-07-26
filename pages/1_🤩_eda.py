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
 EDA of Data set 1,2,3
""")

st.sidebar.subheader("""
 EDA of Data set 1,2,3
""")

select_dataset = st.sidebar.selectbox('select a data-set', ['data-1', 'data-2','data-3'])

if select_dataset == 'data-1':

    st.write('### data-1 (Regression_data)')
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

if select_dataset == 'data-2':

    st.write('### data2 (Binary_classification_data)')
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



if select_dataset == 'data-3':

    st.write('### data3 (Multi_classification_data)')
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