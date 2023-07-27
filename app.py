import streamlit as st
from pyparsing import empty

st.set_page_config(
    page_icon=":goose:",
    page_title="부지런한 거위",
    layout="wide",
)

empty1,col1,empty2 = st.columns([0.3,1.0,0.3])
empty1,col2,col3,empty2 = st.columns([0.3,0.5,0.5,0.3])
empty1,col4,col5,empty2 = st.columns([0.3,0.5,0.5,0.3])


with empty1 :
    empty() # 여백부분1

with col1 :
    st.header("""
            😎 부지런한 거위팀
            #### 부지런한 거위들의 달달한 데이터 블로그에 오신 걸 환영합니다!!! 
            """)
with col2:
    st.image('./img/pkh.png', width=200)
    st.write("""
    ## 박경훈
    """)
    st.write("""
    ### 카리스마 넘치는 코딩능력자인 팀장
    """)

with col3:
    st.image('./img/kky.png', width=200)
    st.write("""
    ## 강규욱
    """)
    st.write("""
    ### 잘 생기고 활기 넘치는 코딩네이터
    """)

with col4:
    st.image('./img/bny.png', width=200)
    st.write("""
    ## 배나연
    """)
    st.write("""
    ### 똑똑하고 시야가 넓은 초고수 기획자
    """)

with col5:
    st.image('./img/jys.png', width=200)
    st.write("""
    ## 조윤서
    """)
    st.write("""
    ### 통통하지만 팀원들의 능력을 끌어올리는 서포터
    """)

with empty2 :
    empty() # 여백부분1
