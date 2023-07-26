import pandas as pd
import pygwalker as pyg
import streamlit as st


st.set_page_config(
    page_icon=":goose:",
    page_title="부지런한 거위",
    layout="wide",
)

data = pd.read_csv('./csv/binary_classification_data.csv')
walker = pyg.walk(data, env='Streamlit')