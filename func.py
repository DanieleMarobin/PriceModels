import streamlit as st
import GDrive as gd
import pandas as pd
import numpy as np
import plotly.express as px
import openai


def get_data():
    PM_DIR = 'Data/Models/Price Models/'
    file_name = PM_DIR+'df_model_all.csv'

    # Delete all the items in Session state (to save memory)
    # for key in st.session_state.keys():
    #     if ((key!='df_model_all') & (key!='heat_map_months')):
    #         del st.session_state[key]

    df_model_all=None
    if ('df_model_all' in st.session_state):
        with st.spinner('Getting data from Memory...'):
            df_model_all=st.session_state['df_model_all']
    else:
        with st.spinner('Getting data from Google Drive...'):
            st.session_state['df_model_all']=gd.read_csv(file_name,parse_dates=['date'], dayfirst=True, index_col='date')
            df_model_all=st.session_state['df_model_all']

    return df_model_all


def prepare_ChatGPT_selection_request(user_request, col_options=[]):
    # https://blog.devgenius.io/chatgpt-how-to-use-it-with-python-5d729ac34c0d
    fo='From the items in this python list '
    fo=fo+'['+ ','.join(col_options) +']'
    fo=fo+user_request
    fo=fo+' and put the result in a python list format'

    return fo

def ChatGPT_answer(request,key):
    # Define OpenAI API key 
    openai.api_key = key

    # Set up the model
    model_engine = "text-davinci-003"

    completion = openai.Completion.create(engine=model_engine, prompt=request,max_tokens=1024,n=1,stop=None,temperature=0.5)
    # completion = openai.Completion.create(engine=model_engine, prompt=request,max_tokens=10000,n=1,stop=None,temperature=0.5)
    response = completion.choices[0].text
    return(response)

def extract_cols_from_ChatGPT_answer(answer):
    answer=answer.replace('[','').replace(']','').replace('"','').replace("'",'').split(',')
    return [c.strip() for c in list(answer)]