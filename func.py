
import streamlit as st
import GDrive as gd
import pandas as pd
import numpy as np
import plotly.express as px


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