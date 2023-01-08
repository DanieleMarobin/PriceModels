from datetime import datetime as dt

import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events

import GDrive as gd

PM_DIR = 'Data/Models/Price Models/'

file_name = PM_DIR+'df_model_all.csv'

df_model_all=gd.read_csv(file_name,parse_dates=['date'], dayfirst=True, index_col='date')
# df_model_all=gd.read_csv(file_name)

st.dataframe(df_model_all)

df = px.data.medals_wide(indexed=True)
fig = px.imshow(df,color_continuous_scale='RdBu_r',text_auto=True)
selected_points = plotly_events(fig)

st.write(selected_points)