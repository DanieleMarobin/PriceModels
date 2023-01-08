from datetime import datetime as dt

import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events

import GDrive as gd

PM_DIR = 'Data/Models/Price Models/'

st.write('hello')

gd.get_credentials()


df = px.data.medals_wide(indexed=True)
fig = px.imshow(df,color_continuous_scale='RdBu_r')

selected_points = plotly_events(fig)
# st.plotly_chart(fig)

st.write(selected_points)