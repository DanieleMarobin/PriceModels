# Imports
if True:
    from datetime import datetime as dt

    import pandas as pd
    import streamlit as st
    import plotly.express as px
    from streamlit_plotly_events import plotly_events

    import Price_Models as mp
    import GDrive as gd
    import Charts as uc
    import numpy as np

# Preliminaries
if True:
    st.set_page_config(page_title='Price Models',layout='wide',initial_sidebar_state='expanded')
    color_scales = uc.get_plotly_colorscales()
    PM_DIR = 'Data/Models/Price Models/'
    file_name = PM_DIR+'df_model_all.csv'

# Data
if True:
    df_model_all=gd.read_csv(file_name,parse_dates=['date'], dayfirst=True, index_col='date')
    model_df_instr=mp.model_df_instructions(df_model_all)
    model_df = mp.from_df_model_all_to_model_df(df_model_all, model_df_instr)
    # st.dataframe(model_df)

# Visualization
if True:
    y_col='a_price_c '
    # mask=((model_df.index.month<=4) | (model_df.index.month>=8))
    mask=(model_df.index.month>0)
    months=list(range(1,13))
    heat_map_df=mp.heat_map_var_months(model_df[mask], y_cols=[y_col], top_n = 10, months=months, parallel=None, show=False)[y_col]

    c=color_scales['RdBu-sequential']

    abs_max=heat_map_df['value'].abs().max() # so the positives are Blue and the negatives are red
    # fig=uc.chart_heat_map(heat_map_df,x_col='report',y_col='v1',z_col='value', sort_by='all', transpose=True, color_continuous_scale=c, range_color=(-abs_max,abs_max), format_labels = '%{z:.1f}',tickangle=-90)
    fig=uc.chart_heat_map(heat_map_df,x_col='report',y_col='v1',z_col='value', sort_by='all', transpose=True, color_continuous_scale=c, range_color=(-abs_max,abs_max), format_labels = '%{z:.1f}') #,tickangle=-90
    fig.update_layout(coloraxis_showscale=False)
    font_size=8
    fig.update_layout(xaxis=dict(titlefont=dict(size=font_size),tickfont=dict(size=font_size)))
    fig.update_layout(yaxis=dict(titlefont=dict(size=font_size),tickfont=dict(size=font_size)))

    # fig.update_layout(width=1500,height=1500)
    # st.plotly_chart(fig)
    selected_points = plotly_events(fig, override_height=600)
    st.write(selected_points)