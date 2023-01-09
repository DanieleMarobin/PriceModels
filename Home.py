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
    import func as fu

# Functions
if True:
    def func_reset():
        if ('df_model_all' in st.session_state):
            del st.session_state['df_model_all']

    def apply_func():
        if ('heat_map_months' in st.session_state):
            del st.session_state['heat_map_months']        

# Preliminaries
if True:
    st.set_page_config(page_title='Price Models',layout='wide',initial_sidebar_state='expanded')
    st.markdown("### Price Models")
    st.markdown("---")
    st.sidebar.markdown("### Price Models")

    color_scales = uc.get_plotly_colorscales()


# Filters and Settings
if True:
    st.sidebar.button('Get Latest File', on_click=func_reset)

    with st.sidebar.form('run'):

        # Top N Variables
        top_n_vars = st.number_input('Top N Variables',1,100,20,1)

        # Form Submit Button
        st.form_submit_button('Apply',on_click=apply_func)

font_size = st.sidebar.number_input('Font Size',5,20,8,1)
chart_height = st.sidebar.number_input('Chart Height',100,2000,600,10)


# Data
if True:
    df_model_all=fu.get_data()
    model_df_instr=mp.model_df_instructions(df_model_all)
    model_df = mp.from_df_model_all_to_model_df(df_model_all, model_df_instr)
    # st.dataframe(model_df)

# Visualization
if True:
    y_col='a_price_c '
    # mask=((model_df.index.month<=4) | (model_df.index.month>=8))
    mask=(model_df.index.month>0)
    months=list(range(1,13))

    if ('heat_map_months' in st.session_state):
        # st.write('Getting the Monthly heatmap from memory')
        heat_map_months=st.session_state['heat_map_months']
    else:
        # st.write('Calculating the Monthly heatmap')
        st.session_state['heat_map_months']=mp.heat_map_var_months(model_df[mask], y_cols=[y_col], top_n = top_n_vars, months=months, parallel=None, show=False)[y_col]
        heat_map_months=st.session_state['heat_map_months']

    abs_max=heat_map_months['value'].abs().max() # so the positives are Blue and the negatives are red

    fig=uc.chart_heat_map(heat_map_months,x_col='report',y_col='v1',z_col='value', sort_by='all', transpose=False, color_continuous_scale=color_scales['RdBu-sequential'], range_color=(-abs_max,abs_max), format_labels = '%{z:.1f}') #,tickangle=-90
    fig.update_layout(coloraxis_showscale=False)
    
    fig.update_layout(xaxis=dict(titlefont=dict(size=font_size),tickfont=dict(size=font_size),side='top'))
    # fig.update_layout(xaxis=dict(side='top'))
    fig.update_layout(yaxis=dict(titlefont=dict(size=font_size),tickfont=dict(size=font_size)))

    # fig.update_layout(width=1500,height=1500)
    # st.plotly_chart(fig)
    selected_points = plotly_events(fig, override_height=chart_height)
    st.write(selected_points)