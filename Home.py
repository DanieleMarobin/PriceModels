# Imports
if True:
    from datetime import datetime as dt

    import numpy as np
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

# Data
if True:
    df_model_all=fu.get_data()
    model_df_instr=mp.model_df_instructions(df_model_all)
    model_df = mp.from_df_model_all_to_model_df(df_model_all, model_df_instr)
    # st.dataframe(model_df)

# Filters and Settings
if True:
    st.sidebar.button('Get Latest File', on_click=func_reset)

    with st.sidebar.form('run'):        
        options = list(model_df.columns)
        # options.sort()
        y_col = st.selectbox('Variable to Model',options, options.index('a_price_c '))

        # Top N Variables
        top_n_vars = st.number_input('Top N Variables',1,100,20,1)

        # Form Submit Button
        st.form_submit_button('Apply',on_click=apply_func)

    font_size = st.sidebar.number_input('HeatMap Font Size',5,20,8,1)
    chart_height = st.sidebar.number_input('HeatMap Chart Height',100,10000,top_n_vars*50,10)

    trendline_scope = st.sidebar.radio('Trendline Scope',('overall','trace', None))


# Visualization
if True:
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

    fig=uc.chart_heat_map(heat_map_months,title=y_col, x_col='report',y_col='v1',z_col='value', sort_by='all', transpose=False, color_continuous_scale=color_scales['RdBu-sequential'], range_color=(-abs_max,abs_max), format_labels = '%{z:.1f}') #,tickangle=-90
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(xaxis=dict(titlefont=dict(size=font_size),tickfont=dict(size=font_size),side='top'))
    fig.update_layout(yaxis=dict(titlefont=dict(size=font_size),tickfont=dict(size=font_size)))

    selected_points = plotly_events(fig, override_height=chart_height)
    
# Scatter Plots
if len(selected_points)==0:
    st.stop()

print(selected_points)
x=selected_points[0]['y']        

mask=(model_df.index.month>0)
# mask=((model_df.index.month>=5) & (model_df.index.month<=7))

df=model_df[mask]
# df_check=pd.concat([df[y],df[v]],axis=1)

# fig=px.scatter(df_check,x=v,y=y, trendline='ols',text=df.index)
c=color_scales['Jet-sequential']

col1, col2 = st.columns([1,1])
with col1:
    st.markdown('#### Month Split')
    st.markdown('---')        
    all_options = st.checkbox("All Months",True)
    options=list(set(df.index.month.astype('str'))); options.sort()

    if all_options:
        sel_months = st.multiselect( 'Months', options, options)
    else:
        sel_months = st.multiselect( 'Months', options)

with col2:
    st.markdown('#### Year Split')
    st.markdown('---')
    all_options = st.checkbox("All Years",True)
    options=list(set(df.index.year.astype('str'))); options.sort()

    if all_options:
        sel_years = st.multiselect('Years', options, options)
    else:
        sel_years = st.multiselect('Years', options)

col1, col2 = st.columns([1,1])
if ((len(sel_months)==0) | (len(sel_years)==0)): st.stop()

mask = np.isin(model_df.index.month.astype('str'),sel_months)
mask = ((mask) & (np.isin(model_df.index.year.astype('str'),sel_years)))
df=model_df[mask]

with col1:        
    color_var=df.index.month.astype('str')
    fig=px.scatter(df,x=x,y=y_col,color=color_var, trendline='ols',trendline_scope=trendline_scope)
    st.plotly_chart(fig)

with col2:    
    color_var=df.index.year.astype('str')
    fig=px.scatter(df,x=x,y=y_col,color=color_var, trendline='ols',trendline_scope=trendline_scope) #, color_discrete_sequence=c, color_continuous_scale=c
    st.plotly_chart(fig)
