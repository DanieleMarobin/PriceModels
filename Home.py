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
    if ('all_options' not in st.session_state):
        st.session_state['all_options']=True

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

# Filters and Settings
if True:
    st.sidebar.button('Get Latest File', on_click=func_reset)

    with st.sidebar.form('run'):        
        options = list(model_df.columns)
        # options.sort()
        y_col = st.selectbox('Variable to Model',options, options.index('a_price_c '))

        # Top N Variables
        top_n_vars = st.number_input('Top N Variables',1,10000,20,1)

        # Form Submit Button
        st.form_submit_button('Apply',on_click=apply_func)

    font_size = st.sidebar.number_input('HeatMap Font Size',5,20,10,1)
    chart_height = st.sidebar.number_input('HeatMap Chart Height',100,10000,750,100)
    st.sidebar.markdown('---')
    scatter_type = st.sidebar.radio('Scatter Color',('Categorical','Continuous'))
    trendline_scope = st.sidebar.radio('Trendline',('overall','trace', None))
    
    colors=list(color_scales.keys())
    colors.sort()
    chart_color_key = st.sidebar.selectbox('Chart Color',colors, colors.index('Plotly-qualitative')) # Plotly-qualitative, Jet, RdYlGn-diverging
    color_list=color_scales[chart_color_key]

# Heat-Map
if True:
    mask=(model_df.index.month>0)
    months=list(range(1,13))

    if ('heat_map_months' in st.session_state):
        # st.write('Getting the Monthly heatmap from memory')
        heat_map_months=st.session_state['heat_map_months']
    else:
        with st.spinner('Calculating the Monthly heatmap for top '+str(top_n_vars) + ' variables...'):
            st.session_state['heat_map_months']=mp.heat_map_var_months(model_df[mask], y_cols=[y_col], top_n = top_n_vars, months=months, parallel=None, show=False)[y_col]
            heat_map_months=st.session_state['heat_map_months']

    abs_max=heat_map_months['value'].abs().max() # so the positives are Blue and the negatives are red

    fig=uc.chart_heat_map(heat_map_months,title=y_col, x_col='report',y_col='v1',z_col='value', sort_by='all_abs', transpose=False, color_continuous_scale=color_scales['RdBu-sequential'], range_color=(-abs_max,abs_max), format_labels = '%{z:.1f}') #,tickangle=-90

    fig.update_layout(coloraxis_showscale=False, xaxis=dict(titlefont=dict(size=font_size),tickfont=dict(size=font_size),side='top'),yaxis=dict(titlefont=dict(size=font_size),tickfont=dict(size=font_size)))
    
    selected_points = plotly_events(fig, override_height=chart_height)
    
# Scatter Plots
if True:
    if len(selected_points)==0: st.stop()

    x=selected_points[0]['y']        

    mask=(model_df.index.month>0)
    df=model_df[mask]

    month_col='month'
    df[month_col]=df.index.month
    
    col1, col2 = st.columns([1,1])
    with col1:        
        st.markdown('#### Month Split')
        st.markdown('---')        
        all_options = st.checkbox("All Months", True)

        options=list(set(df[month_col].astype('int')))
        options.sort()

        if all_options:
            sel_months = st.multiselect( 'Months', options, options)
        else:
            sel_months = st.multiselect( 'Months', options)

    with col2:
        crop_year_col='wasde_us corn nc-crop year-'
        st.markdown('#### Crop Year Split')
        st.markdown('---')
        all_options = st.checkbox("All Years", True)

        options=list(set(df[crop_year_col].astype('int')))
        options.sort()

        if all_options:
            sel_years = st.multiselect('Years', options, options)
        else:
            sel_years = st.multiselect('Years', options)

    col1, col2 = st.columns([1,1])
    if ((len(sel_months)==0) | (len(sel_years)==0)): st.stop()

    mask = np.isin(df[month_col].astype('int'), sel_months)
    mask = ((mask) & (np.isin(df[crop_year_col].astype('int'),sel_years)))
    df=df[mask]

    if scatter_type=='Categorical':
        as_type='str'
    else:
        as_type='int'

    color_var_month=df[month_col].astype(as_type)
    color_var_year=df[crop_year_col].astype(as_type)


    # Chart Labels
    cols_with_none = ['None','year','report']
    cols_with_none.extend(df.columns)
    chart_labels = st.sidebar.selectbox('Chart Labels',cols_with_none, cols_with_none.index('None'))

    if chart_labels=='None':
        chart_labels=None
    elif chart_labels=='year':
        chart_labels=df.index.year
    elif chart_labels=='report':
        chart_labels=df.index

    with col1:        
        fig=px.scatter(df,x=x,y=y_col,color=color_var_month, text=chart_labels, trendline='ols',trendline_scope=trendline_scope, color_discrete_sequence=color_list, color_continuous_scale=color_list)
        fig.update_traces(textposition='top center')
        fig.update_layout(legend_title_text=None,coloraxis_colorbar=dict(title=None))

        st.plotly_chart(fig)
        all_models=px.get_trendline_results(fig)
        if len(all_models)>0:        
            all_models=all_models.px_fit_results
            legend_items=[]
            for trace in fig.data:
                if 'OLS trendline' in trace.hovertemplate:
                    legend_items.append(trace.name)

            for i, tm in enumerate(all_models): # tm: Trendline Model
                st.write(legend_items[i], 'R-Squared:', str(round(tm.rsquared,3)))

    with col2:            
        fig=px.scatter(df,x=x,y=y_col,color=color_var_year, text=chart_labels, trendline='ols',trendline_scope=trendline_scope, color_discrete_sequence=color_list, color_continuous_scale=color_list)
        fig.update_traces(textposition='top center')
        fig.update_layout(legend_title_text=None,coloraxis_colorbar=dict(title=None))
        fig.update_layout()

        st.plotly_chart(fig)
        all_models=px.get_trendline_results(fig)
        if len(all_models)>0:        
            all_models=all_models.px_fit_results
            legend_items=[]
            for trace in fig.data:
                if 'OLS trendline' in trace.hovertemplate:
                    legend_items.append(trace.name)

            for i, tm in enumerate(all_models): # tm: Trendline Model
                st.write(legend_items[i], 'R-Squared:', str(round(tm.rsquared,3)))
