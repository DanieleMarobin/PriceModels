# Imports
if True:
    from datetime import datetime as dt

    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    # from streamlit_plotly_events import plotly_events

    import Price_Models as mp
    import GDrive as gd
    import Charts as uc
    import func as fu

# Functions
if True:
    def func_reset():
        if ('df_model_all' in st.session_state):
            del st.session_state['df_model_all']

    def func_calculate():
        if ('heat_map_months' in st.session_state):
            # x_cols
            del st.session_state['x_cols']
            del st.session_state['heat_map_months']

# Preliminaries
if True:    
    st.set_page_config(page_title='Price Models',layout='wide',initial_sidebar_state='expanded')
    st.markdown("### Price Models")
    st.markdown("---")
    
    st.sidebar.markdown("### Price Models")
    color_scales = uc.get_plotly_colorscales()
    crop_year_col='wasde_us corn nc-crop year-'
    
# Data
if True:
    df_model_all=fu.get_data()
    model_df_instr=mp.model_df_instructions(df_model_all)
    model_df = mp.from_df_model_all_to_model_df(df_model_all, model_df_instr)

# Filters and Settings
if True:
    col1, col2, col3, col4 = st.columns([1,3,0.5,0.5])
    with col1:
        options = list(model_df.columns)
        y_col = st.selectbox('Target',options, options.index('a_price_c '))

    with col2:
        special_vars=['All','All-Stock to use','All-Ending Stocks','All-Yields']
        options=special_vars[:]
        options=options+list(model_df.columns)
        x_cols = st.multiselect('Selected Variables', options, ['All-Stock to use'])
        
        if 'All' in x_cols:
            x_cols=x_cols+list(model_df.columns)
        if 'All-Stock to use' in x_cols:
            x_cols=x_cols+[c for c in model_df.columns if 'stock to use' in c]
        if 'All-Ending Stocks' in x_cols:
            x_cols=x_cols+[c for c in model_df.columns if 'ending stock' in c]
        if 'All-Yields' in x_cols:
            x_cols=x_cols+[c for c in model_df.columns if 'yield' in c]            

        x_cols = list(set(x_cols)-set(special_vars))
        with st.expander(str(len(x_cols)) + ' of ' + str(len(model_df.columns))):
            st.write(x_cols)

        # Remove the 'special_vars'
        x_cols=x_cols+[y_col, crop_year_col] # Adding the trend
        x_cols = list(set(x_cols)-set(special_vars))

    with col3:
        top_n_vars = st.number_input('Top N Variables',1,10000,20,1)

    with col4:
        st.markdown('##')
        st.button('Calculate', on_click=func_calculate)

    st.markdown('---')

    with st.sidebar:
        st.button('Get Latest File', on_click=func_reset)

        with st.expander('Scatter Matrix Settings'):
            sm_height = st.number_input('Scatter Matrix Height',100,100000,750,100)
            sm_x_diagonal = st.checkbox('X Axis on diagonal',True)
            sm_offset = st.number_input('Scatter Matrix Labels Offset',0,1000,50,10)
            sm_font_size = st.number_input('Scatter Matrix Font Size',5,20,12,1)

        with st.expander('Heat Map Settings'):
            hm_height = st.number_input('HeatMap Chart Height',100,100000,750,100)
            hm_font_size = st.number_input('HeatMap Font Size',5,20,10,1)
        
        with st.expander('Scatter Plots Settings'):
            scatter_type = st.radio('Scatter Color',('Categorical','Continuous'))
            trendline_scope = st.radio('Trendline',('overall','trace', None))
            
            chart_labels=st.container()

            colors=list(color_scales.keys())
            colors.sort()
            chart_color_key = st.selectbox('Chart Color',colors, colors.index('Plotly-qualitative')) # Plotly-qualitative, Jet, RdYlGn-diverging
            color_list=color_scales[chart_color_key]

# Get selected Variables (x_cols)
if True:
    if ('x_cols' in st.session_state):
        x_cols=st.session_state['x_cols']
        model_df=model_df[x_cols]
    else:
        model_df=model_df[x_cols]

# Calcute the top N Variables
if True:
    if ('x_cols' in st.session_state):
        x_cols=st.session_state['x_cols']
    else:
        with st.spinner('Calculating the top '+str(top_n_vars) + ' variables...'):
            rank_df=mp.sorted_rsquared_var(model_df=model_df, y_col=y_col, n_var=1)            
            x_cols = rank_df['v1']
            x_cols=list(x_cols[0:top_n_vars])            

            # Adding cols needed to function
            x_cols=[y_col]+x_cols
            st.session_state['x_cols']=x_cols
    
    model_df=model_df[x_cols]

# Scatter Matrix
if True:
    df=model_df[:]
    # df['variables']=np.nan
    # sort_cols=['variables',y_col]
    # sort_cols=sort_cols+[c for c in df.columns if ((c !='variables') & (c !=y_col))]
    # df=df[sort_cols]

    # if False:
    #     df.columns=[c.replace('_','<br />').replace('-','<br />') for c in df.columns]

    #     fig = px.scatter_matrix(df, height=sm_height)

    #     fig.update_traces(marker=dict(size=1, line=dict(width=2)))
    #     fig.update_traces(diagonal_visible=False)

    #     fig.update_layout({"xaxis"+str(i+1): dict(showgrid=False,visible=False, anchor='free',position=1,titlefont=dict(size=sm_font_size), showticklabels=False,automargin=True,side='bottom') for i in range(1)})
    #     if sm_x_diagonal:
    #         fig.update_layout({"xaxis"+str(i+1): dict(zeroline=False,showgrid=False,visible=True, anchor='free',position=1-(i+sm_offset/100)/len(df.columns),titlefont=dict(size=sm_font_size), showticklabels=False,automargin=True,side='top') for i in range(1,len(df.columns))})
    #     else:
    #         fig.update_layout({"xaxis"+str(i+1): dict(zeroline=False,showgrid=False,visible=True, anchor='free',position=1-(sm_offset/100)/len(df.columns),titlefont=dict(size=sm_font_size), showticklabels=False,automargin=True,side='top') for i in range(1,len(df.columns))})

    #     fig.update_layout({"yaxis"+str(i+1): dict(showgrid=False,visible=False,titlefont=dict(size=sm_font_size),showticklabels=False,automargin=True,side='left') for i in range(1)})
    #     fig.update_layout({"yaxis"+str(i+1): dict(anchor='free',position=(sm_offset/100)/len(df.columns),zeroline=False,showgrid=False,visible=True,titlefont=dict(size=sm_font_size),showticklabels=False,automargin=True,side='left') for i in range(1,len(df.columns))})

    #     st.plotly_chart(fig,use_container_width=True)

    # if False:
    #     fig = sns.pairplot(df)
    #     st.pyplot(fig)

    if True:
        fig = uc.scatter_matrix_chart(df)
        fig.update_layout(height=sm_height)
        st.plotly_chart(fig,use_container_width=True)



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

    sorted_cols=['all','M 05','M 06','M 07','M 08','M 09','M 10','M 11','M 12','M 01','M 02','M 03','M 04']

    fig=uc.chart_heat_map(heat_map_months,sorted_cols=sorted_cols,title=y_col+ ' (r-squared)', x_col='report',y_col='v1',z_col='value', sort_by='all_abs', transpose=False, color_continuous_scale=color_scales['RdBu-sequential'], range_color=(-abs_max,abs_max), format_labels = '%{z:.1f}') #,tickangle=-90
    fig.update_layout(coloraxis_showscale=False, xaxis=dict(titlefont=dict(size=hm_font_size),tickfont=dict(size=hm_font_size),side='top'),yaxis=dict(titlefont=dict(size=hm_font_size),tickfont=dict(size=hm_font_size)))
    fig.update_layout(height=hm_height)
    st.plotly_chart(fig,use_container_width=True)


    fig=uc.chart_heat_map(heat_map_months,sorted_cols=sorted_cols,title=y_col+ ' (r-squared monthly variation)',x_col='report',y_col='v1',z_col='value',subtract='all', abs=True, sort_by='all',  transpose=False, color_continuous_scale=color_scales['RdBu-sequential'],range_color=(-20,20), format_labels = '%{z:.1f}')
    fig.update_layout(coloraxis_showscale=False, xaxis=dict(titlefont=dict(size=hm_font_size),tickfont=dict(size=hm_font_size),side='top'),yaxis=dict(titlefont=dict(size=hm_font_size),tickfont=dict(size=hm_font_size)))
    fig.update_layout(height=hm_height)
    st.plotly_chart(fig,use_container_width=True)

    # selected_points = plotly_events(fig, override_height=hm_height)    
# Scatter Plots
if True:
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
    chart_labels = chart_labels.selectbox('Chart Labels',cols_with_none, cols_with_none.index('None'))

    if chart_labels=='None':
        chart_labels=None
    elif chart_labels=='year':
        chart_labels=df.index.year
    elif chart_labels=='report':
        chart_labels=df.index

    for x in x_cols:
        with col1:        
            fig=px.scatter(df,x=x,y=y_col,color=color_var_month, text=chart_labels, trendline='ols',trendline_scope=trendline_scope, color_discrete_sequence=color_list, color_continuous_scale=color_list)
            fig.update_traces(textposition='top center')
            fig.update_layout(legend_title_text=None,coloraxis_colorbar=dict(title=None))

            st.plotly_chart(fig,use_container_width=True)
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

            st.plotly_chart(fig,use_container_width=True)
            all_models=px.get_trendline_results(fig)
            if len(all_models)>0:        
                all_models=all_models.px_fit_results
                legend_items=[]
                for trace in fig.data:
                    if 'OLS trendline' in trace.hovertemplate:
                        legend_items.append(trace.name)

                for i, tm in enumerate(all_models): # tm: Trendline Model
                    st.write(legend_items[i], 'R-Squared:', str(round(tm.rsquared,3)))