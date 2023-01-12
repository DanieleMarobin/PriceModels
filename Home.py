# Imports
if True:
    from datetime import datetime as dt

    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go

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
        st.session_state['run_analysis']=True

        if ('x_cols' in st.session_state):
            del st.session_state['x_cols']

        if ('heat_map_months' in st.session_state):
            del st.session_state['heat_map_months']

        if ('scatter_matrix' in st.session_state):
            del st.session_state['scatter_matrix']

    def disable_analysis():
        st.session_state['run_analysis']=False

    def del_sm():
         if ('scatter_matrix' in st.session_state):
            del st.session_state['scatter_matrix']

# Preliminaries
if True:
    if 'run_analysis' not in st.session_state:
        st.session_state['run_analysis']=True

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
    today_index=model_df.index[-1]
    st.write('today_index',today_index)

# Filters and Settings
if True:
    col_y_sel, col_x_sel, col_n_var, col_calc_button = st.columns([1,3,0.5,0.5])
    with col_y_sel:
        options = list(model_df.columns)
        y_col = st.selectbox('Target',options, options.index('a_price_c '), on_change=disable_analysis)

    with col_x_sel:
        special_vars=['All','All-Stock to use','All-Ending Stocks','All-Yields']
        options=special_vars[:]
        options=options+list(model_df.columns)
        x_cols = st.multiselect('Selected Variables', options, ['All-Stock to use'], on_change=disable_analysis)
        
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
        
        x_cols=x_cols+[y_col]
        x_cols = list(set(x_cols)-set(special_vars)) # Remove the 'special_vars'

    with col_n_var:
        top_n_vars = st.number_input('Top N Variables',1,10000,10,1, on_change=disable_analysis)

    with col_calc_button:
        st.markdown('##')
        st.button('Calculate', on_click=func_calculate)

    st.markdown('---')

    with st.sidebar:
        st.button('Get Latest File', on_click=func_reset)

        with st.expander('Analysis Selection',expanded=True):
            sm_analysis = st.checkbox('Scatter Matrix',True)
            hm_analysis = st.checkbox('Heat Map',True)
            sp_analysis = st.checkbox('Detailed Scatter Plots',False)

        if sm_analysis:
            with st.expander('Scatter Matrix Settings'):
                sm_trendline = st.checkbox('Add Trendline',True,on_change=del_sm)
                sm_r_squared = st.checkbox('Add R-Squared',True,on_change=del_sm)

                sm_height = st.number_input('Scatter Matrix Height',100,100000,750,100)
                sm_vert = st.number_input('Vertical Spacing',0.0,1.0,0.02,0.01,on_change=del_sm)
                sm_hor = st.number_input('Horizontal Spacing',0.0,1.0,0.01,0.01,on_change=del_sm)
                sm_marker_size = st.number_input('Marker Size',1,100,2,1)
                sm_font_size = st.number_input('Scatter Matrix Font Size',1,20,12,1)

        if hm_analysis:
            with st.expander('Heat Map Settings'):
                hm_height = st.number_input('Height',100,100000,750,100)
                hm_font_size = st.number_input('Font Size',1,20,10,1)

        if sp_analysis:
            with st.expander('Scatter Plots Settings'):
                scatter_type = st.radio('Scatter Color',('Categorical','Continuous'))
                trendline_scope = st.radio('Trendline',('overall','trace', None))
                
                chart_labels=st.container()

                colors=list(color_scales.keys())
                colors.sort()
                chart_color_key = st.selectbox('Chart Color',colors, colors.index('Plotly-qualitative')) # Plotly-qualitative, Jet, RdYlGn-diverging
                color_list=color_scales[chart_color_key]

# Get selected Variables for settings or from memory (x_cols) and sort them
if True:
# - if they are in memory, it means they are already sorted
# - if not:
#       1) need to get only the user selected columns
#       2) sort them
#       3) store them in memory    
    if ('x_cols' in st.session_state):
        x_cols=st.session_state['x_cols']    
    else:                
        with st.spinner('Calculating the top '+str(top_n_vars) + ' variables...'):
            # 1)
            df=model_df[set([y_col]+x_cols)]

            # 2)
            rank_df=mp.sorted_rsquared_var(model_df=df, y_col=y_col, n_var=1)            
            x_cols = rank_df['v1']
            x_cols=list(x_cols[0:top_n_vars])

            # 3)            
            st.session_state['x_cols']=x_cols    

# Scatter Matrix
if ((sm_analysis) & (len(x_cols)>0) & (st.session_state['run_analysis'])):
    if ('scatter_matrix' in st.session_state):
        fig=st.session_state['scatter_matrix']
    else:
        with st.spinner('Calculating the Scatter Matrix...'):
            df=model_df[[y_col]+x_cols]
            st.session_state['scatter_matrix'] = uc.scatter_matrix_chart(df,sm_trendline,sm_r_squared,sm_vert,sm_hor)
            fig=st.session_state['scatter_matrix']
            
    fig.update_layout(height=sm_height)

    fig.for_each_trace(lambda trace: trace.update(marker={'size': sm_marker_size}))

    fig.for_each_annotation(lambda anno: anno.update(font=dict(size=sm_font_size)))
    fig.for_each_xaxis(lambda axis: axis.tickfont.update(size=sm_font_size))
    fig.for_each_yaxis(lambda axis: axis.tickfont.update(size=sm_font_size))

    # fig.update_xaxes(row=rr, col=cc, tickangle=90,automargin=True,tickvals=[tick_pos],ticktext=[xc], showgrid=False,zeroline=False)
    st.plotly_chart(fig,use_container_width=True)

# Heat-Map
if ((hm_analysis) & (len(x_cols)>0) & (st.session_state['run_analysis'])):
    if ('heat_map_months' in st.session_state):
        # st.write('Getting the Monthly heatmap from memory')
        heat_map_months=st.session_state['heat_map_months']
    else:
        with st.spinner('Calculating the Monthly heatmap for top '+str(top_n_vars) + ' variables...'):
            df=model_df[[y_col]+x_cols]
            months=list(range(1,13))
            st.session_state['heat_map_months']=mp.heat_map_var_months(df, y_cols=[y_col], top_n = top_n_vars, months=months, parallel=None, show=False)[y_col]
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

# Scatter Plots
if ((sp_analysis) & (len(x_cols)>0) & (st.session_state['run_analysis'])):
    # Settings
    if True:
        mask=(model_df.index.month>0)
        
        crop_year_col='crop_year'
        month_col='month'    

        df=model_df.loc[mask][x_cols+[y_col,crop_year_col,month_col]]

        col_m_sel, col_y_sel = st.columns([1,1])
        with col_m_sel:        
            st.markdown('#### Month Split')
            st.markdown('---')        
            all_options = st.checkbox("All Months", True)

            options=list(set(df[month_col].astype('int')))
            options.sort()

            if all_options:
                sel_months = st.multiselect( 'Months', options, options)
            else:
                sel_months = st.multiselect( 'Months', options)
        with col_y_sel:        
            st.markdown('#### Crop Year Split')
            st.markdown('---')
            all_options = st.checkbox("All Years", True)

            options=list(set(df[crop_year_col].astype('int')))
            options.sort()

            if all_options:
                sel_years = st.multiselect('Years', options, options)
            else:
                sel_years = st.multiselect('Years', options)

        col_m_chart, col_y_chart = st.columns([1,1])
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

    # Charts 
    for x in x_cols:
        with col_m_chart:        
            fig=px.scatter(df,x=x,y=y_col,color=color_var_month, text=chart_labels, trendline='ols',trendline_scope=trendline_scope, color_discrete_sequence=color_list, color_continuous_scale=color_list)
            fig.add_trace(go.Scatter(name='Today',x=[model_df.loc[today_index][x]], y=[model_df.loc[today_index][y_col]], mode = 'markers', marker_symbol = 'star',marker_size = 15, marker_color='red', hovertemplate='Today'))

            fig.update_traces(textposition='top center')
            fig.update_layout(legend_title_text=None,coloraxis_colorbar=dict(title=None))

            st.plotly_chart(fig,use_container_width=True)
            all_models=px.get_trendline_results(fig)
            if len(all_models)>0:        
                all_models=all_models.px_fit_results
                legend_items=[]
                for trace in fig.data:
                    if 'OLS trendline' in trace.hovertemplate: #hovertemplate
                        legend_items.append(trace.name)

                r_sq={'month':[],'rsquared':[]}
                for i, tm in enumerate(all_models): # tm: Trendline Model
                    r_sq['month'].append(legend_items[i])
                    r_sq['rsquared'].append(round(tm.rsquared,3))
                
                st.dataframe(pd.DataFrame(r_sq))
                

        with col_y_chart:            
            fig=px.scatter(df,x=x,y=y_col,color=color_var_year, text=chart_labels, trendline='ols',trendline_scope=trendline_scope, color_discrete_sequence=color_list, color_continuous_scale=color_list)
            fig.add_trace(go.Scatter(name='Today',x=[model_df.loc[today_index][x]], y=[model_df.loc[today_index][y_col]], mode = 'markers', marker_symbol = 'star',marker_size = 15, marker_color='red', hovertemplate='Today'))
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

                r_sq={'year':[],'rsquared':[]}
                for i, tm in enumerate(all_models): # tm: Trendline Model
                    r_sq['year'].append(legend_items[i])
                    r_sq['rsquared'].append(round(tm.rsquared,3))
                
                st.dataframe(pd.DataFrame(r_sq))