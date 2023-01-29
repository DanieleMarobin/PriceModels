'''
https://docs.bokeh.org/en/test/docs/reference/models/widgets/inputs.html#bokeh.models.MultiSelect
https://docs.bokeh.org/en/2.4.2/docs/user_guide/interaction/callbacks.html
'''

# Preliminaries
if True:
    from datetime import datetime as dt
    import streamlit as st
    from bokeh.models import CustomJS, MultiSelect
    from streamlit_bokeh_events import streamlit_bokeh_events
    import pandas as pd

    import Prices as up
    import Charts as uc
    import GDrive as gd
    import plotly.express as px

    st.set_page_config(page_title='Seasonals',layout='wide',initial_sidebar_state='expanded')

# Events
def sec_selection_on_change():
    st.session_state['bokeh_multiselect_on_change']=[]
    st.session_state['seas_df']=[]
    st.session_state['re_run']=True

def format_timeframe_date(item):
    return item.strftime("%b %Y")

# Preliminaries
if True:
    service = gd.build_service()

    if 'cloud_map_dict' not in st.session_state:
        st.session_state['cloud_map_dict']=up.get_cloud_sec_map(service=service)
    if 'seas_df' not in st.session_state:
        st.session_state['seas_df']=[]
    if 're_run' not in st.session_state:
        st.session_state['re_run']=True

    var_options=['close_price',
    'open_price',
    'low_price',
    'high_price',
    'open_interest',
    'volume',

    'implied_vol_dm',
    'implied_vol_dm_call_25d',
    'implied_vol_dm_put_25d',    

    'implied_vol_hist_call',
    'implied_vol_call_50d',
    'implied_vol_hist_put',
    'implied_vol_put_50d',
    'bvol_50d',

    'implied_vol_call_25d',
    'bvol_call_25d',

    'implied_vol_put_25d',            
    'bvol_put_25d',
    ]

    cloud_map_dict=st.session_state['cloud_map_dict']
        
# Controls
if True:
    options=up.select_securities(include_continuous=False, cloud_map_dict=cloud_map_dict)
    options=list(set([up.info_ticker_and_letter(s) for s in options]))
    options.sort()
    sec_selection = st.selectbox('Ticker',['']+options, options.index('w n')+1,  key='y_col', on_change=sec_selection_on_change)
    expression = st.text_input('Expression', key='y_expression', on_change=sec_selection_on_change)
    print('expression',expression)
    st.write(expression)

    seas_interval=[dt.date(dt.today()-pd.DateOffset(months=6)+pd.DateOffset(days=1)), dt.date(dt.today()+pd.DateOffset(months=6))]
    options=pd.date_range(seas_interval[0]-pd.DateOffset(months=18), seas_interval[1]+pd.DateOffset(months=18))
    date_start, date_end = st.select_slider('Seasonals Window', options=options, value=(seas_interval[0], seas_interval[1]), format_func=format_timeframe_date, on_change=sec_selection_on_change)
    # date_start, date_end = st.select_slider('Seasonals Window', options=options, value=(options[0], options[-1]), on_change=sec_selection_on_change)

    with st.sidebar:
        var_selection = st.selectbox('Variable',var_options, var_options.index('close_price'),  key='var_selection', on_change=sec_selection_on_change)

# Calculations
if (expression != ''):
    seas_df = st.session_state['seas_df']

    # Core Calc
    if len(seas_df)==0:
        with st.spinner('Downloading Data...'):
            symbols=up.extract_symbols_from_expression(expression)
            print('symbols',symbols)
            sel_sec=[]
            for s in symbols:
                sel_sec=sel_sec+up.select_securities(ticker_and_letter=up.info_ticker_and_letter(up.symbol_no_offset(s)), cloud_map_dict=cloud_map_dict)

            # sec_dfs = {'w n_2020' : df}
            sec_dfs= up.read_security_list(sel_sec, parallel='thread')

        with st.spinner('Making the Seasonals Calculation...'):
            if '_vol_' in var_selection:
                for key, df in sec_dfs.items():
                    df=up.calc_volatility(df, vol_to_calc=var_selection, min_vol=0, max_vol=150, max_daily_ratio_move=2.0, holes_ratio_limit=1.2)

            # pass the expression together with the 'sec_dfs'
            st.session_state['seas_df']=up.create_seas_df(expression, sec_dfs, var_selection, seas_interval= [date_start, date_end])
            seas_df=st.session_state['seas_df']

    # Years Selection
    col1, col2, col3 = st.columns([12,0.2,1])
    with col3:
        # Create a list of options for the MultiSelect widget
        # st.write('#')
        # st.write('#')
        options = list(seas_df.columns)
        options.sort()
        options.reverse()
        options = [f'{o}' for o in options]
        
        # Create the MultiSelect widget   
        pre_selection=options[0: min(20,len(options))]
        bokeh_multiselect = MultiSelect(value=pre_selection, options=options, size = 45, width =80)
        bokeh_multiselect.js_on_change("value", CustomJS(args=dict(xx='Hello Daniele'), code='console.log(xx.toString());document.dispatchEvent(new CustomEvent("GET_OPTIONS", {detail: this.value}));'))                
        sel_years = streamlit_bokeh_events(bokeh_multiselect,events="GET_OPTIONS",key='bokeh_multiselect_on_change', override_height=750, debounce_time=200, refresh_on_update=False)
    if (sel_years is None) or len(sel_years)==0 or (st.session_state['re_run']):
        cols=[int(y) for y in pre_selection]
    else:
        cols=[int(y) for y in sel_years['GET_OPTIONS']]

    # Chart
    with col1:
        df=seas_df[cols]
        df['mean']=df.mean(skipna=True, axis=1)
        cols=['mean']+cols

        fig = px.line(df[cols])
        fig.update_traces(line=dict(width=1))

        traces=[t['legendgroup'] for t in fig.data]
        
        if str(dt.today().year) in traces:
            id=traces.index(str(dt.today().year))
            fig.data[id].update(line=dict(width=3, color='red'))

        if str('mean') in traces:
            id=traces.index('mean')
            fig.data[id].update(line=dict(width=3, color='black'))

        fig.update_layout(height=750, showlegend=False, xaxis=dict(title=None), yaxis=dict(title=None))
        fig.update_layout(margin=dict(l=50, r=0, t=0, b=20))

        st.plotly_chart(fig,use_container_width=True, config={'scrollZoom': True, 'displayModeBar':False})

    # Re-Run hack
    if st.session_state['re_run']:
        st.session_state['re_run']=False        
        st.experimental_rerun()
