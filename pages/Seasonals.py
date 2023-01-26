'''
https://docs.bokeh.org/en/test/docs/reference/models/widgets/inputs.html#bokeh.models.MultiSelect
https://docs.bokeh.org/en/2.4.2/docs/user_guide/interaction/callbacks.html
'''


from datetime import datetime as dt
import streamlit as st
from bokeh.models import CustomJS, MultiSelect
from streamlit_bokeh_events import streamlit_bokeh_events

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

# Preliminaries
if True:
    service = gd.build_service()

    if 'cloud_map_dict' not in st.session_state:
        st.session_state['cloud_map_dict']=up.get_cloud_sec_map(service=service)
    if 'seas_df' not in st.session_state:
        st.session_state['seas_df']=[]
    if 're_run' not in st.session_state:
        st.session_state['re_run']=True

    cloud_map_dict=st.session_state['cloud_map_dict']
        
# Controls
if True:    
    options=list(cloud_map_dict.keys())
    sec_selection = st.selectbox('Ticker',['']+options, options.index('w n_2020.csv')+1,  key='y_col', on_change=sec_selection_on_change)

if sec_selection != '':
    seas_df = st.session_state['seas_df']

    if len(seas_df)==0:
        with st.spinner('Downloading Data...'):
            sel_sec=up.select_securities(ticker_and_letter=up.info_ticker_and_letter(sec_selection), cloud_map_dict=cloud_map_dict)
            sec_dfs= up.read_security_list(sel_sec, parallel='thread') # {'w n_2020' : df}
            st.session_state['seas_df']=up.create_seas_dict(sec_dfs)
            seas_df=st.session_state['seas_df']

    col1, col2 = st.columns([5,1])
    with col2:
        # Create a list of options for the MultiSelect widget
        # st.write('#')
        st.write('#')
        options = list(seas_df.columns)
        options.sort()
        options.reverse()
        options = [f'{o}' for o in options]
        
        # Create the MultiSelect widget   
        pre_selection=options[0: min(20,len(options))]

        bokeh_multiselect = MultiSelect(title="Years", value=pre_selection, options=options, size = 40, width =100)
        bokeh_multiselect.js_on_change("value", CustomJS(args=dict(xx='Hello Daniele'), code='console.log(xx.toString());document.dispatchEvent(new CustomEvent("GET_OPTIONS", {detail: this.value}));'))                
        sel_years = streamlit_bokeh_events(bokeh_multiselect,events="GET_OPTIONS",key='bokeh_multiselect_on_change', override_height=750, debounce_time=200, refresh_on_update=False)


    if (sel_years is None) or len(sel_years)==0 or (st.session_state['re_run']):
        cols=[int(y) for y in pre_selection]
    else:
        cols=[int(y) for y in sel_years['GET_OPTIONS']]


    with col1:        
        fig = px.line(seas_df[cols])
        fig.update_traces(line=dict(width=1))

        traces=[t['legendgroup'] for t in fig.data]
        
        if str(dt.today().year) in traces:
            id=traces.index(str(dt.today().year))
            fig.data[id].update(line=dict(width=3, color='red'))

        fig.update_layout(height=750, showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    
    if st.session_state['re_run']:
        st.session_state['re_run']=False        
        st.experimental_rerun()
