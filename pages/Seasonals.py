'''
https://docs.bokeh.org/en/test/docs/reference/models/widgets/inputs.html#bokeh.models.MultiSelect

'''

import streamlit as st
from bokeh.models import CustomJS, MultiSelect
from streamlit_bokeh_events import streamlit_bokeh_events

import Prices as up
import Charts as uc
import GDrive as gd

# Preliminaries
if True:
    service = gd.build_service()

    if 'cloud_map_dict' not in st.session_state:
        st.session_state['cloud_map_dict']=up.get_cloud_sec_map(service=service)
    if 'seas_dict' not in st.session_state:
        st.session_state['seas_dict']=[]

    cloud_map_dict=st.session_state['cloud_map_dict']
        
# Controls
if True:
    
    options=list(cloud_map_dict.keys())
    sec_selection = st.selectbox('Ticker',['']+options,  key='y_col')


if sec_selection != '':
    col1, col2 = st.columns([5,1])

    with col1:        
        sel_sec=up.select_securities(ticker_and_letter=up.info_ticker_and_letter(sec_selection) , cloud_map_dict=cloud_map_dict)
        sec_dfs= up.read_security_list(sel_sec, parallel='thread')
        st.session_state['seas_dict']=sec_dfs
        print(len(sec_dfs))
        # st.write(list(sec_df.keys()))
        # st.write(sec_df['w n_2020'])

        df=list(sec_dfs.values())[20]
        
        fig = uc.chart_security_Ohlc(df)
        fig.update_layout(height=750)
        st.plotly_chart(fig,use_container_width=True, )

    with col2:
        # Create a list of options for the MultiSelect widget
        st.write('#')
        st.write('#')
        options = [f'{i}' for i in range(2024,1959,-1)]
        pre_selection=[f'{i}' for i in range(2024,2017,-1)]

        # Create the MultiSelect widget
        bokeh_multiselect = MultiSelect(title="", value=pre_selection, options=options, size = 40, width =100)
        bokeh_multiselect.js_on_change("value",CustomJS(code='document.dispatchEvent(new CustomEvent("GET_OPTIONS", {detail: this.value}));'))
        selected_options = streamlit_bokeh_events(bokeh_multiselect,events="GET_OPTIONS",key='bokeh_multiselect_on_change', override_height=750, debounce_time=0, refresh_on_update=False)