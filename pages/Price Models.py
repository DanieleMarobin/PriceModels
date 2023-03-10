# Update the underlying data
'''
1) Open E:\grains trading\Streamlit\Monitor\HelloWorlds\Price_Models.ipynb
2) Run cell with: df_model_all=mp.Get_Data_All_Parallel(overwrite=True)
3) Check updates this one: E:\grains trading\Streamlit\Monitor\Data\Models\Price Models\Price_Models_df_model_all.csv
'''

# Imports
if True:
    from datetime import datetime as dt

    import numpy as np
    import pandas as pd
    
    import streamlit as st
    from bokeh.models import CustomJS, RadioButtonGroup
    from streamlit_bokeh_events import streamlit_bokeh_events
    import plotly.express as px
    import plotly.graph_objects as go

    import Price_Models as mp
    import GDrive as gd
    import Charts as uc
    import func as fu
    
# Functions and Callbacks
if True:
    def send_GPT_request_click(simple_chat):        
        st.session_state['ChatGPT_request']= st.session_state['ChatGPT_request_widget'] # save the request
        if simple_chat:
            st.session_state['ChatGPT_request_widget']= ''# clean the text box

    def y_col_change():
        st.session_state['run_analysis']=False

    def var_builder_change():
        # st.session_state['text_x_cols_custom']= ' '.join(st.session_state['x_cols_custom_builder']) # good

        # test
        if len(st.session_state['x_cols_custom_builder'])>0:
            st.session_state['text_x_cols_custom']= st.session_state['text_x_cols_custom']+' '+st.session_state['x_cols_custom_builder'][0]

        st.session_state['x_cols_custom_builder']=[]

    def y_col_custom_change():
        st.session_state['run_analysis']=False

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

        if ('scatter_y_matrix' in st.session_state):
            del st.session_state['scatter_y_matrix']

    def disable_analysis():
        st.session_state['run_analysis']=False
    
    def clear_x_selection():
        st.session_state['multi_x_cols'] = []
        st.session_state['text_x_cols_custom']=''
        return

    def add_custom_var_click():
        st.session_state['col_selection']= list(set(st.session_state['col_selection'] + [st.session_state['text_x_cols_custom']]))
        st.session_state['text_x_cols_custom']=''
        st.session_state['x_cols_custom_builder']=[]

    def clean_selections():
        st.session_state['chatgpt_selection']=[]

    def del_sm():
        if ('scatter_matrix' in st.session_state):
            del st.session_state['scatter_matrix']

        if ('scatter_y_matrix' in st.session_state):
            del st.session_state['scatter_y_matrix']            

    def format_report_date(item):
        return item.strftime("%b %Y")
    
# Preliminaries
if True:
    if 'run_analysis' not in st.session_state:
        st.session_state['run_analysis']=True
    if 'col_selection' not in st.session_state:
        st.session_state['col_selection']=[]
    if 'chatgpt_key' not in st.session_state:
        st.session_state['chatgpt_key']=gd.read_csv('Data/ChatGPT/ChatGPT_Info.txt').columns[0]
    if 'chatgpt_run' not in st.session_state:
        st.session_state['chatgpt_run']=False
    if 'chatgpt_selection' not in st.session_state:
        st.session_state['chatgpt_selection']=[]
    if 'ChatGPT_request' not in st.session_state:
        st.session_state['ChatGPT_request']=[]


    st.markdown("### Price Models")
    st.markdown("---")
    
    st.sidebar.markdown("### Price Models")
    color_scales = uc.get_plotly_colorscales()

# Retrieve the ALL Data
if True:
    df_model_all=fu.get_data()

# Time Frame (before everything, because it changes the amount of available variables)
with st.expander('Time Frame', expanded=False):
    # Trade Entry and Exit
    # options =list(df_model_all.index)   
    report_days=df_model_all.index
    first_training_date, last_training_date = st.select_slider('Reports', options=report_days, value=(report_days[4], report_days[-1]), format_func=format_report_date, on_change=disable_analysis)

    col1,col2=st.columns([1,8])
    with col1:
        st.write('#')
        all_options = st.checkbox("All Months", True, key='all_report_months', on_change=disable_analysis)
    with col2:
        options=list(set(df_model_all.index.month))
        options.sort()

        if all_options:
            sel_months = st.multiselect('Report Months', options, options, on_change=disable_analysis)
        else:
            sel_months = st.multiselect('Report Months', options, on_change=disable_analysis)

# From 'df_model_all' to 'model_df'
if True:
    model_df_instr=mp.model_df_instructions(df_model_all,first_training_date,last_training_date)
    model_df = mp.from_df_model_all_to_model_df(df_model_all, model_df_instr)

    # WIP: bring it into the above functions (running out of time for the presentation)
    if len(sel_months)==0: st.stop()
    mask = np.isin(model_df.index.month, sel_months)
    model_df=model_df[mask]

    options = list(model_df.columns)
    today_index=model_df.index[-1]

# Filters and Settings
if True:
    col_y_sel, col_x_sel= st.columns([1,3])

    # Standard Variables
    with col_y_sel:
        y_col = st.selectbox('Target (y)',['']+options, on_change=y_col_change, key='y_col')
    with col_x_sel:
        special_vars=['All','All-Corn','All-Wheat','All-Stock to use','All-Ending Stocks','All-Yields','All-Funds']
        options=special_vars[:]
        options=options+list(model_df.columns)
        x_cols = st.multiselect('Variables (X) - '+ str(len(options)) + ' Available', options, on_change=disable_analysis, key='multi_x_cols')
        
        draw_search=False
        draw_selected=False        

        if 'All' in x_cols:
            # x_cols=x_cols+list(model_df.columns)
            x_cols=x_cols+[c for c in model_df.columns if 'month' not in c]
            draw_search=True
        if 'All-Corn' in x_cols:
            x_cols=x_cols+[c for c in model_df.columns if 'corn' in c]
            draw_search=True
        if 'All-Wheat' in x_cols:
            x_cols=x_cols+[c for c in model_df.columns if 'wheat' in c]
            draw_search=True            
        if 'All-Stock to use' in x_cols:
            x_cols=x_cols+[c for c in model_df.columns if 'stock to use' in c]
            draw_search=True
        if 'All-Ending Stocks' in x_cols:
            x_cols=x_cols+[c for c in model_df.columns if 'ending stock' in c]
            draw_search=True
        if 'All-Yields' in x_cols:
            x_cols=x_cols+[c for c in model_df.columns if 'yield' in c]    
            draw_search=True
        if 'All-Funds' in x_cols:
            x_cols=x_cols+[c for c in model_df.columns if 'fund' in c]
            draw_search=True            

    # Custom Variables
    with st.expander('Custom Variables', expanded=False):
        custom_col_y_sel, custom_col_x_sel, col_add_custom_var= st.columns([1,3,0.2])
        with custom_col_y_sel:
            y_col_custom = st.text_input('Custom Target', on_change=y_col_custom_change, key='y_col_custom')
        with custom_col_x_sel:
            x_cols_custom = st.text_input('Custom Variables (X)',key='text_x_cols_custom')
            # if len(x_cols_custom)>0:
            #     x_cols = x_cols+[x_cols_custom]
            
            x_cols_custom_builder = st.multiselect('Variables Builder',options=['0','1','2','3','4','5','6','7','8','9','+','-','*','/','^','.','(',')']+list(model_df.columns), key='x_cols_custom_builder', on_change=var_builder_change)            
        with col_add_custom_var:
            st.write('#')
            st.write('#')
            st.button('Add', on_click=add_custom_var_click)

    # Start populating the st.session_state['col_selection'] to show in the selected table
    x_cols = list(set(x_cols)-set(special_vars))
    if ((not draw_search) & (len(x_cols)>0)):
        st.session_state['col_selection']= list(set(st.session_state['col_selection']+ list(set(x_cols))))
    
    # draw_selected=len(st.session_state['col_selection'])>0

    with st.expander('Artificial Intelligence Selection', expanded=False):
        # Voice Recognition
        if True:
            print('-----------------------------------')

            LABELS = ["Clear","Speak"]

            radio_button_group = RadioButtonGroup(labels=LABELS, active=0)
            radio_button_group.js_on_click(CustomJS(code=
            """        
                console.log('radio_button_group: active=' + this.active, this.toString())

                if (this.active==1)
                {
                    var recognition = new webkitSpeechRecognition();
                    recognition.continuous = true;
                    recognition.interimResults = true;
                
                    recognition.onresult = function (e) 
                    {
                        var value = "";
                        for (var i = e.resultIndex; i < e.results.length; ++i) 
                        {
                            if (e.results[i].isFinal) {
                                value += e.results[i][0].transcript;
                            }
                        }
                        if ( value != "") 
                        {
                            document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                        }
                    }
                    recognition.start();
                }
                else if (this.active==0)
                {
                    var value = "Clear"
                    document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                }        
            """))

            voice_recognition_results = streamlit_bokeh_events(radio_button_group,events="GET_TEXT",key='hard_issue', override_height=37,debounce_time=0, refresh_on_update=False)

            voice_request=''
            if voice_recognition_results:
                if "GET_TEXT" in voice_recognition_results:
                    voice_request=voice_recognition_results.get("GET_TEXT")

                    if (voice_request=='Clear'):
                        voice_request=''
                    else:
                        print(dt.now(),'voice_request:',voice_request)

            print(dt.now(),'voice_request:',voice_request)
            print(dt.now(),'st.session_state[chatgpt_run]:',st.session_state['chatgpt_run'])
            simple_conversation = st.checkbox('Simple Chat (if you feel lonely and want to talk to someone)',False)

        # ChatGPT
        if True:        
            gpt_cols=[]
            
            selection_request=st.text_area('From all available variables please... (select only the funds, exclude the prices, ...)', voice_request, key='ChatGPT_request_widget')
            send_request = st.button('Send Request', on_click=send_GPT_request_click, args=(simple_conversation,))

            if ((send_request) & (len(st.session_state['ChatGPT_request'])>0)):
                st.session_state['chatgpt_run']=True

        if st.session_state['chatgpt_run']:
            with st.spinner('Waiting for ChatGPT answers...'):
                options = list(model_df.columns)
                ChatGPT_requests=fu.prepare_ChatGPT_selection_requests(st.session_state['ChatGPT_request'], options,100, simple_conversation)
                ChatGPT_selection = fu.ChatGPT_parallel_answers(ChatGPT_requests, st.session_state['chatgpt_key'])

                gpt_cols_extraction=fu.extract_cols_from_ChatGPT_answers(ChatGPT_selection)
                gpt_cols = fu.maximize_columns_matching(gpt_cols_extraction, options)

                st.session_state['chatgpt_selection']=gpt_cols
                st.session_state['chatgpt_run']=False                

                st.write('ChatGPT Answer:')
                for a in ChatGPT_selection:
                    st.text(a)

    # If 'chat_gpt' picks up something, put it into the search to decide if we want to add it or not to the final selection
    if len(st.session_state['chatgpt_selection'])>0:
        x_cols=x_cols+st.session_state['chatgpt_selection']
        draw_search=True

    # Search and Select Variables
    if '' in st.session_state['col_selection']:
        st.session_state['col_selection'].remove('')

    draw_selected=len(st.session_state['col_selection'])>0    
    with st.expander('Selected Variables', expanded=True):
        if ((draw_search) | (draw_selected)):
            with st.form("my_form"):
                col_selected, col_buttons, col_search =st.columns([4,1,4])

                if draw_search:
                    with col_search:
                        df_search = pd.DataFrame({'Selection':x_cols})
                        grid_response_search = uc.aggrid_var_search(df_search, rows_per_page=50,pre_selected_rows=[])
                    with col_buttons:
                        add_but = st.form_submit_button("Add to selection")
                        sub_but = st.form_submit_button("Remove from list", on_click=clear_x_selection)
                        clean_but = st.form_submit_button("Clean Search", on_click=clean_selections)
                        df_x_cols_search=pd.DataFrame(grid_response_search['selected_rows'])                    

                        if add_but:                        
                            if len(df_x_cols_search)>0:
                                st.session_state['col_selection']=list(set(st.session_state['col_selection']+ list(df_x_cols_search['Selection'])))
                                st.session_state['chatgpt_selection']=[]
                                st.experimental_rerun()
                else:
                    with col_buttons:
                        sub_but = st.form_submit_button("Remove", on_click=clear_x_selection)

                with col_selected:
                    df_selected = pd.DataFrame({'Selection':st.session_state['col_selection']})
                    grid_response_selected = uc.aggrid_var_selected(df_selected, rows_per_page=50) 
                    df_x_cols_selected=pd.DataFrame(grid_response_selected['selected_rows'])

                    if sub_but:
                        if len(df_x_cols_selected)>0: 
                            st.session_state['col_selection']=list(set(st.session_state['col_selection'])-set(df_x_cols_selected['Selection']))

                        if ('x_cols' in st.session_state):
                            del st.session_state['x_cols']
                            
                            disable_analysis()
                            st.experimental_rerun()

        x_cols = list(set(st.session_state['col_selection']))

# Sidebar Settings
if True:
    with st.sidebar:
        st.button('Calculate', on_click=func_calculate)
        top_n_vars = st.number_input('Top N Variables',1,10000,5,1, on_change=disable_analysis)        

        with st.expander('Analysis Selection',expanded=True):
            sy_analysis = st.checkbox('Scatter Matrix (only Y-axis)',True)
            # sm_analysis = st.checkbox('Scatter Matrix',False)
            sm_analysis = False
            hm_analysis = st.checkbox('Heat Map',False)
            sp_analysis = st.checkbox('Detailed Scatter Plots',False)

        if ((sy_analysis) | (sm_analysis)):
            with st.expander('Scatter Matrix Settings'):
                tab1, tab2, tab3 = st.tabs(["Sizing", "Colors", "Trendline"])

                with tab1:
                    sy_cols = st.number_input('Number of Charts in each Row',1,50,5,1, key='syc',on_change=del_sm)

                    sm_height = st.number_input('Height',100,100000,1000,100, key='smh')
                    sm_vert = st.number_input('Vertical Spacing',0.0,1.0,0.03,0.01,on_change=del_sm)
                    sm_hor = st.number_input('Horizontal Spacing',0.0,1.0,0.01,0.01,on_change=del_sm)
                    sm_marker_size = st.number_input('Marker Size',1,100,2,1,key='sms')
                    sm_font_size = st.number_input('Font Size',1,20,12,1,key='smf', on_change=del_sm)

                    sm_y_tickangle= st.number_input('Y Title angle',-90,90,0,1,key='smy', on_change=del_sm)
                    sm_x_tickangle= st.number_input('X Title angle',-90,90,0,1,key='smx', on_change=del_sm)
                    

                with tab2:
                    sm_color = st.color_picker('Single Color', '#2929E8',on_change=del_sm)

                with tab3:
                    sm_title=False

                    sm_today_index=None
                    sm_today_size=1

                    sm_pred_index=None
                    sm_pred_size=1

                    sm_add_today = st.checkbox('Add Today',True, on_change=del_sm)
                    if sm_add_today:
                        sm_today_index=today_index
                        sm_today_size = st.number_input('Today Marker Size',1,100,10,1,key='smt', on_change=del_sm)

                    
                    sm_trendline = st.checkbox('Add Trendline',True,on_change=del_sm)                    
                    if sm_trendline:
                        sm_title = st.checkbox('Add Title',True,on_change=del_sm)
                        sm_add_pred = st.checkbox('Add Prediction', True,on_change=del_sm)

                        if sm_add_pred:
                            sm_pred_index=today_index
                            sm_pred_size = st.number_input('Prediction Size',1,100,5,1,key='smps', on_change=del_sm)

        if hm_analysis:
            with st.expander('Heat Map Settings'):
                hm_height = st.number_input('Height',100,100000,750,100, key='hmh')
                hm_font_size = st.number_input('Font Size',1,20,10,1, key='hmf')

        if sp_analysis:
            with st.expander('Scatter Plots Settings'):
                tab1, tab2, tab3 = st.tabs(["Scatter", "Colors", "Trendline"])
                with tab1:
                    chart_labels=st.container()
                    sp_height = st.number_input('Height',100,100000,750,100, key='sph')
                    sp_marker_size = st.number_input('Marker Size',1,100,5,1,key='sps')

                with tab2:
                    scatter_type = st.radio('Scatter Color',('Categorical','Continuous'))

                    single_color = st.checkbox('Single Color',False)
                    if single_color:
                        chart_color_single = st.color_picker('Single Color', '#2929E8')
                        color_list=[chart_color_single,chart_color_single]
                    else:
                        colors=list(color_scales.keys())
                        colors.sort()
                        chart_color_key = st.selectbox('Chart Color Scales',colors, colors.index('Plotly-qualitative')) # Plotly-qualitative, Jet, RdYlGn-diverging
                        color_list=color_scales[chart_color_key]

                with tab3:
                    trendline_scope = st.radio('Scope',('overall','trace', None))

                    sp_add_star = st.checkbox('Add Today Star', True)
                    if sp_add_star:
                        sp_today_size = st.number_input('Today Star Size',1,100,15,1)

                    sp_add_pred = st.checkbox('Add Prediction', True, key='sp_add_pred')
                    if sp_add_pred:
                        sp_pred_size = st.number_input('Prediction Size',1,100,10,1)

# Get selected Variables for settings or from memory (x_cols) and sort them
if True:
    # get the y_col (or y_col_custom)
    if len(y_col)>0:
        y_col=y_col
    elif len(y_col_custom)>0:
        y_col=y_col_custom
    else:
        st.stop()
                
    # Calculate all the expressions
    x_cols=x_cols+[y_col]
    model_df=fu.add_missing_cols(model_df, x_cols)

    # - if the "top_n_x_cols" (x_cols) are in memory, it means they are already sorted
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
            st.session_state['scatter_matrix'] = uc.scatter_matrix_chart(df, marker_color=sm_color, add_trendline=sm_trendline, add_title=sm_title, vertical_spacing=sm_vert, horizontal_spacing=sm_hor, today_index=sm_today_index, today_size=sm_today_size, prediction_index=sm_pred_index,prediction_size=sm_pred_size)
            fig=st.session_state['scatter_matrix']              
            
    fig.update_layout(height=sm_height)

    fig.for_each_annotation(lambda anno: anno.update(font=dict(size=sm_font_size)))
    fig.for_each_xaxis(lambda axis: axis.tickfont.update(size=sm_font_size))
    fig.for_each_yaxis(lambda axis: axis.tickfont.update(size=sm_font_size))

    st.plotly_chart(fig,use_container_width=True)

if ((sy_analysis) & (len(x_cols)>0) & (st.session_state['run_analysis'])):
    if ('scatter_y_matrix' in st.session_state):
        fig=st.session_state['scatter_y_matrix']
    else:
        with st.spinner('Calculating the Y only Scatter Matrix...'):
            df=model_df[[y_col]+x_cols]
            st.session_state['scatter_y_matrix'] = uc.sorted_scatter_chart(df, y_col, N_col_subplots=sy_cols, x_tickangle=sm_x_tickangle, y_tickangle=sm_y_tickangle, marker_color=sm_color, add_trendline=sm_trendline, add_title=sm_title, vertical_spacing=sm_vert, horizontal_spacing=sm_hor, today_index=sm_today_index, today_size=sm_today_size, prediction_index=sm_pred_index,prediction_size=sm_pred_size)
            fig=st.session_state['scatter_y_matrix']                
            
    fig.update_layout(height=sm_height)

    fig.for_each_annotation(lambda anno: anno.update(font=dict(size=sm_font_size)))
    fig.for_each_xaxis(lambda axis: axis.tickfont.update(size=sm_font_size))
    fig.for_each_yaxis(lambda axis: axis.tickfont.update(size=sm_font_size))

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


    fig=uc.chart_heat_map(heat_map_months,sorted_cols=sorted_cols,title=y_col+ ' (r-squared monthly variation)',x_col='report',y_col='v1',z_col='value',subtract='all', abs=True, sort_by='all_abs',  transpose=False, color_continuous_scale=color_scales['RdBu-sequential'],range_color=(-20,20), format_labels = '%{z:.1f}')
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
                sel_months = st.multiselect('Months', options, options)
            else:
                sel_months = st.multiselect('Months', options)
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

        if trendline_scope=='trace':
            trendline_color = None
        else:
            trendline_color = 'black'

        color_var_month=df[month_col].astype(as_type)
        color_var_year=df[crop_year_col].astype(as_type)

        # Chart Labels
        cols_with_none = ['None','calendar year','report']
        cols_with_none.extend(df.columns)
        chart_labels = chart_labels.selectbox('Chart Labels',cols_with_none, cols_with_none.index('None'))

        if chart_labels=='None':
            chart_labels=None
        elif chart_labels=='calendar year':
            chart_labels=df.index.year
        elif chart_labels=='report':
            chart_labels=df.index

    # Charts 
    for x in x_cols:
        loop_cols=[col_m_chart, col_y_chart]
        loop_color_var=[color_var_month,color_var_year]
        for c, col_chart in enumerate(loop_cols):
            with col_chart:
                fig=px.scatter(df,x=x,y=y_col,color=loop_color_var[c], text=chart_labels, trendline='ols',trendline_color_override=trendline_color, trendline_scope=trendline_scope, color_discrete_sequence=color_list, color_continuous_scale=color_list)            

                fig.update_traces(textposition='top center',marker=dict(size=sp_marker_size))
                fig.update_layout(height=sp_height, legend_title_text=None,coloraxis_colorbar=dict(title=None))

                if sp_add_star:
                    uc.add_today(fig,model_df,x,y_col,today_index, size=sp_today_size) # adding after updating layout (so I don't change the marker size)
            
                all_models=px.get_trendline_results(fig)
                if len(all_models)>0:
                    all_models=all_models.px_fit_results
                    legend_items=[]
                    colors=[]
                    for trace in fig.data:
                        if 'OLS trendline' in trace.hovertemplate: #hovertemplate
                            legend_items.append(trace.name)
                            if trace.name == 'Overall Trendline':
                                legend_items[-1]='Trend'
                                colors.append(trace.line.color)
                            else:
                                colors.append(trace.marker.color)

                    r_sq={'trace':[], 'r-squared':[], 'p-values':[], 'prediction':[]}
                    for i, tm in enumerate(all_models): # tm: Trendline Model
                        pred=np.NaN
                        if sp_add_pred:
                            pred=uc.add_today(fig,model_df,x,y_col,today_index, size=sp_pred_size, model=tm, symbol='x', color=colors[i], name='Pred: '+ legend_items[i])

                        r_sq['trace'].append(legend_items[i])
                        r_sq['r-squared'].append(tm.rsquared)
                        r_sq['p-values'].append(tm.pvalues[-1])
                        r_sq['prediction'].append(pred)

                st.plotly_chart(fig,use_container_width=True)
                if len(all_models)>0:
                    r_sq=pd.DataFrame(r_sq)
                    r_sq=r_sq.dropna(axis=1)
                    st.dataframe(pd.DataFrame(r_sq),use_container_width=False)