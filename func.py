import streamlit as st
import GDrive as gd
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import concurrent.futures
import sympy as sym
import re

def get_data():
    PM_DIR = 'Data/Models/Price Models/'
    file_name = PM_DIR+'Price_Models_df_model_all.csv'

    # Delete all the items in Session state (to save memory)
    # for key in st.session_state.keys():
    #     if ((key!='df_model_all') & (key!='heat_map_months')):
    #         del st.session_state[key]

    df_model_all=None
    if ('df_model_all' in st.session_state):
        with st.spinner('Getting data from Memory...'):
            df_model_all=st.session_state['df_model_all']
    else:
        with st.spinner('Getting data from Google Drive...'):
            st.session_state['df_model_all']=gd.read_csv(file_name,parse_dates=['date'], dayfirst=True, index_col='date')
            df_model_all=st.session_state['df_model_all']

    return df_model_all

def dm_split(string, separators = "-+*/()'^."):
    result = re.split('|'.join(map(re.escape, separators)), string)
    return result

def dm_replace(string, args_dict={}):
    for k, v in args_dict.items():
        string = string.replace(k, v)
    return string

def reverse_dict(original_dict):
    return {v: k for k, v in original_dict.items()}

def extract_symbols_from_expression(expression):
    # the symbolic package doesn't like spaces in symbols
    # so this function returns a dictionary {original:modified}
    separators = "-+*/()'^.,"
    fo=dm_split(expression,separators)
    fo=[s.strip() for s in fo]
    fo=[s for s in fo if not s.isnumeric()]

    fo={s: s.replace(' ','_') for s in fo}

    return fo

def evaluate_expression(df,expression):
    symbols_dict=extract_symbols_from_expression(expression)
    expression=dm_replace(expression, symbols_dict)

    symbols = sym.symbols(list(symbols_dict.values()))
    
    expression = sym.sympify(expression)
    f = sym.lambdify([symbols], expression, 'numpy')

    cols=list(symbols_dict.keys())
    var_list=[df[c] for c in cols] # preserving the index
    # var_list=df[cols].values.T # nicer code
    return f(var_list)

def add_missing_cols(df, cols):
    # the assumption is that the missing columns names are the expressions to be evaluated
    for c in cols:
        if c not in df.columns:
            df[c]=evaluate_expression(df,c)

    return df

def prepare_ChatGPT_selection_requests(user_request, col_options=[], subset_size=100, carry_on_conversation=False):
    """
    as ChatGPT has around 4000 input tokens limitation (https://beta.openai.com/docs/models/gpt-3)
    I need to split the columns list in batches

    https://blog.devgenius.io/chatgpt-how-to-use-it-with-python-5d729ac34c0d
    """

    requests=[]
    if (carry_on_conversation):
        requests=requests+[user_request]
    else:
        subsets_col_options=[]
        for i in range(0, len(col_options), subset_size):
            subsets_col_options=subsets_col_options+[col_options[i:i+subset_size]]
        
        requests=[]

        for s in subsets_col_options:
            fo='From the items in this python list '
            # fo='From the variables in this python list '
            fo=fo+'['+ ','.join(s) +']'
            fo=fo+user_request
            fo=fo+' and put the result in a python list format'

            requests=requests+[fo]

    return requests

def ChatGPT_parallel_answers(requests=[],key=None):

    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        results={}
        for i,r in enumerate(requests):
            results[i] = executor.submit(ChatGPT_answer, r, key)
    
    fo=[]
    for var, res in results.items():
        fo=fo+[res.result()]

    return fo

def ChatGPT_answer(request,key):
    # Define OpenAI API key 
    openai.api_key = key

    # Set up the model
    model_engine = "text-davinci-003"

    completion = openai.Completion.create(engine=model_engine, prompt=request,max_tokens=1024,n=1,stop=None,temperature=0.5)
    
    response = completion.choices[0].text
    return(response)

def extract_cols_from_ChatGPT_answers(answers=[]):
    fo=[]
    for a in answers:
        a=a.replace('[','').replace(']','').replace('"','').replace("'",'').split(',')
        fo=fo+[c.strip() for c in list(a)]
    return fo

def maximize_columns_matching(tentative_cols, correct_cols):
    augmented=[]
    for c in tentative_cols:
        augmented=augmented+[c, ' '+c, c+' '] # add leading and trailing spaces

    fo = list(set(augmented).intersection(correct_cols))
    return fo
