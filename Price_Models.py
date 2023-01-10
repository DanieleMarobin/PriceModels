# Comment
"""
This file relies on the library 'pygad' for the Genetic Algorithms calculations
Unfortunately there are certain functions that do not accept external inputs
so the only way to pass variables to them is to have some global variables
"""

# Imports
if True:
    import sys;
    sys.path.append(r'\\ac-geneva-24\E\grains trading\Streamlit\Monitor\\')

    import warnings # supress warnings
    warnings.filterwarnings('ignore')

    from datetime import datetime as dt
    import numpy as np
    import pandas as pd
    from itertools import combinations, combinations_with_replacement

    import statsmodels.api as sm

    import concurrent.futures
    import Charts as uc
    

# functions
def Fit_Model(df, y_col: str, x_cols=[], exclude_from=None, extract_only=None):
    """
    'exclude_from' needs to be consistent with the df index
    """

    if not ('const' in df.columns):
        df = sm.add_constant(df, has_constant='add')

    if not ('const' in x_cols):        
        x_cols.append('const')

    if exclude_from!=None:
        df=df.loc[df.index<exclude_from]

    y_df = df[[y_col]]

    if (len(x_cols)>0):
        X_df=df[x_cols]
    else:
        X_df=df.drop(columns = y_col)

    model = sm.OLS(y_df, X_df).fit()

    if extract_only is None:
        fo = model
    elif extract_only == 'rsquared':
        fo = model.rsquared

    return fo

def run_multiple_models(df, y_col: str, x_cols_list=[], extract_only='rsquared', parallel=None, max_workers=None):
    """
    'x_cols_list' (list of list):
        -   1 list for each model
        -   1 list of 'x_cols' (all the explanatory variables)


    the below [:] is needed because in python the lists are always passed by reference
    for a great explanation ask chatgpt the below:
            - how can I pass a list by value in python?        
    """
    fo={}

    if parallel is None:
        for x_cols in x_cols_list:            
            key = '/'.join(x_cols)
            fo[key] = Fit_Model(df, y_col, x_cols[:], None, extract_only)

    elif parallel=='thread':
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results={}
            for x_cols in x_cols_list:
                key = '/'.join(x_cols)             
                results[key] = executor.submit(Fit_Model, df, y_col, x_cols[:], None, extract_only)
        
        for key, res in results.items():
            fo[key]=res.result()

    elif parallel=='process':
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results={}
            for x_cols in x_cols_list:
                key = '/'.join(x_cols)             
                results[key] = executor.submit(Fit_Model, df, y_col, x_cols[:], None, extract_only)
        
        for key, res in results.items():
            fo[key]=res.result()

    return fo

def sorted_rsquared_var(model_df, y_col='y', n_var=1, with_replacement=False, cols_excluded=[], parallel=None, max_workers=None):
    """
    with_replacement = True
        - makes a difference only if 'n_var>1'
        - it means that if we have 'n_var==2', it will also try the model [v1, v1] so the same variable n times
        - 'n_var==3' => [v1, v1, v1]        
    """
    # Creating the results dictiony: 
    #       1) Adding Variables cols
    #       2) then the 'value'
    results_dict={}
    for v in range(n_var):
        results_dict['v'+str(v+1)]=[]
    results_dict['value']=[]

    x_cols_list=[]
    cols_excluded = cols_excluded+[y_col]
    cols_model = list(set(model_df.columns)-set(cols_excluded))

    if with_replacement:        
        comb = combinations(cols_model, n_var)
    else:
        comb = combinations_with_replacement(cols_model, n_var)

    x_cols_list = [list(c) for c in comb] # converting 'list of tuples' to 'list of lists' 
    models_results=run_multiple_models(df=model_df, y_col=y_col, x_cols_list=x_cols_list, extract_only=None, parallel=parallel, max_workers=max_workers)

    # Visualize with the heat map
    for key, model in models_results.items():
        # Add the variable names
        vars_split=key.split('/')        
        [results_dict['v'+str(i+1)].append(v) for i,v in enumerate(vars_split)]

        # Add the R-Squared
        if n_var>1:
            results_dict['value'].append(100.0*model.rsquared)
        else:
            # if there is only 1 variable, I also put the sign of the relationship
            results_dict['value'].append(np.sign(model.params[key])*100.0*model.rsquared)
            

    # Create and Sort the 'Ranking DataFrame'
    rank_df=pd.DataFrame(results_dict)

    rank_df['abs_value']=rank_df['value'].abs()
    rank_df=rank_df.sort_values(by='abs_value',ascending=False)

    # to extract the top N
    # sorted_vars = rank_df['variable']
    # x_cols_list=sorted_vars[0:top_n]

    return rank_df

# Data
if True:
    def model_df_instructions(df_model_all):
        """
        These are the instructions to reduce the 'df_model_all' to the final 'model_df' that will be used for the search
        """
        fo={}
        # Timing
        first_training_date=dt(1995,5,1) # to start with a fresh new crop year
        last_training_date=dt(2022,12,1)

        # Shifts
        cols_to_shift=[]
        # cols_to_shift=cols_to_shift+[c for c in df_model_all.columns if 'price_' in c]
        cols_to_shift=cols_to_shift+[c for c in df_model_all.columns if 'fund' in c]

        # Columns to use
        cols_to_use=[]
        # cols_to_use=cols_to_use+[c for c in df_model_all.columns] # everything
        cols_to_use=cols_to_use+[c for c in df_model_all.columns if (('price_' in c) & ('security' not in c))] # funds
        cols_to_use=cols_to_use+[c for c in df_model_all.columns if 'fund' in c] # funds
        cols_to_use=cols_to_use+[c for c in df_model_all.columns if 'wasde' in c] # wasde

        # shifts columns
        # cols_to_use=cols_to_use+[c for c in cols_to_shift if (('price_' in c) & ('security' not in c))] # all the prices (as one of them will be the 'y_col' to model)
        # cols_to_use=cols_to_use+[c+'_shift1' for c in cols_to_shift if 'security' not in c] # shif1 columns

        fo['first_training_date']=first_training_date
        fo['last_training_date']=last_training_date
        fo['cols_to_shift']=list(set(cols_to_shift))
        fo['cols_to_use']=list(set(cols_to_use))

        return fo

    def from_df_model_all_to_model_df(df_model_all, instructions):
        """
        it takes ALL the variables and selects what it is needed for the analysis
        """

        first_training_date=instructions['first_training_date']
        last_training_date=instructions['last_training_date']
        cols_to_shift=instructions['cols_to_shift']
        cols_to_use=instructions['cols_to_use']

        # Adding Shifts
        dfs=[df_model_all]
        for s in range(1,2):
            df_temp=df_model_all[cols_to_shift].shift(s)
            df_temp.columns=[c+'_shift'+str(s) for c in cols_to_shift]
            dfs.append(df_temp)

        df_model=pd.concat(dfs,axis=1)

        # cutting according to the start date
        mask=((df_model.index>=first_training_date) & (df_model.index<=last_training_date))
        df_model=df_model[mask]

        # keep only the selected columns
        df_model=df_model[cols_to_use]

        # Dropping columns that don't have values from the beginning, for example 'ethanol' (as it was only reported from May 2004)
        df_model=df_model.dropna(axis=1,how='any')

        print('df_model.shape', df_model.shape)

        if df_model.isna().sum().sum() > 0:
                print('There are NaN in the data')
                return None

        return df_model

# Results visualization
if True:
    def heat_map_var_months(model_df, y_cols=['a_price_c '], top_n = 40, months=list(range(1,13)), cols_excluded=[], parallel=None, max_workers=None, show=False):
        """
        1 'heat_map' for each item in 'y_cols'
        """
        fo={}        
        for y in y_cols:
            fo[y]=[]
            # Calculating the 'Top N Variables'
            rank_df=sorted_rsquared_var(model_df=model_df, y_col=y, n_var=1, cols_excluded=cols_excluded, parallel=parallel, max_workers=max_workers)
            rank_df['report']='all'
            sorted_vars = rank_df['v1']
            cols_model=list(sorted_vars[0:top_n])
            cols_model.append(y)

            fo[y].append(rank_df)
            print('Done Sorting All Months')

            # print(model_df[mask][cols_model])

            for m in months:        
                mask=(model_df.index.month==m)

                rank_df=sorted_rsquared_var(model_df=model_df[mask][cols_model], y_col=y, n_var=1, cols_excluded=cols_excluded, parallel=None, max_workers=max_workers)
                rank_df['report']= 'M ' + '{:0>2}'.format(m)
                fo[y].append(rank_df)       
                
                print('Month:', m)

            heat_map_df=pd.concat(fo[y])

            mask=np.isin(heat_map_df['v1'], cols_model)
            heat_map_df=heat_map_df[mask]

            if show:
                color_scales = uc.get_plotly_colorscales()
                # Heat-Map
                c=color_scales['RdBu-sequential']
                abs_max=heat_map_df['value'].abs().max() # so the positives are Blue and the negatives are red
                fig=uc.chart_heat_map(heat_map_df,x_col='report',y_col='v1',z_col='value', sort_by='all', transpose=True, color_continuous_scale=c, format_labels = '%{z:.1f}', title=y, range_color=(-abs_max,abs_max), tickangle=-90)
                fig.show('browser')

                # Heat-Map Differences
                fig=uc.chart_heat_map(heat_map_df,x_col='report',y_col='v1',z_col='value',subtract='all', abs=True, sort_by='all',  transpose=True, color_continuous_scale=c,range_color=(-20,20), format_labels = '%{z:.1f}', title=y, tickangle=-90)
                fig.show('browser')

            fo[y]=heat_map_df

        return fo