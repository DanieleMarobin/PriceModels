import re
from datetime import datetime as dt
import pandas as pd
import numpy as np
import sympy as sym

import concurrent.futures
import GDrive as gd

SEC_DIR = 'Data/Securities/'
SEC_MAP_PATH = 'Data/Search_Indices/GDrive_Securities_index.csv'
SEC_MAP_NAME = 'GDrive_Securities_index.csv'
SEC_MAP_ID = '1eZ4FkRcW2JwS6i1LST5U4jqTRtQk701p' # id of the file mapping {name:id} for the folder 'Data/Securities'

# Find securities
if True:
    def get_cloud_sec_map(cloud_map_id=SEC_MAP_ID, service=None):
        cloud_map = gd.get_GDrive_map_from_id(cloud_map_id, service=service)
        return cloud_map

    def select_securities(ticker=None, ticker_and_letter=None, cloud_map_id=None, cloud_map_dict=None, service=None):
        folder=SEC_DIR
        if cloud_map_dict==None:
            cloud_map_id=SEC_MAP_ID

        all_files=gd.listdir(folder, cloud_map_id=cloud_map_id, cloud_map_dict=cloud_map_dict, service=service)

        if ticker is not None:            
            fo = [sec for sec in all_files if info_ticker(sec)==ticker]
        elif ticker_and_letter is not None:
            fo = [sec for sec in all_files if info_ticker_and_letter(sec)==ticker_and_letter]

        fo = [sec.replace('.csv','') for sec in fo]
        return fo

    def select_funds():
        all_files = gd.listdir(SEC_DIR)
        fo =[f.split('_')[0] for f in all_files if 'funds' in f]
        return fo

# Read securities, Calendars, VaR files, etc...
if True:
    def read_security_list(sec_list=[], parallel=None, max_workers=500, cloud_map_dict=None, service=None, cloud=False):
        fo={}
        if cloud:
            parallel='thread'

        if parallel is None:
            for sec in sec_list:
                fo[sec] = read_security(sec, cloud, service)

        elif parallel=='thread':
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results={}
                for sec in sec_list:
                    results[sec] = executor.submit(read_security, sec, cloud, service)
            
            for key, res in results.items():
                fo[key]=res.result()

        elif parallel=='process':
            with concurrent.futures.ProcessPoolExecutor(max_workers= min(max_workers,61)) as executor:
                results={}
                for sec in sec_list:
                    results[sec] = executor.submit(read_security, sec, cloud, service)
            
            for key, res in results.items():
                fo[key]=res.result()

        if gd.is_cloud_id(sec_list[0]) and (cloud_map_dict is not None):
            cloud_map_dict = {v: k for k, v in cloud_map_dict.items()} # reverse the dictionary to {id:file_name}
            fo= {cloud_map_dict[id].replace('.csv','') :df for id, df in fo.items()}
                
        return fo
    
    def read_security(sec, cloud=False, service=None):
        if gd.is_cloud_id(sec):
            file=sec
        else:
            if '_' not in sec:
                sec=sec+'_0'
            if '.csv' not in sec:
                sec=sec+'.csv'

        if cloud:            
            file=sec # in this way it goes faster (as it doesn't have to request for the folders in the path)
        else:
            file = SEC_DIR + sec

        df = gd.read_csv(file_path=file, service=service, parse_dates=['date'], dayfirst=True, index_col='date')

        return df

    def read_calendar():
        file = SEC_DIR+ 'all_calendar.csv'
        df=gd.read_csv(file,parse_dates=['start','end'], dayfirst=True, index_col='security')
        return df


# get Info
if True:
    def info_ticker_and_letter(sec):
        split=sec.split('_')
        
        if len(split)==1:
            # there is no '_' in the security
            return sec
        
        if split[-1]=='0':
            return split[-2]
        
        else:
            return split[-2]
        
    def info_ticker(sec):
        split=sec.split('_')
        
        if len(split)==1:
            # there is no '_' in the security
            return sec
        
        if split[-1]=='0':
            return split[-2]
        
        else:
            return split[-2][0:-1]

    def info_maturity(sec):
        year= int(sec.split('_')[1])
        letter= sec.split('_')[0][-1]
        month=month_from_letter(letter)
        return dt(year,month,1)

# Accessories
if True:
    def month_from_letter(letter):
        if letter=='f':
            return 1
        elif letter=='g':
            return 2
        elif letter=='h':
            return 3
        elif letter=='j':
            return 4
        elif letter=='k':
            return 5
        elif letter=='m':
            return 6
        elif letter=='n':
            return 7
        elif letter=='q':
            return 8
        elif letter=='u':
            return 9
        elif letter=='v':
            return 10
        elif letter=='x':
            return 11
        elif letter=='z':
            return 12

    def letter_from_month(month):
        if month==1:
            return 'f'
        elif month==2:
            return 'g'
        elif month==3:
            return 'h'
        elif month==4:
            return 'j'
        elif month==5:
            return 'k'
        elif month==6:
            return 'm'
        elif month==7:
            return 'n'
        elif month==8:
            return 'q'
        elif month==9:
            return 'u'
        elif month==10:
            return 'v'
        elif month==11:
            return 'x'
        elif month==12:
            return 'z'  

    def relative_security(security, base_year):
        # 'security' must be in the form 'c z_2022'
        split=security.split('_')

        ticker=split[0][0:-1]
        letter=split[0][-1]    
        relative_year=int(split[1])-base_year

        return ticker+letter+str(relative_year)  


# Symbolic Expressions
if True:
    def dm_split(string, separators = "-+*/()'^."):
        result = re.split('|'.join(map(re.escape, separators)), string)
        return result

    def dm_replace(string, args_dict={}):
        for k, v in args_dict.items():
            string = string.replace(k, v)
        return string

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
        # var_list=[df[c] for c in cols] # preserving the index
        var_list=df[cols].values.T # nicer code
        return f(var_list)

# Wasde
if True:
    def wasde_price_single(ticker, wasde_reports_series):
        df=read_security(ticker)
        df=df.resample('1d').ffill() # this to make sure that if there are holes, they will be filled

        # equivalent of the above line
        # new_index=pd.date_range(df.index.min(),df.index.max())
        # df=df.reindex(index=new_index,method='ffill')

        df = pd.merge(left=df, right=wasde_reports_series, left_index=True,right_index=True,how='left')
        df['report']=df['report'].fillna(method='ffill')
        df = df[['close_price','report']].groupby('report').mean()
        df=df.rename(columns={'close_price':ticker})

        return df    
    def parallel_wasde_price_single(single_variables, wasde_reports_series, parallel=None,max_workers=None):
        fo={}
        if parallel is None:
            for ticker in single_variables:
                fo[ticker] = wasde_price_single(ticker, wasde_reports_series)

        elif parallel=='thread':
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results={}
                for ticker in single_variables:
                    results[ticker] = executor.submit(wasde_price_single, ticker, wasde_reports_series)
            
            for key, res in results.items():
                fo[key]=res.result()

        elif parallel=='process':
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results={}
                for ticker in single_variables:
                    results[ticker] = executor.submit(wasde_price_single, ticker, wasde_reports_series)
            
            for key, res in results.items():
                fo[key]=res.result()

        # fo=pd.concat(fo,axis=1)
        # fo.columns = fo.columns.droplevel(level=1)

        fo = list(fo.values())
        fo=pd.concat(fo,axis=1)
        fo=fo.fillna(method='ffill')
        print('df_price_single.shape',fo.shape)
        return fo

    def wasde_price_multi(setting, wasde_reports_series, futures_calendar,  sel_years = range(1995,2024)):
        """
        example of 'setting':
            - setting={'security':'k1', 'start_month':5, 'prefix':'k_price_'}
            - setting={'security':None, 'start_month':None, 'prefix':'a_price_'}
        """
        fo={}
        ticker=setting['ticker']
        new_col_name=ticker+setting['suffix']

        if setting['delivery'] is not None:
            user_delivery=ticker+setting['delivery']
        else:
            user_delivery=None

        report_security_selection=wasde_report_security(ticker=ticker, futures_calendar=futures_calendar, wasde_report_dates=wasde_reports_series, user_delivery=user_delivery, user_start_month=setting['start_month'])
        
        sel_months=set([r[-2:-1] for r in report_security_selection])

        # Reading the price data according to 'sel_years' and 'sel_months'
        dfs={}
        for y in sel_years:
            for m in sel_months:
                sec = ticker+m+'_'+str(y)
                df=read_security(sec)
                df=df.resample('1d').ffill() # this to make sure that if there are holes, they will be filled
                df = pd.merge(left=df, right=wasde_reports_series, left_index=True,right_index=True,how='left')
                df['report']=df['report'].fillna(method='ffill')
                df = df[['close_price','report']].groupby('report').mean()
                df=df.rename(columns={'close_price':sec})            
                dfs[sec]=df

        df_price=pd.concat(dfs,axis=1)
        df_price.columns = df_price.columns.droplevel(level=0) # 'level' which level to drop (0: drops the first, 1: drop the second, etc etc)
            
        # Select according to 'report_security_selection'
        df_price=pd.melt(df_price, ignore_index=False)

        df_price=df_price.dropna(how='any') # this drops rows like: 'average price of 'w k_1998' for the report of 'may 2014'
        df_price['report_security']=[str(index.month)+"-"+relative_security( row['variable'],index.year) for index,row in df_price.iterrows()]

        mask = np.isin(df_price['report_security'],report_security_selection)
        df_price = df_price[mask]
        
        df_price = df_price.rename(columns={'variable':'security_'+new_col_name,'value':new_col_name})
        # df_price = df_price.drop(columns=['report_security','security_'+new_col_name])
        df_price = df_price.drop(columns=['report_security'])
        
        print(ticker)
        fo[new_col_name]=df_price
        return fo
    def parallel_wasde_price_multi(multi_variables, wasde_reports_series, futures_calendar,  sel_years = range(1995,2024), parallel=None, max_workers=None):
        """
        example of 'multi_variables':
            - multi_variables={'ticker':'c ', 'delivery':'k1', 'start_month':5, 'prefix':'k_price_'}
            - multi_variables={'ticker':'ng','delivery':None, 'start_month':None, 'prefix':'a_price_'}
        """
        fo={}
        if parallel is None:
            for setting in multi_variables:
                key=setting['ticker']+setting['suffix']
                fo[key] = wasde_price_multi(setting, wasde_reports_series, futures_calendar,  sel_years)

        elif parallel=='thread':
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results={}
                for setting in multi_variables:
                    key=setting['ticker']+setting['suffix']
                    results[key] = executor.submit(wasde_price_multi, setting, wasde_reports_series, futures_calendar,  sel_years)
            
            for key, res in results.items():
                fo[key]=res.result()

        elif parallel=='process':
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results={}
                for setting in multi_variables:
                    key=setting['ticker']+setting['suffix']
                    results[key] = executor.submit(wasde_price_multi, setting, wasde_reports_series, futures_calendar,  sel_years)
            
            for key, res in results.items():
                fo[key]=res.result()

        fo = list(fo.values())
        fo =[list(df.values())[0] for df in fo]
        fo=pd.concat(fo, axis=1)
        fo=fo.fillna(method='ffill')
        print('df_price_multi.shape',fo.shape)
        return fo

    def wasde_report_security(ticker, futures_calendar=None,wasde_report_dates=None, user_delivery=None, user_start_month=None, return_full_df=False):
        """
        hello world
            - up.wasde_report_security('c ', futures_calendar, wasde_reports_series)

        it gives which future to use for a certain WASDE Report month (based on a futures calendar schedule)
        the output looks like this:

        report_month
        1      1-c h0
        2      2-c h0
        3      3-c k0
        4      4-c k0
        5      5-c n0
        6      6-c n0
        7      7-c z0
        8      8-c z0
        9      9-c z0
        10    10-c z0
        11    11-c z0
        12    12-c h1

        if I don't want the automatic calculation, it is necessary to provide:
            - user_delivery (like 'c k1'), to force the use of a specific security
            - user_start_month (like ), to force the use of a specific security

        return_full_df = True:
            - returns the table before selecting the 'most frequenct'
        """

        # Automatic Security Selection
        if user_delivery is None:
            fo={'report':[],'security':[]}

            if ('ticker' not in futures_calendar.columns):
                futures_calendar['ticker']=[info_ticker(sec) for sec in futures_calendar.index]

            for report_day in wasde_report_dates: 
                mask = ((futures_calendar['start']<=report_day) & (report_day<=futures_calendar['end']) & (futures_calendar['ticker']==ticker))
                fo['report'].append(report_day)
                fo['security'].append(futures_calendar[mask].index[0])    

            df=pd.DataFrame(fo)
            df['delivery']=[info_maturity(sec) for sec in df['security']]
            df['sec_month']=df['delivery'].dt.month
            df['report_month']=df['report'].dt.month
            df['report_year']=df['report'].dt.year
            
            df['relative_sec']=[relative_security(row['security'],row['report_year']) for index,row in df.iterrows()]    

            df=df.pivot(index='report_year',columns='report_month',values='relative_sec')

            if return_full_df:
                return df.sort_index(ascending=False)

            df=df.mode().T
            df['selection']=df.index.astype(str) +'-'+df[0]
            report_security_selection=df['selection']

        # User Specified
        else:        
            prev_offset=int(user_delivery[-1])-1
            prev_user_delivery = user_delivery[0:-1]+str(prev_offset)
            
            series_dict={}
            for m in range(1,user_start_month):
                series_dict[m]=str(m)+'-'+prev_user_delivery

            for m in range(user_start_month,13):
                series_dict[m]=str(m)+'-'+user_delivery

            report_security_selection=pd.Series(series_dict)

        return report_security_selection    