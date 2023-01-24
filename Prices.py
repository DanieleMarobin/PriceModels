import os
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
    def read_security_list(sec_list=[], parallel=None, max_workers=500, service=None):
        fo={}
        if parallel is None:
            for sec in sec_list:
                fo[sec] = read_security(sec, service)

        elif parallel=='thread':
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results={}
                for sec in sec_list:
                    results[sec] = executor.submit(read_security, sec, service)
            
            for key, res in results.items():
                fo[key]=res.result()

        elif parallel=='process':
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results={}
                for sec in sec_list:
                    results[sec] = executor.submit(read_security, sec, service)
            
            for key, res in results.items():
                fo[key]=res.result()

        return fo
    
    def read_security(sec, service=None):
        folder=SEC_DIR

        if '_' not in sec:
            sec=sec+'_0'
        if '.csv' not in sec:
            sec=sec+'.csv'

        file = folder+ sec

        df = gd.read_csv(file_path=file, service=service, parse_dates=['date'], dayfirst=True, index_col='date')

        return df

    def read_calendar():
        file = SEC_DIR+ 'all_calendar.csv'
        df = gd.read_csv(file_path=file, parse_dates=['date'], dayfirst=True, index_col='date')


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

# Accessories
if True:
    def info_maturity(sec):
        year= int(sec.split('_')[1])
        letter= sec.split('_')[0][-1]
        month=month_from_letter(letter)
        return dt(year,month,1)

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