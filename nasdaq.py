# Utility to scrape NASDAQ IPO lists

import pandas as pd

import datetime
from datetime import timedelta
from collections import OrderedDict

def get_ipo_list(start_date, end_date=datetime.datetime.today().strftime('%Y-%m-%d')):
    """Scrape NASDAQ IPO lists, returns DataFrame

    arguments:
    start_date -- %Y-%m-%d
    end_date -- %Y-%m-%d
    """

    #make dates range
    date_range = [start_date, end_date]
    start, end = [datetime.datetime.strptime(_, "%Y-%m-%d") for _ in date_range]
    date_dict = OrderedDict(((start + timedelta(_)).strftime(r"%Y-%m"), None) for _ in range((end - start).days)).keys()
    print('date range:', date_dict)
    
    #scrape
    df_symbols = pd.DataFrame()

    for x in date_dict:
        df_symbols = df_symbols.append(pd.read_html('https://www.nasdaq.com/markets/ipos/activity.aspx?tab=pricings&month=' + x)[0], ignore_index=True)

    df_symbols.index = df_symbols['Symbol']        
    df_symbols.drop_duplicates('Symbol', inplace=True)
    df_symbols.dropna(inplace=True, subset=['Symbol'])    
    
    return df_symbols