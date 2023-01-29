# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:40:26 2020

@author: Yung
"""
import requests
import pandas as pd
import json
import datetime


class td_api_tools():
    def __init__(self, filepath):
        self.api_file = open(filepath)
        self.keys = json.load(self.api_file)["key"]
        
    def get_price_history(self, symbol:str, **kwargs) -> str:
        """
    
        Parameters
        ----------
        symbol : str
            DESCRIPTION.
        **kwargs : TYPE
            keyword arguments MUST match api call parameters
    
        Returns
        -------
        Pandas DataFrame of API Call
    
        """
        try:
            #initialize base_url and request parameters
            base_url = r"https://api.tdameritrade.com/v1/marketdata/{}/pricehistory".format(symbol) #base url and format with input string
            payload={"apikey": self.keys} #define/initiate search  param
    
            for key, values in kwargs.items(): #update kwargs
                if key == "startDate" or key=="endDate":
                    values = self.datetime_to_unix(values)
                payload[key] = values
            # get requests
            content = requests.get(url=base_url, params=payload) #get item content
        
            return self.apiout_to_df(content.json())
        except ValueError:
            print("Check Value Formats")
            
    def datetime_to_unix(self, time:str):
        """
    
        Parameters
        ----------
        time : str
            Time in YYYY-MM-DD HH:MM:SS format
    
        Returns
        -------
        UNIX time converted from datetime
    
        """
        dates = pd.to_datetime(time)
        check_time = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        return str(check_time)+"000"
    
    def apiout_to_df(self, apiout:dict)->pd.DataFrame:
        out_df = pd.DataFrame(apiout["candles"])
        out_df["to_datetime"] = out_df.datetime.apply(lambda x: int(str(x)[:-3]))
        out_df.to_datetime = out_df.to_datetime.apply(lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S.%f"))
        out_df.datetime = out_df.to_datetime
        out_df.index = out_df.datetime
        out_df.drop(["datetime", "to_datetime"], inplace=True, axis=1)
        return out_df