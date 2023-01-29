# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:43:24 2020

@author: Yung
"""
from twelvedata import TDClient
import json
import os

file = open("../td_secret.json")
keys = json.load(file)
td = TDClient(apikey=keys["key"])

def td_get(sym:str, interval:str, size:int, timezone:str):
    try:
        return td.time_series(symbol=sym, 
                              interval=interval,
                              outputsize=size,
                              timezone=timezone).as_pandas() 
    except ValueError:
        print("Check Value Formats")
        