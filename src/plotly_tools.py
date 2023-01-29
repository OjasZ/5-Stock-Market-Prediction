# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:14:47 2021

@author: Yung
"""
import plotly.graph_objects as go
def make_candlestick(data, name, covid_show=False):
    chart_title = f"{name} Daily Prices"
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, 
                             open=data.open, 
                             high=data.high,
                             low=data.low,
                             close=data.close))
    fig.update_layout(title=chart_title,
                      yaxis_title="Close ($)")
    
    if covid_show:
        fig.add_shape(type="rect", x0="2020-02-20", x1="2020-03-23", y0=0, y1=1, xref="x", yref="paper", line=dict(color="rgba(255,0,0,0.5)"))
                
        fig.update_layout(annotations=[dict(x="2020-03-24", y=0.05, xref="x", yref="paper", xanchor="left",
                                    showarrow=False, text="COVID-19 Crash")])
    fig.show()
    return fig