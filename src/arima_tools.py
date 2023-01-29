# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 19:34:37 2021

@author: Yung
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



class arima_tools():
    def __init__(self):
        self._ = ""
    
    
    def moving_averages(self, input_df, days_list:list = [10, 50], col:str="close"):
        """
        This function returns a dataframe output_df with specified col and its moving averages per specified by days_list
        
        Input
        ========
        input_df: Pandas DatafFrame of stock prices. It is recommended that index is datetime and the columns contain "open, high, low, and close"
        
        days_list: list of integers to create moving averages. DEFAULT = [10, 50]
        
        col: name of the columns to create MA from the input_df. This must be part of input_df.columns. DEFAULT = "close"
        
        Output
        ========
        output_df: A pandas dataframe with specified col and its moving averages per specified by days_list
        """    
        
        output_df = input_df.copy() # to not disturb original data
        out_ma_columns = [col] #intial lize output column list with input column
    
        for day in days_list:
            column = f"MA({day})"
            output_df[column] = output_df[col].rolling(day).mean() #make rolling/moving averages
            out_ma_columns.append(column) #build output column list
        
        return output_df[out_ma_columns].dropna() #output df with no errors

    def differences (self, input_df, nlags=1):
        """
        This function returns a dataframe output_df with lags of all columns on dataframe specified by orders of lags.
        
        Input
        ========
        input_df: Pandas DatafFrame of stock prices. It is recommended that index is datetime and the columns contain "open, high, low, and close"
        
        lags: # of lags DEFAULT = 1
        
        Output
        ========
        output_df: A pandas dataframe with lags of all columns
        
        """
        output_df = input_df.copy() #to not disturb input data
        input_columns = output_df.columns

        
        # for all lags

        for col in input_columns:
            for i in range(1, nlags+1):
                col_name = "{}_lag_{}".format(col, i)
                output_df[col_name] = output_df[col].diff(i)
        return output_df


    
    def adf_test(self, timeseries:pd.Series, lags="AIC", alpha=0.05):
        """
        This function returns a dataframe output_df with specified col and its moving averages per specified by days_list
        
        Input
        ========
        input_df: Pandas DatafFrame of stock prices. It is recommended that index is datetime and the columns contain "open, high, low, and close"
        
        days_list: list of integers to create moving averages. DEFAULT = [10, 50]
        
        col: name of the columns to create MA from the input_df. This must be part of input_df.columns. DEFAULT = "close"
        
        Output
        ========
        output_df: A pandas dataframe with specified col and its moving averages per specified by days_list
        
        """
        
        dftest = adfuller(timeseries, autolag=lags) #adf_result of this time series
        dfoutput = pd.Series([timeseries.name], index=["series name"]) #get inputname
        dfoutput = pd.concat([dfoutput, pd.Series(dftest[0:4], index=["Test Statistic", 
                                                                      "p-value", 
                                                                      "#Lags Used", 
                                                                      "Number of Observations Used"])]) #get adf result values
        
        #================Reject Null Hypothesis based on p-value and alpha==========================#
        # data has potential to be stationary if this is true
        if dfoutput["p-value"] <= alpha:
            dfoutput["reject null hypothesis"] = True
        else:
            dfoutput["reject null hypothesis"] = False
        
        
        
        for key,value in dftest[4].items():
            dfoutput["Critical Value (%s)"%key] = value #add critical value
            
        return dfoutput
    
    def find_lag_stationary(self, input_series, nlags=5):
        """

        Parameters
        ----------
        input_series : pd.Series 
            series to auto determine ARIMA lag d.
        nlags : int, optional
            number of lags to check up to . The default is 5.

        Returns
        -------
        TYPE
            first lag to be stationary with adf on differencing.

        """
        if self.adf_test(input_series)["reject null hypothesis"]:
           return 0
        for i in range (1, nlags+1):
            if self.adf_test(input_series.diff(i).dropna())["reject null hypothesis"]:
                break
        return i
        
    def pacf_order(self, data, name="", columns:list=["close"], nlags=20, plot=False):
        """
        This function returns a dataframe output_df with specified col and its moving averages per specified by days_list
        
        Input
        ========
        input_df: Pandas DatafFrame of stock prices. It is recommended that index is datetime and the columns contain "open, high, low, and close"
        
        days_list: list of integers to create moving averages. DEFAULT = [10, 50]
        
        col: name of the columns to create MA from the input_df. This must be part of input_df.columns. DEFAULT = "close"
        
        Output
        ========
        output_df: A pandas dataframe with specified col and its moving averages per specified by days_list
        
        """
        lag_list = []
        for col in columns: # for specified columns
            # PACF significance and Results
            pacf_sig = 2/np.sqrt(len(data[col]))
            pacf_results = pacf(data[col], nlags=nlags)
            pacf_df = pd.DataFrame(pacf_results) 
            pacf_df.columns = ["sig"]
    
            # Differencing the significance to create keys for sorting by largest lag change
            pacf_df_diff = abs(pacf_df.sig.diff())
            pacf_df_diff.sort_values(ascending=False, inplace=True)
            sig_lag = pacf_df_diff.index[0] #since the lag we want is in order (integer order)
    
            if pacf_df.sig[sig_lag-1] > pacf_sig:
                #store lag_list
                lag_list.append(sig_lag)
        if plot:
            for col in columns:
                plot_title = "{} ACF and PACF".format(name)
                fig, ax = plt.subplots(1,2, figsize=(12,8))
                plot_acf(data[col].values, ax=ax[0], alpha=0.05);
                plot_pacf(data[col], ax=ax[1], alpha=0.05);
                fig.suptitle(plot_title, size=18)
                
       #return a dataframe of results     
        return dict(zip(columns, lag_list))
    
    def ts_tts(self, data, ntest=20):
        """
        Convenient splitter for timeseries testing

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        ntrain : TYPE, optional
            DESCRIPTION. The default is 230.
        ntest : TYPE, optional
            DESCRIPTION. The default is 20.
        nholdout : TYPE, optional
            DESCRIPTION. The default is 5.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        return data[:-ntest], data[-ntest:]
    
    def month_breakdown_grid(self, input_df, name, savepath="", n_rows=4, n_cols=3, figsize=(24, 24), crossover=True):
        if crossover:
            input_df = self.crossovers(input_df)
        
        total = n_rows * n_cols
        year = input_df.index[0].year
    
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    
        for i in range(total): #for loop to iterate viz
            # ========= format visualization position =====
            startmonth,  endmonth = i+1, i+2
            rows, cols = i//n_cols, i%n_cols
    
            # ================= format date ===============
            if startmonth < 10:
                startmonth = "".join(["0", str(startmonth)])
            else:
                startmonth = str(startmonth)
    
            if endmonth < 10:
                endmonth = "".join(["0", str(endmonth)])
            else:
                endmonth = str(endmonth)
    
            start = "-".join([str(year), startmonth, "01"])
            end = "-".join([str(year), endmonth, "01"])
    
            # ============== parse data =================
            if i < 11:
                current_plot_df = input_df[(input_df.index > start) & (input_df.index < end)]
            else:
                current_plot_df = input_df[input_df.index > start]
    
            # ============== plotting =================
            for col in current_plot_df.columns:
                # graph items
                xticks = [current_plot_df.index[0], current_plot_df.index[-1]]
                xlabels = [str(date)[:10] for date in xticks]
                axtitle = "{} Prices".format(xlabels[0][:7])
                
                if col == "indicator" and crossover:
                    true_df = current_plot_df[current_plot_df.indicator == True].close
                    ax[rows][cols].scatter(x=true_df.index, y=true_df.values, label="cross-overs", marker="d", color="red")
                else:
                # plot
                    ax[rows][cols].plot(current_plot_df[col], label=col)
    
                # more concise x axis
                ax[rows][cols].set_xticks(xticks)
                ax[rows][cols].set_xticklabels(xlabels)
                # add ax title
                ax[rows][cols].set_title(axtitle, size=20)
                ax[rows][cols].legend(bbox_to_anchor=(0, -0.01, 1, -.05),
                                      loc="upper left",
                                      ncol=3,
                                      mode="expand")
    
            fig.suptitle(f"{name} {year} Price Per Month", size=36, y=0.92)
        if len(savepath):
            plt.savefig(savepath)
        plt.show()
        return fig, ax
    
    def rmse(self, true, input_series):
        return np.sqrt(mean_squared_error(true, input_series))
    
    def crossovers(self, input_df, columns=0):
        start_df = []

        for i in range(len(input_df)):
            close = input_df.iloc[i,:].close 
            max_val = input_df.iloc[i,:].values.max()
            if close < max_val:
                 start_df.append(input_df.iloc[i,:].name)
        
        start_df = pd.DataFrame(start_df)
        
        start_df["indicator"] = start_df[columns].apply(lambda x: True)
        start_df["datetime"]= start_df[columns]
        start_df.index = start_df.datetime
        start_df.drop(columns=[columns, "datetime"], inplace=True)
        
        start_df = pd.concat([input_df, start_df], axis=1)
        return start_df