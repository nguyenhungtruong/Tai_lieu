# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 06:43:01 2023

@author: Nguyen Hung Truong
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
# from vnstock import *
from datetime import datetime
import time
import math
import re
import sys
import random
import warnings
from openpyxl import load_workbook
warnings.filterwarnings("ignore")

"""
Input
"""
Data = pd.read_parquet(r'D:\Tool_Stock\Data\VNINDEX_01082023.parquet')
Data.sort_values(["TradingDate"],ascending=False,inplace = True)  
Data['Index'] = np.arange(0,len(Data))

"""
Usage Class
"""

class Indicator():
    def __init__(self,day):
        self.day = day
    # -day  diff là lấy số liền trước trừ số liền sau
    # -day shift là lùi dãy số lên trên
    def Returns(self,data,value):
        data.sort_values(["TradingDate"],ascending = False,inplace = True)
        data['Returns_{}_{}'.format(value,self.day)] = data[value].diff(periods = -self.day) / data[value].shift(-self.day) 
        return data
    def Rolling(self,data,value,method):
        data.sort_values(["TradingDate"],ascending = True,inplace = True)
        if method == 'Mean':
           data['MA_{}_{}'.format(value,self.day)] = data['{}'.format(value)].rolling(self.day).mean()
           data.sort_values(["TradingDate"],ascending = False,inplace = True)
           return data
        if method == 'Std':
           data['STD_{}_{}'.format(value,self.day)] = data['{}'.format(value)].rolling(self.day).std()
           data.sort_values(["TradingDate"],ascending = False,inplace = True) 
    def Pearson_correlation(self,data,delay,value_1,value_2):
        data.sort_values(["TradingDate"],ascending = False,inplace = True) 
        data.reset_index(inplace = True)
        data.drop(columns=['index'],inplace = True)
        list_corr = []
        for start in range(0,len(data)-self.day-delay):
             chain = data.loc[data.index.isin(pd.RangeIndex(start,start + self.day, step=1)),[value_1,value_2]]
             x = chain[value_1];y = chain[value_2]
             Sum_xy = sum((x-x.mean())*(y-y.mean()))
             Sum_x_squared = sum((x-x.mean())**2)
             Sum_y_squared = sum((y-y.mean())**2)      
             corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
             list_corr.append(corr)    
        list_corr = np.append(list_corr, np.repeat(np.nan,self.day))
        data['Corr_{}_{}_{}'.format(value_1,value_2,self.day)] = list_corr              
        return data  
    def Rsi(self,data,value):
        data.sort_values(["TradingDate"],ascending = False,inplace = True)
        data.reset_index(inplace = True)
        data.drop(columns=['index'],inplace = True)
        list_rsi = []
        for start in range(0,len(data)-self.day):
            chain = data.loc[data.index.isin(pd.RangeIndex(start,start + self.day, step=1)),[value]]
            chain['DIFF'] = chain[value].diff(periods = -1)
            Avr_Gain = chain.loc[chain.DIFF > 0 ]['DIFF'].sum() / self.day
            Avr_Loss = abs(chain.loc[chain.DIFF < 0 ]['DIFF']).sum() / self.day
            RS = Avr_Gain / Avr_Loss
            RSI = 100 - (100/(1+RS))
            list_rsi.append(RSI) 
        list_rsi = np.append(list_rsi, np.repeat(np.nan,self.day))
        data['RSI_{}_{}'.format(value,self.day)] = list_rsi
        return data
            
class Method_Trading():
    def __init__(self,buy1,buy2,sell1,sell2,tw1,tw2,sign1,sign2,sign3,sign4): # var
        self.buy1 = buy1
        self.buy2 = buy2
        self.sell1 = sell1
        self.sell2 = sell2
        
        self.tw1 = tw1
        self.tw2 = tw2
        
        self.sign1 = sign1
        self.sign2 = sign2
        self.sign3 = sign3
        self.sign4 = sign4

    def Backtest(self,data,riskfreerate):
        data['Condition'] = np.nan
        condition_buy = data[self.buy1] > data[self.buy2]
        condition_sell = data[self.sell1] > data[self.sell2]
        
        condition_sign1 = (np.sign(data['Returns_Close_{}'.format(self.tw1)]) == self.sign1) & (np.sign(data['Returns_Volume_{}'.format(self.tw1)]) == self.sign2)
        condition_sign2 = (np.sign(data['Returns_Close_{}'.format(self.tw2)]) == self.sign3) & (np.sign(data['Returns_Volume_{}'.format(self.tw2)]) == self.sign4)
        
        data.loc[condition_buy & condition_sign1 ,'Condition'] = 'Buy'  
        data.loc[condition_sell & condition_sign2,'Condition'] = 'Sell'
          
        data.sort_values(["TradingDate"],ascending=True,inplace = True)       
        data['Condition'].fillna(method="ffill",inplace = True)              
        data_buy = data.loc[data.Condition == 'Buy']       
        data_buy.sort_values(["TradingDate"],ascending=False,inplace = True)        
        data_buy['Gap'] = data_buy['Index'] - data_buy['Index'].shift(1)
        data_sell = data.loc[data.Condition == 'Sell']   
        data_sell.sort_values(["TradingDate"],ascending=False,inplace = True) 
        data_sell['Gap'] = data_sell['Index'] - data_sell['Index'].shift(1)
        data_use = data_buy.loc[(data_buy['Gap'] == 1)]

        data_use['Returns_Use'] = data_use['Returns'].map(lambda x : x+1)
        data_use['Returns_Riskfreerate'] = data_use['Returns'].map(lambda x : x - riskfreerate/252)
        Avr_Returns = 100*(data_use['Returns_Use'].prod()**(1/(round((data.TradingDate.max() - data.TradingDate.min()).days/365,0))) - 1)
        total_buy = len(data_buy.loc[~(data_buy['Gap'] == 1)])   
        total_sell = len(data_sell.loc[~(data_sell['Gap'] == 1)])  
        Sharpe_ratio = math.sqrt(252)*data_use.Returns_Riskfreerate.mean() / data_use.Returns_Riskfreerate.std()
        d = {'Total_Buy': [total_buy],'Total_Sell': [total_sell],'Sharpe_Ratio': [Sharpe_ratio],'Avr_Returns':[Avr_Returns],\
              'Buy_1':[self.buy1],'Buy_2':[self.buy2],'Sell_1':[self.sell1],'Sell_2':[self.sell2],'TW1':[self.tw1],'TW2':[self.tw2],\
              'Sign_1':[self.sign1],'Sign_2':[self.sign2],'Sign_3':[self.sign3],'Sign_4':[self.sign4]} 
        return pd.DataFrame(d)  
    
    

        
"""
Data Derivation
"""
Data_Use = Data.copy() 
# Caculate Returns One Days
Indicator(1).Returns(Data_Use,'Close') 
Data_Use.rename(columns={'Returns_Close_1': 'Returns'},inplace = True)

List_Day =  [5,10,20,60,120,240]         
for day in List_Day:
    print(day)
    Ob_Indicator = Indicator(day)
    Ob_Indicator.Rolling(Data_Use,'Close','Mean')
    # Ob_Indicator.Rolling(Data_Use,'Close','Std')
    # Ob_Indicator.Rolling(Data_Use,'Volume','Std')
    # Ob_Indicator.Rolling(Data_Use,'Volume','Mean')
    # Ob_Indicator.Returns(Data_Use,'Close')
    # Ob_Indicator.Returns(Data_Use,'Volume')
    # Ob_Indicator.Rsi(Data_Use,'Close')
    
    
    
    




# Define Indicator

TW = [5,10,20,60,120,240]

TW1 = [5,10,20,60,120]
TW2 = [10,20,60,120,240]

TW3 = [240,120,60,20,10]
TW4 = [120,60,20,10,5]

s1 = [1,-1]
s2 = [1,-1]
s3 = [1,-1]
s4 = [1,-1]
buy1 = ['MA_Close_{}'.format(i) for i in TW1]
buy2 = ['MA_Close_{}'.format(i) for i in TW2]
sell1 = ['MA_Close_{}'.format(i) for i in TW3]
sell2 = ['MA_Close_{}'.format(i) for i in TW4]
"""
Run
"""
Total = len(buy1) * len(buy2) * len(sell1) * len(sell2) * len(TW1)* len(TW2) * len(s1)* len(s2)* len(s3)* len(s4)

Result = pd.DataFrame()

count = 0
for x1 in buy1:
    for x2 in buy2:
        for x3 in sell1:
            for x4 in sell2:
                if (int(x1.split('_')[2]) >=  int(x2.split('_')[2])) | (int(x3.split('_')[2]) <=  int(x4.split('_')[2])):
                    continue                
                for x5 in TW1:
                    for x6 in TW2:
                        for x7 in s1:
                            for x8 in s2:
                                for x9 in s3:
                                    for x10 in s4:
                                        OB_Backtest = Method_Trading(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)
                                        Result = pd.concat([Result,OB_Backtest.Backtest(Data_Use.copy())])
                                        Result.sort_values(["Avr_Returns"],ascending=False,inplace = True)
                                        count = count + 1
                                        if count % 100 == 0:
                                            print(count*100/Total)

              
Result.to_parquet(r'D:\Tool_Stock\Result\Close_Volume_Indicator.parquet')        

"""
Estimated Drawdown Ratio
"""
import pandas as pd
import numpy as np

#Input dau vao
Data = pd.read_parquet(r'D:\Tool_Stock\Data\VNINDEX_01082023.parquet')
Data.sort_values(["TradingDate"],ascending=True,inplace = True)  
Data['Index'] = np.arange(0,len(Data))
Data_Use = Data.copy()

# Tao cac indicator
TW = [5,10,20,60,120,240]
def Returns(data,value,day):
    data.sort_values(["TradingDate"],ascending = True,inplace = True)
    data['Returns_{}_{}'.format(value,day)] = data[value].diff(periods = day) / data[value]
    data['Returns_{}_{}'.format(value,day)] = data['Returns_{}_{}'.format(value,day)].shift(periods=-day)
    return data
for i in TW:
    Returns(Data_Use,'Close',i)

for day in TW:
    print(day)
    Ob_Indicator = Indicator(day)
    Ob_Indicator.Rolling(Data_Use,'Close','Mean')

Data_Use['CONDITION'] = np.nan
Data_Use.loc[Data_Use['MA_Close_5'] < Data_Use['MA_Close_10'],'CONDITION'] = 1
Data_1 = Data_Use.loc[Data_Use['CONDITION'] == 1]

Result = pd.DataFrame()
for i in TW:
    Df = Data_1.loc[Data_1['Returns_Close_{}'.format(i)].notna()]
    Series = Df['Returns_Close_{}'.format(i)]
    d= {'Time_Window':[i],'Loss':[np.percentile(Series,10)],'Gain':[np.percentile(Series,90)] ,'Average':[np.percentile(Series,50)] }
    Result = pd.concat([Result,pd.DataFrame(d)])

