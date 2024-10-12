import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from tradingview_screener import Query, Column, constants, Scanner
import warnings
warnings.filterwarnings("ignore")

st.title("Stock Screening")

def get_screen_result():
    row,df = Query().select('close','Value.Traded').where(
             Column('SMA20') > Column('SMA120')
             ,Column('SMA60') > Column('SMA120')
             ,Column('close') > Column('SMA200')
             ,Column('exchange').isin(["NYSE","NASDAQ"])
             ,Column('RSI9') < 35
        ,Column('Value.Traded|1W') > 10000000).limit(1000).get_scanner_data()
    df["Amount"] = df["Value.Traded"]/1000000
    df = df.drop("Value.Traded",axis=1)
    df["ticker"] = df["ticker"].apply(lambda x: x.split(":")[1].strip())

    ticker_lst = df["ticker"].to_list()
    all_data = yf.download(ticker_lst,start = "2020-01-01",progress=False)
    tick_lst = all_data["Close"].columns
    to_pick_lst = []

    for tick in tick_lst:
        try:
            ticker_close = all_data["Adj Close"][[tick]]
            ticker_close['MA20'] = ticker_close[tick].rolling(window=20).mean()
            ticker_close['Bias20'] = ((ticker_close[tick] - ticker_close['MA20'])/ticker_close['MA20'])*100
            ticker_close = ticker_close.dropna()
            left_threshold = np.percentile(ticker_close['Bias20'], 10)
            right_threshold = np.percentile(ticker_close['Bias20'], 90)
            if ticker_close['Bias20'][-1] <= left_threshold:
                to_pick_lst.append(tick)
        except:
            pass
    result = df[df["ticker"].isin(to_pick_lst)]
    return result

def get_screen_result2(result_df):
    wash_indicator = []
    get_in_indicator = []

    for tic in result_df["ticker"].values:
        stock = yf.download(tic,start = "2020-01-01",progress=False)
        stock.columns = list(map(lambda x : x.lower(),stock.columns))
        stock["close"] = stock["adj close"]
        stock = stock.drop("adj close",axis=1)
        stock["var1"] = (stock.low + stock.open + stock.close + stock.high)/4
        stock["var1"] = stock["var1"].shift(1)
        stock["var2_1"] = abs(stock.low-stock.var1)
        var_1 = stock.copy()
        var_1["close"] = stock["var2_1"]
        stock["var2_2"] = var_1["close"].rolling(window=13).mean()
        stock["var2_3"] = stock.low - stock.var1
        stock["var2_4"] = stock["var2_3"].apply(lambda x: max(x,0))
        var_2 = stock.copy()
        var_2["close"] = stock["var2_4"]
        stock["var2_5"] = var_2["close"].rolling(window=10).mean()
        stock["var2"] = stock["var2_2"]/stock["var2_5"]
        var_3 = stock.copy()
        var_3["close"] = stock["var2"]
        stock["var3"] = var_3["close"].ewm(span=10, adjust=False).mean()
        stock["var4"] = stock["low"].rolling(window=33).min()
        stock["var5_1"] = stock.apply(lambda x: x["var3"] if x["low"] <= x["var4"] else 0,axis=1)
        var_4 = stock.copy()
        var_4["close"] = stock["var5_1"]
        stock["var5"] = var_4["close"].ewm(span=3, adjust=False).mean()
        stock["var5_previous"] = stock["var5"].shift(1)
        stock["get_in"] = stock.apply(lambda x: x["var5"] if x["var5"] > x["var5_previous"] else 0,axis=1)
        stock["wash"] = stock.apply(lambda x: x["var5"] if x["var5"] < x["var5_previous"] else 0,axis=1)
        stock["suck"] = stock["get_in"] + stock["wash"]

        if stock["wash"].tail(5).sum() > 0.0001:
            wash_indicator.append(1)
        else:
            wash_indicator.append(0)

        if stock["get_in"].tail(5).sum() > 0.0001:
            get_in_indicator.append(1)
        else:
            get_in_indicator.append(0)

    result_df["wash_5days_go"] = wash_indicator
    result_df["getin_5_days_ready"] = get_in_indicator
    return result_df

result_df = get_screen_result()
result_df2 = get_screen_result2(result_df)

st.dataframe(result_df2
             ,use_container_width=True
             ,hide_index=True)

