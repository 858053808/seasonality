import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt

st.title("Hello")

st.subheader("Input a Stock Ticker: ")
ticker = st.text_input("NASDQ: ^IXIC ; SP500: ^GSPC (Yahoo Finance Ticker)", "^IXIC")

st.subheader("Select a month for analysis: ")
month = st.select_slider(
    "Select a month for analysis",
    options=list(range(1,13)),
    label_visibility = "hidden"
)
month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

years_to_test = int(st.text_input("Years for plotting (0 = all):", "0"))


st.header("%s Performance Distribution of %s"%(month_names[month],ticker))

def get_dist(ticker,month,years_to_test=0):
    data = yf.download(ticker, interval='1mo',progress=False)
    data["change"] = round(100 * data["Adj Close"].pct_change(),1)
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data["month"] = data["Date"].dt.month
    month_data = data[data["month"]==month]
    if years_to_test != 0:
        change_data = month_data["change"][-1*years_to_test:]
    else:
        change_data = month_data["change"]
    #information
    years = len(change_data)
    pos_prob = str(int((change_data > 0).sum()/years*100))+"%"
    neg_prob = str(int((change_data < 0).sum()/years*100))+"%"
    median = str(round(change_data.describe()["50%"],2))+"%"
    mean = str(round(change_data.describe()["mean"],2))+"%"
    current_val = data["change"].values[-1]
    current = str(current_val)+"%"
    up_prob = str(int((change_data > current_val).sum()/len(change_data)*100))+"%"
    down_prob = str(int((change_data < current_val).sum()/len(change_data)*100))+"%"

    info = """
    
    No. of years tested: %d
    Month: %d
    +ve Probability: %s
    -ve Probability: %s
    Median: %s
    Average : %s
    
    Current : %s
    Probability up further: %s
    Probability down further: %s
    
    """%(years,month,pos_prob,neg_prob,median,mean,current,up_prob,down_prob)

    # Plotting
    plt.figure(figsize=(10, 6),dpi=600)
    n, bins, patches = plt.hist(change_data,bins=np.arange(min(change_data)//6*6,(max(change_data)//6+1)*6,2), edgecolor='black')

    # Color the bins
    for patch, left_side in zip(patches, bins[:-1]):
        if left_side < 0:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('green')

    # Add a vertical red line at 0
    plt.axvline(0, color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(data["change"].values[-1], color='black', linestyle='dashed', linewidth=2,label="Current MTD: %s"%(current))

    # Labels and title
    #plt.title('Monthly Performance distribution')
    plt.xticks(bins)
    plt.legend(loc=0)
    plt.xlabel('% Change')
    plt.ylabel('Frequency')
    plt.grid(False)
    return plt,info

plt,info = get_dist(ticker = ticker,month = month,years_to_test=years_to_test)

st.pyplot(plt)
st.write("Information:",info)
