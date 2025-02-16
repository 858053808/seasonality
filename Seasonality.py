import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px

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

data = yf.download(ticker, interval='1mo',progress=False)
data["change"] = round(100 * data["Close"].pct_change(),1)
data.dropna(inplace=True)
data.reset_index(inplace=True)

def get_dist(data,month,years_to_test=0):
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

plt,info = get_dist(data = data,month = month,years_to_test=years_to_test)

st.pyplot(plt)
st.write("Information:",info)

def get_monthly_chart(data,years_to_test=0):
    data["month_name"] = data["Date"].dt.month_name().apply(lambda x: x[:3])
    data["month"] = data["Date"].dt.month
    data["year"] = data["Date"].dt.year

    pvt_data = data[["year","month","change"]].pivot(index='year', columns='month', values='change')
    pvt_data = pvt_data.rename_axis(None, axis=1).reset_index()
    pvt_data.columns = ["Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pvt_data = pvt_data.set_index('Year')
    
    if years_to_test != 0:
        pvt_data = pvt_data[-1*years_to_test:]
    
    # Create a heatmap
    fig = px.imshow(pvt_data,
                    color_continuous_scale=[(0, "red"), (0.5, "white"), (1, "green")],
                    labels={'x': 'Month', 'y': 'Year', 'color': '%change'},
                    zmin=pvt_data.min().min(),
                    zmax=pvt_data.max().max(),
                    text_auto=True,
                    width=1500,
                    height=1000)

    # Show the figure
    fig.update_yaxes(tickvals=list(pvt_data.index), ticktext=list(pvt_data.index))
    fig.update_layout(
        modebar_remove=["toImage", "zoom", "pan", "resetScale", "hover"],
        showlegend=False,
        yaxis_title=None
    )
    return fig ,pvt_data

fig, data = get_monthly_chart(data,years_to_test)
avg = pd.DataFrame(data.mean().apply(lambda x: round(x,2))).transpose()

st.header("Monthly Performance of %s"%(ticker))
st.dataframe(avg,hide_index=True)
st.plotly_chart(fig)
