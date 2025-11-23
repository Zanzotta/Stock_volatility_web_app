import streamlit as st
import yfinance as yf
import pandas as pd
from hmmlearn import hmm
import plotly.graph_objects as go

st.title("Historical stock's volatility")
st.subheader("Intro")
st.write("""**One of the most frequently asked questions in stock selection is determining the optimal 
            entry price. This is a challenging problem because we lack visibility into other market 
            participants' strategies, meaning we cannot reliably estimate future demand for the stock. 
            A pragmatic approach is to purchase shares during periods of low volatility. While this 
            strategy does not guarantee buying at the absolute lowest price, it significantly reduces 
            the risk of entering at a local peak and subsequently watching the price fluctuate wildly 
            in the following weeks.
            To support this timing strategy, we have developed a classification model that identifies 
            two distinct market states: low-volatility and high-volatility periods. The model's 
            performance in distinguishing between these volatility regimes based on historical price 
            data is demonstrated below.**""")




#--------------------------------#
# Interactive inputs
#--------------------------------#
with st.container(border=True):
    ticker = st.text_input("Ticker", 'AAPL')
    start_date = st.date_input("From", pd.to_datetime("2015-01-01"))

#--------------------------------#
# Data
#--------------------------------#
df = yf.download(ticker, start=start_date)['Close']
df['weekday'] = df.index.dayofweek

fridays = df[df['weekday'] == 4].copy()
fridays['returns'] = fridays[ticker].pct_change() * 100
fridays = fridays[[ticker, 'returns']].iloc[1:]

#--------------------------------#
# HMM
#--------------------------------#
returns_array = fridays['returns'].values.reshape(-1, 1)

model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=10000, random_state=2606)
model.fit(returns_array)
states = model.predict(returns_array)

var_0 = fridays[states == 0]['returns'].var()
var_1 = fridays[states == 1]['returns'].var()
if var_0 > var_1:
    labels = {0: 'High Vol', 1: 'Low Vol'}
else:
    labels = {1: 'High Vol', 0: 'Low Vol'}

fridays['state'] = pd.Categorical(states).rename_categories(labels)

#--------------------------------#
# Chart
#--------------------------------#
tab1, tab2 = st.tabs(["Chart", "Dataframe"])
fig = go.Figure()
for i in range(len(fridays) - 1):
    color = 'yellow' if fridays['state'].iloc[i] == 'High Vol' else 'blue'
    fig.add_trace(go.Scatter(
        x=fridays.index[i:i+2],
        y=fridays['returns'].iloc[i:i+2],
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False
    ))

fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='lines',
    line=dict(color='yellow', width=2),
    name='High Volatility'
))
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='lines',
    line=dict(color='blue', width=2),
    name='Low Volatility'
))

fig.update_layout(height=450, xaxis_title='Date', yaxis_title='Returns')
tab1.plotly_chart(fig, use_container_width=True)
tab2.dataframe(fridays, height=450, use_container_width=True)

st.subheader("Model Insight")
st.write("""**The model estimates volatility states by analyzing weekly stock returns calculated 
    from Friday closing prices. This weekly sampling approach serves two important purposes. 
    First, it filters out the daily noise and short term price fluctuations that can obscure 
    the underlying volatility trend we are trying to identify. 
    Second, using Friday as our anchor point provides consistency in our measurements. Since 
    markets are closed on weekends, Friday prices represent the market's final assessment of 
    value for that week, incorporating all the information and trading activity from the 
    previous five days.
    The developed Model then examines patterns in these weekly returns to classify whether 
    the stock is currently in a low volatility state (characterized by smaller and more predictable 
    price movements) or a high volatility state (marked by larger swings).**""")

st.subheader("Conclusion")
st.write("""**While classifying historical volatility patterns is valuable, stock picking requires 
            forward looking forecasts rather than backward looking analysis. Investment 
            decisions must be made before events occur, so predictive models that estimate 
            future volatility are the goal. This forecasting 
            capability will be implemented in future versions.**""")




#streamlit run app.py

