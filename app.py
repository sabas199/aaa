# app.py
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

st.title("株価予想アプリ")

ticker = st.text_input("銘柄コード（例: 7203.T）", "7203.T")
days = st.slider("過去データ（日数）", 30, 365, 90)
future_days = st.slider("予測する未来日数", 7, 60, 30)

if st.button("予測する"):
    df = yf.download(ticker, period=f"{days}d")
    df = df.reset_index()
    df["Date_ordinal"] = df["Date"].map(pd.Timestamp.toordinal)

    X = df["Date_ordinal"].values.reshape(-1, 1)
    y = df["Close"].values
    model = LinearRegression().fit(X, y)

    future_dates = [df["Date"].max() + pd.Timedelta(days=i) for i in range(1, future_days+1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    preds = model.predict(future_ordinals)

    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], y, label="過去の株価")
    plt.plot(future_dates, preds, label="予測", linestyle="--")
    plt.xlabel("日付")
    plt.ylabel("株価")
    plt.title(f"{ticker} の株価予想")
    plt.legend()
    st.pyplot(plt)
