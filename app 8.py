import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

st.title("AI株価予測アプリ（1日先）")

ticker = st.text_input("銘柄コードを入力（例: AAPL, GOOG, TSLA）", "AAPL")

@st.cache_data
def load_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=60)
    data = yf.download(ticker, start=start, end=end)
    data = data[["Close"]].dropna()
    data["Days"] = range(len(data))
    return data

if ticker:
    try:
        df = load_data(ticker)
        X = df[["Days"]]
        y = df["Close"]
        model = LinearRegression()
        model.fit(X, y)
        next_day = [[len(df)]]
        predicted_price = model.predict(next_day)[0]
        st.line_chart(df["Close"])
        st.success(f"{ticker} の翌日の予測株価: ${predicted_price:.2f}")
    except Exception as e:
        st.error("データ取得に失敗しました。銘柄コードを確認してね。")
