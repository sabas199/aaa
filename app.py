import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

st.title("AI株価予測アプリ（1日先）")

# 銘柄の入力
ticker = st.text_input("銘柄コードを入力（例: AAPL, GOOG, TSLA）", "AAPL")

# 株価取得関数
@st.cache_data
def load_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=60)
    data = yf.download(ticker, start=start, end=end)
    data = data[["Close"]].dropna()
    data["Days"] = range(len(data))  # 日数を数値化
    return data

# 銘柄が入力されたら処理開始
if ticker:
    try:
        df = load_data(ticker)
        
        # 学習用データ
        X = df[["Days"]]
        y = df["Close"]
        
        # モデル作成・学習
        model = LinearRegression()
        model.fit(X, y)

        # 翌日の予測
        next_day = [[len(df)]]
        predicted_price = model.predict(next_day)[0]

        # グラフ表示
        st.line_chart(df["Close"])
        st.success(f
