import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="yfinance 차트", layout="wide")

st.title("yfinance 종목 차트 보기")

ticker = st.text_input("종목 티커 입력 (예: AAPL, MSFT, 005930.KS, 000660.KS)", value="AAPL")

col1, col2, col3 = st.columns(3)
with col1:
    period = st.selectbox("기간(period)", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
with col2:
    interval = st.selectbox("간격(interval)", ["1d", "1wk", "1mo"], index=0)
with col3:
    price_col = st.selectbox("가격 컬럼", ["Close", "Open", "High", "Low"], index=0)

btn = st.button("차트 조회")

@st.cache_data(ttl=300)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # Date 컬럼이 DatetimeIndex가 아닌 경우도 있어 안전하게 변환
    df["Date"] = pd.to_datetime(df["Date"])
    return df

if btn:
    t = ticker.strip()
    if not t:
        st.error("티커를 입력하세요.")
        st.stop()

    df = load_data(t, period, interval)
    if df.empty:
        st.warning("데이터가 없습니다. 티커가 올바른지 확인하세요. (예: 005930.KS)")
        st.stop()

    st.subheader("데이터 미리보기")
    st.dataframe(df.tail(20), use_container_width=True)

    st.subheader("가격 차트")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df[price_col])
    ax.set_xlabel("Date")
    ax.set_ylabel(price_col)
    ax.set_title(f"{t} - {price_col} ({period}, {interval})")
    st.pyplot(fig, clear_figure=True)

    st.subheader("거래량 차트")
    if "Volume" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(df["Date"], df["Volume"])
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volume")
        ax2.set_title(f"{t} - Volume")
        st.pyplot(fig2, clear_figure=True)