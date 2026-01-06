import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="yfinance 차트", layout="wide")
st.title("yfinance 종목 차트 보기 (Streamlit 차트)")

ticker = st.text_input(
    "종목 티커 입력 (예: AAPL, MSFT, 005930.KS, 000660.KS)",
    value="AAPL"
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    period = st.selectbox("기간(period)", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
with col2:
    interval = st.selectbox("간격(interval)", ["1d", "1wk", "1mo"], index=0)
with col3:
    price_col = st.selectbox("가격 컬럼", ["Close", "Open", "High", "Low"], index=0)
with col4:
    st.write("이동평균선")
    ma_5 = st.checkbox("MA5", value=True)
    ma_20 = st.checkbox("MA20", value=True)
    ma_60 = st.checkbox("MA60", value=False)

btn = st.button("차트 조회")

@st.cache_data(ttl=300)
def load_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column"
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # 단일 티커여도 환경에 따라 MultiIndex가 생길 수 있어 방어
    if isinstance(df.columns, pd.MultiIndex):
        if len(set(df.columns.get_level_values(1))) == 1:
            df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

if btn:
    t = ticker.strip()
    if not t:
        st.error("티커를 입력하세요.")
        st.stop()

    df = load_ohlcv(t, period, interval)
    if df.empty:
        st.warning("데이터가 없습니다. 티커가 올바른지 확인하세요. (예: 005930.KS)")
        st.stop()

    st.subheader("데이터 미리보기")
    st.dataframe(df.tail(20), use_container_width=True)

    df2 = df.set_index("Date").copy()

    # 이동평균선 계산(선택된 가격 컬럼 기준)
    if ma_5:
        df2["MA5"] = df2[price_col].rolling(5).mean()
    if ma_20:
        df2["MA20"] = df2[price_col].rolling(20).mean()
    if ma_60:
        df2["MA60"] = df2[price_col].rolling(60).mean()

    st.subheader(f"가격 차트 (Streamlit) - {price_col} + 이동평균선")
    # Streamlit 내장 차트는 여러 컬럼을 동시에 그릴 수 있음
    cols_to_plot = [price_col] + [c for c in ["MA5", "MA20", "MA60"] if c in df2.columns]
    st.line_chart(df2[cols_to_plot], height=350)

    if "Volume" in df2.columns:
        st.subheader("거래량 차트 (Streamlit)")
        st.bar_chart(df2[["Volume"]], height=250)
