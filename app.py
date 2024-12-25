import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import ta
from finvizfinance.quote import finvizfinance
from statsmodels.tsa.statespace.sarimax import SARIMAX
import holidays
from langchain_community.llms import Ollama

# Page configuration
st.set_page_config(
    page_title="🤖 AI Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stMetric .label { font-size: 18px !important; }
    .stAlert { padding: 20px !important; }
    .stProgress .st-bo { height: 20px !important; }
    [data-testid="stMetricDelta"] svg { display: none; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(model='llama3')

###################
# Helper Functions #
###################

def classify_sentiment(title):
    output = st.session_state.llm.invoke(
        f"Classify the sentiment as 'POSITIVE' or 'NEGATIVE' or 'NEUTRAL' with just that one word only: {title}"
    )
    return output.strip()

def fetch_stock_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError("No data found for the specified ticker")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_news_sentiment(ticker):
    with st.spinner('Analyzing news sentiment...'):
        stock = finvizfinance(ticker)
        news_df = stock.ticker_news()
        news_df['Title'] = news_df['Title'].str.lower()
        news_df['sentiment'] = news_df['Title'].apply(classify_sentiment)
        news_df['sentiment'] = news_df['sentiment'].str.upper()
        
        # Fix date parsing
        def convert_date(date_str):
            if 'Today' in date_str:
                return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            return pd.to_datetime(date_str)
        
        news_df['Date'] = news_df['Date'].apply(convert_date)
        return news_df
        
def calculate_technical_indicators(data):
    indicators = {}
    indicators['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    indicators['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    indicators['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    indicators['MACD'] = ta.trend.macd_diff(data['Close'])
    
    # Fix for Bollinger Bands
    indicator_bb = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
    indicators['BB_upper'] = indicator_bb.bollinger_hband()
    indicators['BB_middle'] = indicator_bb.bollinger_mavg()
    indicators['BB_lower'] = indicator_bb.bollinger_lband()
    
    return indicators

def create_price_chart(data, chart_type, selected_indicators):
    fig = go.Figure()
    
    # Main price data
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            name='Price',
            line=dict(color='#2962FF')
        ))
    
    # Add indicators
    if 'Bollinger Bands' in selected_indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper',
                               line=dict(color='rgba(250, 128, 114, 0.5)', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower',
                               line=dict(color='rgba(250, 128, 114, 0.5)', dash='dash'),
                               fill='tonexty'))
    
    if 'SMA' in selected_indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20',
                               line=dict(color='#FF6B6B')))
    
    if 'EMA' in selected_indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], name='EMA 20',
                               line=dict(color='#4CAF50')))
    
    fig.update_layout(
        title=dict(text=f"Price Chart with Technical Indicators", x=0.5),
        yaxis_title="Price (USD)",
        template='plotly_dark',
        height=600,
        hovermode='x unified',
        showlegend=True
    )
    return fig

###############
# Main App UI #
###############

# Sidebar
with st.sidebar:
    st.title("🤖 AI Stock Analysis")
    st.markdown("---")
    
    # Input parameters
    ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()
    
    col1, col2 = st.columns(2)
    with col1:
        time_period = st.selectbox('Time Period', 
                                 ['1d', '1wk', '1mo', '3mo', '6mo', '1y'])
    with col2:
        chart_type = st.selectbox('Chart Type', 
                                ['Line', 'Candlestick'])
    
    indicators = st.multiselect('Technical Indicators',
                              ['SMA', 'EMA', 'RSI', 'MACD', 'Bollinger Bands'],
                              default=['SMA', 'RSI'])
    
    analyze_button = st.button('🔄 Analyze Stock', use_container_width=True)
    
    # Watchlist
    st.markdown("---")
    st.subheader("📊 Market Watchlist")
    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    for symbol in watchlist:
        try:
            # Get recent data
            recent_data = yf.download(symbol, period='2d', interval='1d')
            if len(recent_data) >= 1:
                current_price = recent_data['Close'].iloc[-1]
                daily_change = 0  # Default to 0 if not enough data
                if len(recent_data) >= 2:
                    daily_change = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[-2] - 1) * 100
                
                st.metric(
                    symbol,
                    f"${current_price:.2f}",
                    f"{daily_change:+.2f}%",
                    delta_color="normal"
                )
            else:
                st.error(f"Insufficient data for {symbol}")
        except Exception as e:
            st.error(f"Could not fetch data for {symbol}")

# Main content
if analyze_button:
    try:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📈 Technical Analysis", "📰 Sentiment Analysis", "🔮 Predictions"])
        
        # Fetch data
        data = fetch_stock_data(ticker, time_period, '1d')
        if data is None:
            st.stop()
            
        # Calculate indicators
        for name, indicator in calculate_technical_indicators(data).items():
            data[name] = indicator
        
        # Tab 1: Technical Analysis
        with tab1:
            # Display main metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            current_price = data['Close'].iloc[-1]
            price_change = 0
            price_change_pct = 0
            
            if len(data) >= 2:
                prev_close = data['Close'].iloc[-2]
                price_change = current_price - prev_close
                price_change_pct = (price_change / prev_close) * 100
            
            metrics_col1.metric("Current Price", f"${current_price:.2f}", 
                              f"{price_change_pct:+.2f}%")
            metrics_col2.metric("24h High", f"${data['High'].iloc[-1]:.2f}")
            metrics_col3.metric("24h Low", f"${data['Low'].iloc[-1]:.2f}")
            metrics_col4.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
            
            # Price chart
            st.plotly_chart(create_price_chart(data, chart_type, indicators),
                          use_container_width=True)
            
            # Technical indicators detail
            if 'RSI' in indicators or 'MACD' in indicators:
                detail_col1, detail_col2 = st.columns(2)
                
                if 'RSI' in indicators:
                    with detail_col1:
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'],
                                                   name='RSI',
                                                   line=dict(color='#FF6B6B')))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                        fig_rsi.update_layout(title='RSI (14)',
                                           height=300,
                                           template='plotly_dark')
                        st.plotly_chart(fig_rsi, use_container_width=True)
                
                if 'MACD' in indicators:
                    with detail_col2:
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'],
                                                    name='MACD',
                                                    line=dict(color='#4CAF50')))
                        fig_macd.add_hline(y=0, line_dash="dash")
                        fig_macd.update_layout(title='MACD',
                                            height=300,
                                            template='plotly_dark')
                        st.plotly_chart(fig_macd, use_container_width=True)
        
        # Tab 2: Sentiment Analysis
        with tab2:
            news_df = get_news_sentiment(ticker)
            
            # Display sentiment summary
            pos_count = (news_df['sentiment'] == 'POSITIVE').sum()
            neg_count = (news_df['sentiment'] == 'NEGATIVE').sum()
            total_count = pos_count + neg_count
            
            if total_count > 0:
                sentiment_score = (pos_count / total_count) * 100
                
                st.markdown(f"### 📊 Sentiment Analysis")
                st.progress(sentiment_score / 100)
                st.markdown(f"Current sentiment score: **{sentiment_score:.1f}%** positive")
                
                # Recent news with sentiment
                st.markdown("### 📰 Recent News")
                for _, row in news_df.head(10).iterrows():
                    sentiment_icon = "🟢" if row['sentiment'] == "POSITIVE" else "🔴"
                    st.markdown(f"{sentiment_icon} **{row['Date'].strftime('%Y-%m-%d')}**: {row['Title']}")
            
        # Tab 3: Predictions
        with tab3:
            st.markdown("### 🔮 Price Predictions")
            
            # Calculate predictions using SARIMAX
            try:
                with st.spinner("Calculating predictions..."):
                    # Prepare data
                    data['returns'] = data['Close'].pct_change()
                    model = SARIMAX(data['returns'].dropna(), order=(1, 1, 1))
                    results = model.fit(disp=False)
                    
                    # Generate forecast
                    forecast = results.forecast(steps=5)
                    forecast_prices = data['Close'].iloc[-1] * (1 + forecast)
                    
                    # Display predictions
                    for i, (pred_date, pred_price) in enumerate(zip(pd.date_range(start=data.index[-1], periods=6)[1:],
                                                                  forecast_prices)):
                        change = ((pred_price / data['Close'].iloc[-1]) - 1) * 100
                        st.metric(
                            f"Predicted Price ({pred_date.strftime('%Y-%m-%d')})",
                            f"${pred_price:.2f}",
                            f"{change:+.2f}%"
                        )
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Created by Deep Charts")
