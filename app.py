import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import stock_predictor as sp
import time

# --- Setup Page Config ---
st.set_page_config(
    page_title="Future Price Pro",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Styling (Colorful) ---
st.markdown("""
<style>
    /* Premium Colorful Theme */
    .stApp {
        background: linear-gradient(135deg, #e0f2f1 0%, #f3e5f5 100%);
        color: #333;
        font-family: 'Inter', sans-serif;
    }

    /* Vibrant Header */
    .header-container {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(106, 17, 203, 0.3);
        text-align: center;
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 400;
        margin-top: 10px;
    }

    /* Glassmorphism Cards */
    .card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.15);
    }

    /* Vibrant Multi-Color Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #ff4e50, #f9d423);
        color: #fff;
        border: none;
        padding: 1rem 2.5rem;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(255, 78, 80, 0.4);
        width: 100%;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 78, 80, 0.6);
        background: linear-gradient(45deg, #f9d423, #ff4e50);
    }

    /* Colorful Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricDelta"] svg {
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }

</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("""
<div class="header-container">
    <div class="header-title">🌈 Future Price Pro</div>
    <div class="header-subtitle">Intelligent Stock Analysis with Vibrant AI</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3893/3893114.png", width=60) # Generic trendy icon
    st.markdown("## ⚙️ **Control Panel**")
    
    TICKER_OPTIONS = {
        "AAPL": "Apple Inc. 🍎",
        "TSLA": "Tesla, Inc. 🚘",
        "MSFT": "Microsoft Corp. 💻",
        "GOOGL": "Alphabet Inc. 🔍",
        "AMZN": "Amazon.com 🛒",
        "NVDA": "NVIDIA Corp. 🎮",
        "META": "Meta Platforms ♾️",
        "NFLX": "Netflix, Inc. 🎬"
    }
    
    selected_key = st.selectbox(
        "Select Asset", 
        options=list(TICKER_OPTIONS.keys()),
        format_func=lambda x: f"{x} | {TICKER_OPTIONS[x]}",
        index=0
    )
    
    ticker = selected_key
    
    st.write("---")
    start_date = st.date_input("📅 Training Start Date", value=pd.to_datetime("2020-01-01"))
    st.caption("ℹ️ *Longer history = Slower training but potentially better accuracy.*")
    
    st.markdown("### 📊 **Model Specs**")
    st.info("**Algorithm:** Random Forest Regressor\n\n**Features:** Open, High, Low, Volume")

# --- Main Dashboard ---

col_main, = st.columns(1) # Centered layout concept

with col_main:
    if st.button("✨ GENERATE PREDICTION ✨", type="primary"):
        
        # --- Live Status Log ---
        status = st.status("🚀 **Initializing Neural Engine...**", expanded=True)
        
        try:
            # 1. Fetch
            status.write(f"📡 Fetching live market data for **{ticker}**...")
            time.sleep(0.8)
            data = sp.fetch_data(ticker, start_date=str(start_date))

            if data.empty:
                status.update(label="❌ Error: Data not found", state="error")
                st.error("Failed to retrieve data. Please try another ticker.")
            else:
                status.write(f"📥 Downloaded **{len(data):,}** historical records.")
                time.sleep(0.5)

                # 2. Process
                status.write("⚙️ Engineering features (OHLCV)...")
                X, y, dates = sp.prepare_data(data)
                
                # 3. Train
                status.write(f"🧠 Training AI Model on **{len(X)}** samples...")
                model, X_test, y_test = sp.train_model(X, y)
                status.write("✅ Model converged successfully.")
                
                status.update(label="✅ **Analysis Complete!**", state="complete", expanded=False)
                
                # 4. Predict
                predictions = model.predict(X_test)
                last_row = data.iloc[[-1]][['Open', 'High', 'Low', 'Close', 'Volume']]
                next_price = sp.predict_next_day(model, last_row)
                
                # --- Metrics Display ---
                st.markdown("## 🎯 **Prediction Results**")
                
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.metric("Predicted Close", f"${next_price:.2f}", delta=" Next Trading Day")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with m2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    last_close = data['Close'].iloc[-1]
                    # Handle Series if needed
                    try:
                        last_close_val = last_close.item()
                    except:
                        last_close_val = last_close
                        
                    change = next_price - last_close_val
                    pct_change = (change / last_close_val) * 100
                    st.metric("Current Price", f"${last_close_val:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with m3:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    from sklearn.metrics import mean_absolute_error
                    mae = mean_absolute_error(y_test, predictions)
                    st.metric("Model Error (MAE)", f"±${mae:.2f}", delta_color="inverse")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")

                # --- Interactive Charts ---
                st.markdown("### 📈 **Market Trends & Forecast**")
                
                fig = go.Figure()
                
                # Area Chart for History
                fig.add_trace(go.Scatter(
                    x=dates[-len(y_test):], 
                    y=y_test,
                    mode='lines',
                    name='Actual Market Price',
                    line=dict(color='#4b6cb7', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(75, 108, 183, 0.1)' 
                ))
                
                # Dashed Line for Prediction
                fig.add_trace(go.Scatter(
                    x=dates[-len(y_test):], 
                    y=predictions,
                    mode='lines',
                    name='AI Prediction',
                    line=dict(color='#ff6b6b', width=2, dash='dash')
                ))

                fig.update_layout(
                    height=500,
                    template="plotly_white",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(family="Helvetica Neue", size=12, color="#444")
                )
                
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
            status.update(label="❌ System Failure", state="error")

    else:
        # Default State / Welcome Message
        st.info("👈 **Select a stock** from the sidebar and click **'Generate Prediction'** to start using the AI model.")
        
        # Static Placeholder Chart for Aesthetics
        import numpy as np
        dates_dummy = pd.date_range("2023-01-01", periods=100)
        prices_dummy = np.cumsum(np.random.randn(100)) + 100
        
        fig = go.Figure(go.Scatter(x=dates_dummy, y=prices_dummy, mode='lines', line=dict(color='#ddd'), fill='tozeroy'))
        fig.update_layout(title="Market Preview", template="plotly_white", xaxis_visible=False, yaxis_visible=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
