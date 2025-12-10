import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, date
import time
import random

# --- Page Configuration ---
st.set_page_config(page_title="Convexity Screener", layout="wide")

# --- Helper Functions ---

def days_until(date_str):
    d = datetime.strptime(date_str, '%Y-%m-%d').date()
    return (d - date.today()).days

def get_session():
    """
    Creates a custom session with a browser-like User-Agent.
    This helps trick Yahoo into thinking we are a real user, not a script.
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

@st.cache_data(ttl=3600, show_spinner=False)
def get_options_data(ticker_symbol, min_dte_days, max_dte_days, limit_requests=True):
    session = get_session()
    tk = yf.Ticker(ticker_symbol, session=session)
    
    try:
        expirations = tk.options
    except Exception:
        return None, "Could not fetch expirations. Yahoo might be blocking this IP temporarily."
    
    if not expirations:
        return None, "No options data found."

    # 1. Filter Dates
    relevant_dates = []
    today = date.today()
    for exp_str in expirations:
        try:
            d = datetime.strptime(exp_str, '%Y-%m-%d').date()
            dte = (d - today).days
            if min_dte_days <= dte <= max_dte_days:
                relevant_dates.append(exp_str)
        except ValueError:
            continue

    if not relevant_dates:
        return None, f"No expirations found between {min_dte_days}-{max_dte_days} days."

    # 2. LIMIT VOLUME (Crucial Fix)
    # If we try to fetch 20 dates, Yahoo bans us.
    # We will pick at most 3 evenly spaced dates to keep traffic low.
    if limit_requests and len(relevant_dates) > 3:
        # Pick first, middle, and last
        first = relevant_dates[0]
        last = relevant_dates[-1]
        mid = relevant_dates[len(relevant_dates)//2]
        # remove duplicates if list is short
        subset = sorted(list(set([first, mid, last])))
        
        st.toast(f"⚠️ To avoid rate limits, scanning only 3 dates: {', '.join(subset)}", icon="🛡️")
        relevant_dates = subset

    all_puts = []
    progress_text = f"Scanning {len(relevant_dates)} dates..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, exp_date in enumerate(relevant_dates):
        my_bar.progress(int(((i + 1) / len(relevant_dates)) * 100), text=f"Fetching {exp_date}...")
        
        # Sleep to be polite
        time.sleep(random.uniform(1.0, 2.0))
        
        try:
            # Fetch chain
            chain = tk.option_chain(exp_date)
            puts = chain.puts
            puts['expiration'] = exp_date
            
            # Clean columns
            if 'lastPrice' in puts.columns and 'strike' in puts.columns:
                all_puts.append(puts)
        except Exception:
            continue
            
    my_bar.empty()
    
    if not all_puts:
        return None, "No put contracts found (or connection blocked)."
        
    df = pd.concat(all_puts, ignore_index=True)
    return df, None

def calculate_metrics(df, underlying_price, crash_drop=0.35):
    df['dte'] = df['expiration'].apply(days_until)
    df['otm_pct'] = (underlying_price - df['strike']) / underlying_price
    df['prem_frac'] = df['lastPrice'] / underlying_price
    
    crash_price = underlying_price * (1 - crash_drop)
    df['crash_intrinsic'] = (df['strike'] - crash_price).clip(lower=0)
    
    df['crash_multiple'] = np.where(
        df['lastPrice'] > 0.01, 
        df['crash_intrinsic'] / df['lastPrice'], 
        0
    )
    return df

# --- Main UI ---

st.title("🛡️ Tail-Risk Convexity Screener (Lite)")
st.caption("Optimized to avoid Yahoo Finance rate limits")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="SPY").upper()
    
    st.subheader("Time Horizon")
    min_dte = st.number_input("Min DTE", value=30)
    max_dte = st.number_input("Max DTE", value=90)
    
    # Checkbox to force 'Safe Mode'
    safe_mode = st.checkbox("Safe Mode (Limit to 3 dates)", value=True, help="Uncheck this to scan ALL dates, but you risk getting blocked.")

    st.subheader("Filters")
    min_otm = st.slider("Min OTM %", 0.1, 0.5, 0.2)
    max_prem_pct = st.number_input("Max Premium %", value=0.01, format="%.4f")
    
    st.subheader("Scenario")
    crash_drop = st.number_input("Crash Drop %", value=0.30)
    
    run_btn = st.button("Run Scanner", type="primary")

if run_btn:
    try:
        # 1. Get Spot Price
        session = get_session()
        stock = yf.Ticker(ticker, session=session)
        fast_info = stock.fast_info
        
        # Fallback if fast_info fails
        current_price = fast_info.last_price if fast_info.last_price else stock.history(period='1d')['Close'].iloc[-1]
        
        st.success(f"**{ticker}** Spot: **${current_price:,.2f}**")
        
        # 2. Get Data
        raw_df, error = get_options_data(ticker, min_dte, max_dte, limit_requests=safe_mode)
        
        if error:
            st.warning(error)
            st.stop()
            
        # 3. Calculate
        df = calculate_metrics(raw_df, current_price, crash_drop)
        
        # 4. Filter
        mask_otm = df['otm_pct'] >= min_otm
        mask_prem = df['prem_frac'] <= max_prem_pct
        
        filtered = df[mask_otm & mask_prem].copy()
        
        if filtered.empty:
            st.info("No options found. Try lowering the 'Min OTM' or increasing 'Max Premium'.")
        else:
            filtered = filtered.sort_values('crash_multiple', ascending=False).head(20)
            
            # Simple Display
            display = filtered[['expiration', 'strike', 'lastPrice', 'otm_pct', 'crash_multiple']].copy()
            display['otm_pct'] = display['otm_pct'].apply(lambda x: f"{x:.1%}")
            display['crash_multiple'] = display['crash_multiple'].apply(lambda x: f"{x:.1f}x")
            display['lastPrice'] = display['lastPrice'].apply(lambda x: f"${x:.2f}")
            
            st.dataframe(display, use_container_width=True)
            st.success(f"Top {len(filtered)} candidates shown above.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.markdown("**Troubleshooting:** If you see 'Rate Limited', wait 2 minutes and try again.")
