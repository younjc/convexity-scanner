import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
import time
import random

# --- Page Configuration ---
st.set_page_config(page_title="Convexity Screener", layout="wide")

# --- Helper Functions ---

def days_until(date_str):
    d = datetime.strptime(date_str, '%Y-%m-%d').date()
    return (d - date.today()).days

@st.cache_data(ttl=3600, show_spinner=False)
def get_options_data(ticker_symbol, min_dte_days, max_dte_days, limit_requests=True):
    # Let yfinance handle the session internally to avoid "curl_cffi" errors
    tk = yf.Ticker(ticker_symbol)
    
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

    # 2. LIMIT VOLUME
    if limit_requests and len(relevant_dates) > 3:
        first = relevant_dates[0]
        last = relevant_dates[-1]
        mid = relevant_dates[len(relevant_dates)//2]
        subset = sorted(list(set([first, mid, last])))
        
        st.toast(f"⚠️ Safe Mode: Scanning only 3 dates to avoid blocks: {', '.join(subset)}", icon="🛡️")
        relevant_dates = subset

    all_puts = []
    progress_text = f"Scanning {len(relevant_dates)} dates..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, exp_date in enumerate(relevant_dates):
        my_bar.progress(int(((i + 1) / len(relevant_dates)) * 100), text=f"Fetching {exp_date}...")
        time.sleep(random.uniform(1.0, 2.0))
        
        try:
            chain = tk.option_chain(exp_date)
            puts = chain.puts
            puts['expiration'] = exp_date
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
    
    # --- NEW: Explicit Crash Values ---
    crash_price = underlying_price * (1 - crash_drop)
    
    # The value of the put if market hits crash_price: max(Strike - CrashPrice, 0)
    df['crash_value'] = (df['strike'] - crash_price).clip(lower=0)
    
    # The Multiple: Crash Value / Current Cost
    df['crash_multiple'] = np.where(
        df['lastPrice'] > 0.01, 
        df['crash_value'] / df['lastPrice'], 
        0
    )
    return df, crash_price

# --- Main UI ---

st.title("🛡️ Tail-Risk Convexity Screener")
st.caption("Find cheap options that explode in value during a market crash.")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="SPY").upper()
    
    st.subheader("Time Horizon")
    min_dte = st.number_input("Min DTE", value=30)
    max_dte = st.number_input("Max DTE", value=90)
    safe_mode = st.checkbox("Safe Mode (Limit to 3 dates)", value=True)

    st.subheader("Filters")
    min_otm = st.slider("Min OTM %", 0.1, 0.5, 0.15)
    max_prem_pct = st.number_input("Max Premium %", value=0.01, format="%.4f")
    
    st.subheader("Scenario")
    crash_drop = st.number_input("Crash Drop %", value=0.25)
    
    run_btn = st.button("Run Scanner", type="primary")

if run_btn:
    try:
        # 1. Get Spot Price
        stock = yf.Ticker(ticker)
        fast_info = stock.fast_info
        current_price = fast_info.last_price if fast_info.last_price else stock.history(period='1d')['Close'].iloc[-1]
        
        st.success(f"**{ticker}** Spot: **${current_price:,.2f}**")
        
        # 2. Get Data
        raw_df, error = get_options_data(ticker, min_dte, max_dte, limit_requests=safe_mode)
        
        if error:
            st.warning(error)
            st.stop()
            
        # 3. Calculate
        df, crash_price = calculate_metrics(raw_df, current_price, crash_drop)
        
        st.info(f"📉 **Scenario:** If {ticker} drops **{crash_drop:.0%}** (to **${crash_price:,.2f}**)...")

        # 4. Filter
        mask_otm = df['otm_pct'] >= min_otm
        mask_prem = df['prem_frac'] <= max_prem_pct
        
        filtered = df[mask_otm & mask_prem].copy()
        
        if filtered.empty:
            st.info("No options found. Try lowering the 'Min OTM' or increasing 'Max Premium'.")
        else:
            filtered = filtered.sort_values('crash_multiple', ascending=False).head(20)
            
            # --- NEW: Clean "Story" Table ---
            display = pd.DataFrame()
            display['Expiration'] = filtered['expiration']
            display['Strike'] = filtered['strike']
            display['Cost Now'] = filtered['lastPrice']
            display['Value in Crash'] = filtered['crash_value']
            display['Multiplier (x)'] = filtered['crash_multiple']
            display['OTM %'] = filtered['otm_pct']

            # Formatting
            display['OTM %'] = display['OTM %'].apply(lambda x: f"{x:.1%}")
            display['Multiplier (x)'] = display['Multiplier (x)'].apply(lambda x: f"{x:.1f}x")
            display['Cost Now'] = display['Cost Now'].apply(lambda x: f"${x:.2f}")
            display['Value in Crash'] = display['Value in Crash'].apply(lambda x: f"${x:.2f}")
            display['Strike'] = display['Strike'].apply(lambda x: f"${x:.0f}")
            
            st.dataframe(display, use_container_width=True)
            st.success(f"Top {len(filtered)} candidates shown above.")
            st.markdown(
                f"> **Intuition Example:** For the top row, you pay **{display.iloc[0]['Cost Now']}** today. "
                f"If the market crashes to **${crash_price:,.2f}**, that option becomes worth **{display.iloc[0]['Value in Crash']}**. "
                f"That is a **{display.iloc[0]['Multiplier (x)']}** return."
            )

    except Exception as e:
        st.error(f"Error: {e}")
