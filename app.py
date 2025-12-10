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

    # 2. LIMIT VOLUME (Safe Mode)
    if limit_requests and len(relevant_dates) > 3:
        first = relevant_dates[0]
        last = relevant_dates[-1]
        mid = relevant_dates[len(relevant_dates)//2]
        subset = sorted(list(set([first, mid, last])))
        
        st.toast(f"⚠️ Safe Mode: Scanning only 3 dates: {', '.join(subset)}", icon="🛡️")
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
    
    # Crash Scenario
    crash_price = underlying_price * (1 - crash_drop)
    df['crash_value'] = (df['strike'] - crash_price).clip(lower=0)
    
    # Avoid division by zero/pennies
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
        
        # 4. Filter
        mask_otm = df['otm_pct'] >= min_otm
        mask_prem = df['prem_frac'] <= max_prem_pct
        
        filtered = df[mask_otm & mask_prem].copy()
        
        if filtered.empty:
            st.info("No options found. Try lowering the 'Min OTM' or increasing 'Max Premium'.")
        else:
            filtered = filtered.sort_values('crash_multiple', ascending=False).head(20)
            
            # --- PREPARE DATA FOR DISPLAY ---
            # We keep the data numeric so the gradient works, then format it later
            display = filtered[['expiration', 'strike', 'lastPrice', 'crash_value', 'crash_multiple', 'otm_pct']].copy()
            
            # Rename for the user
            display.columns = ['Expiration', 'Strike', 'Cost Now', 'Value in Crash', 'Multiplier (x)', 'OTM %']

            # --- EXPLANATION BOX ---
            st.markdown(f"""
            ---
            ### 🔮 What you're seeing
            Each row is a **Put Option** on **{ticker}**.
            * **Cost Now:** What you pay today to buy 1 contract (x100).
            * **OTM %:** How far below the current price the strike is.
            * **Multiplier (x):** How many times your money this option might be worth if **{ticker}** instantly dropped **{crash_drop:.0%}** (to **${crash_price:,.2f}**).
            
            > *Note: This ignores bid/ask spreads and IV changes. It is a theoretical "intrinsic value" calculation.*
            """)

            # --- STYLING MAGIC ---
            # 1. Format the numbers (Currency, Percent, etc.)
            # 2. Color scale the 'Multiplier'
            # 3. Bold the top 5 rows
            
            def bold_top_rows(x):
                # Returns a list of CSS strings, one for each row.
                # If row index is < 5, return 'font-weight: bold'
                return ['font-weight: bold' if i < 5 else '' for i in range(len(x))]

            styler = display.style\
                .format({
                    'Strike': '${:,.0f}',
                    'Cost Now': '${:.2f}',
                    'Value in Crash': '${:.2f}',
                    'Multiplier (x)': '{:.1f}x',
                    'OTM %': '{:.1%}'
                })\
                .background_gradient(subset=['Multiplier (x)'], cmap='Greens')\
                .apply(bold_top_rows, axis=0) # Apply bold to top 5 rows

            st.dataframe(styler, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
