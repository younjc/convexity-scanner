import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
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
def get_options_data(ticker_symbol, min_dte_days, max_dte_days, max_dates_to_scan):
    """
    Fetches data with a user-defined limit on how many dates to scan.
    """
    tk = yf.Ticker(ticker_symbol)
    
    try:
        expirations = tk.options
    except Exception:
        return None, "Could not fetch expirations. Yahoo might be blocking this IP temporarily.", None
    
    if not expirations:
        return None, "No options data found.", None

    # 1. Filter Dates (Get all valid dates first)
    valid_dates = []
    today = date.today()
    for exp_str in expirations:
        try:
            d = datetime.strptime(exp_str, '%Y-%m-%d').date()
            dte = (d - today).days
            if min_dte_days <= dte <= max_dte_days:
                valid_dates.append(exp_str)
        except ValueError:
            continue

    if not valid_dates:
        return None, f"No expirations found between {min_dte_days}-{max_dte_days} days.", None

    # 2. APPLY USER LIMIT (The Slider)
    warning_msg = None
    if len(valid_dates) > max_dates_to_scan:
        # If we have 20 dates but user only wants 5, pick 5 evenly spaced ones
        indices = np.linspace(0, len(valid_dates) - 1, max_dates_to_scan, dtype=int)
        # Use a set to avoid duplicates if range is small
        unique_indices = sorted(list(set(indices)))
        
        subset = [valid_dates[i] for i in unique_indices]
        
        warning_msg = f"⚠️ Limit Active: Scanning {len(subset)} dates (out of {len(valid_dates)} available) to prevent blocking."
        valid_dates = subset

    all_puts = []
    
    for i, exp_date in enumerate(valid_dates):
        # Random sleep to look like a human
        time.sleep(random.uniform(1.0, 2.0))
        
        try:
            chain = tk.option_chain(exp_date)
            puts = chain.puts
            puts['expiration'] = exp_date
            
            if 'lastPrice' in puts.columns and 'strike' in puts.columns:
                puts['volume'] = puts.get('volume', pd.Series([0]*len(puts))).fillna(0)
                puts['openInterest'] = puts.get('openInterest', pd.Series([0]*len(puts))).fillna(0)
                all_puts.append(puts)
        except Exception:
            continue
            
    if not all_puts:
        return None, "No put contracts found (or connection blocked).", warning_msg
        
    df = pd.concat(all_puts, ignore_index=True)
    return df, None, warning_msg

def calculate_metrics(df, underlying_price, crash_drop=0.35):
    df['dte'] = df['expiration'].apply(days_until)
    df['otm_pct'] = (underlying_price - df['strike']) / underlying_price
    df['prem_frac'] = df['lastPrice'] / underlying_price
    
    # Crash Scenario
    crash_price = underlying_price * (1 - crash_drop)
    df['crash_value'] = (df['strike'] - crash_price).clip(lower=0)
    
    # Avoid division by zero
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
    
    st.subheader("Time Horizon & Speed")
    min_dte = st.number_input("Min DTE", value=30)
    max_dte = st.number_input("Max DTE", value=90)
    
    # NEW SLIDER: Replaces the 'Safe Mode' checkbox
    max_dates = st.slider(
        "Max Dates to Scan", 
        min_value=1, 
        max_value=20, 
        value=3, 
        help="Higher = More data, but higher risk of 'Rate Limited' error."
    )

    st.subheader("Liquidity Filters")
    min_vol = st.number_input("Min Volume (Today)", value=0, step=1)
    min_oi = st.number_input("Min Open Interest", value=0, step=50)

    st.subheader("Convexity Filters")
    min_otm = st.slider("Min OTM %", 0.0, 0.5, 0.10)
    max_prem_pct = st.number_input("Max Premium %", value=0.02, format="%.4f")
    
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
        with st.spinner(f"Scanning up to {max_dates} expiration dates..."):
            raw_df, error, warning = get_options_data(ticker, min_dte, max_dte, max_dates)
        
        if warning:
            st.toast(warning, icon="🛡️")
            
        if error:
            st.warning(error)
            st.stop()
            
        # 3. Calculate
        df, crash_price = calculate_metrics(raw_df, current_price, crash_drop)
        
        # 4. Filter
        mask_otm = df['otm_pct'] >= min_otm
        mask_prem = df['prem_frac'] <= max_prem_pct
        mask_vol = df['volume'] >= min_vol
        mask_oi = df['openInterest'] >= min_oi
        
        filtered = df[mask_otm & mask_prem & mask_vol & mask_oi].copy()
        
        if filtered.empty:
            st.info(f"No options found. Try lowering 'Min Volume' or 'Min OTM'. (Scanned {len(df)} contracts).")
        else:
            filtered = filtered.sort_values('crash_multiple', ascending=False).head(20)
            
            # --- PREPARE DATA FOR DISPLAY ---
            display = filtered[['expiration', 'strike', 'lastPrice', 'volume', 'openInterest', 'crash_value', 'crash_multiple', 'otm_pct']].copy()
            display.columns = ['Expiration', 'Strike', 'Cost Now', 'Vol', 'Open Int', 'Value in Crash', 'Multiplier (x)', 'OTM %']

            # --- EXPLANATION BOXES ---
            st.markdown(f"""
            ---
            ### 🔮 What you're seeing
            Each row is a **Put Option** on **{ticker}**.
            * **Cost Now:** What you pay today.
            * **Multiplier (x):** Theoretical return if **{ticker}** drops **{crash_drop:.0%}** (to **${crash_price:,.2f}**).
            """)
            
            st.info("""
            **What this means (Convexity in 3 sentences):**
            1. **Convexity measures** how much an option’s value can accelerate during sharp market moves.
            2. **Far out-of-the-money puts** often behave like “insurance with leverage,” costing very little but responding explosively in a crash.
            3. **Understanding convexity** helps you see which options offer the most asymmetry—small, controlled cost today for disproportionately large protection in rare events.
            """)

            # --- TABLE ---
            def bold_top_rows(x):
                return ['font-weight: bold' if i < 5 else '' for i in range(len(x))]

            styler = display.style\
                .format({
                    'Strike': '${:,.0f}',
                    'Cost Now': '${:.2f}',
                    'Value in Crash': '${:.2f}',
                    'Multiplier (x)': '{:.1f}x',
                    'OTM %': '{:.1%}',
                    'Vol': '{:,.0f}',
                    'Open Int': '{:,.0f}'
                })\
                .background_gradient(subset=['Multiplier (x)'], cmap='Greens')\
                .apply(bold_top_rows, axis=0)

            st.dataframe(styler, use_container_width=True)

            # --- CHART: Convexity Bubble View ---
            st.divider()
            
            filtered["dte"] = filtered["dte"].astype(float)
            filtered["expiration_str"] = filtered["expiration"].astype(str)

            summary = (
                filtered.groupby(["expiration_str", "dte"])
                .agg(
                    avg_multiple=("crash_multiple", "mean"),
                    max_multiple=("crash_multiple", "max"),
                    count=("crash_multiple", "size"),
                )
                .reset_index()
            )

            if not summary.empty:
                st.markdown("### 📆 Convexity by Expiration (Bubble View)")
                st.caption(
                    "Vertical position = Average Crash Multiplier. Bubble size = Number of cheap options found."
                )

                fig, ax = plt.subplots(figsize=(8, 4))
                
                scatter = ax.scatter(
                    summary["dte"],
                    summary["avg_multiple"],
                    s=summary["count"] * 50.0,
                    alpha=0.6,
                    c='#22c55e',
                    edgecolors='black'
                )

                ax.set_xlabel("Days to Expiration")
                ax.set_ylabel("Avg Multiplier (x)")
                ax.grid(True, linestyle='--', alpha=0.3)

                for _, row in summary.iterrows():
                    ax.text(
                        row["dte"],
                        row["avg_multiple"],
                        row["expiration_str"],
                        fontsize=8,
                        ha="center",
                        va="bottom"
                    )

                st.pyplot(fig)

            # --- DISCLAIMER ---
            st.divider()
            st.caption("""
            **Simple Disclaimer:** This tool is for educational and research purposes only. It does not provide financial advice, trading signals, or predictions.  
            Values shown (including crash scenarios and multipliers) are theoretical, simplified, and ignore real-world factors like volatility changes, liquidity, spreads, slippage, and execution risk.  
            Use this to understand convexity, not to make trading decisions.
            """)

    except Exception as e:
        st.error(f"Error: {e}")
