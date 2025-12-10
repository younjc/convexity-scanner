import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    tk = yf.Ticker(ticker_symbol)
    
    try:
        expirations = tk.options
    except Exception:
        return None, f"Could not fetch expirations for {ticker_symbol}.", None
    
    if not expirations:
        return None, f"No options data found for {ticker_symbol}.", None

    # 1. Filter Dates
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
        return None, f"No expirations found for {ticker_symbol} in range.", None

    # 2. APPLY USER LIMIT
    warning_msg = None
    if len(valid_dates) > max_dates_to_scan:
        indices = np.linspace(0, len(valid_dates) - 1, max_dates_to_scan, dtype=int)
        unique_indices = sorted(list(set(indices)))
        subset = [valid_dates[i] for i in unique_indices]
        warning_msg = f"Scanning subset for {ticker_symbol}"
        valid_dates = subset

    all_puts = []
    
    for i, exp_date in enumerate(valid_dates):
        # Slightly shorter sleep for batch mode
        time.sleep(random.uniform(0.5, 1.5))
        
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
        return None, f"No puts found for {ticker_symbol}.", warning_msg
        
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

st.title("🛡️ Tail-Risk Convexity Screener (Batch Mode)")
st.caption("Compare crash protection across multiple assets simultaneously.")

with st.sidebar:
    st.header("Settings")
    
    st.subheader("Target Assets")
    ticker_input = st.text_input(
        "Enter Tickers (comma-separated)", 
        value="SPY, QQQ, IWM",
        help="Example: SPY, QQQ, IWM, HYG, FXI"
    )
    
    st.subheader("Time Horizon & Speed")
    min_dte = st.number_input("Min DTE", value=30)
    max_dte = st.number_input("Max DTE", value=90)
    
    max_dates = st.slider(
        "Max Dates per Ticker", 
        min_value=1, 
        max_value=10, 
        value=3, 
        help="Kept low to allow faster batch scanning."
    )

    st.subheader("Liquidity Filters")
    min_vol = st.number_input("Min Volume", value=0)
    min_oi = st.number_input("Min Open Interest", value=0)

    st.subheader("Convexity Filters")
    min_otm = st.slider("Min OTM %", 0.0, 0.5, 0.10)
    max_prem_pct = st.number_input("Max Premium %", value=0.02, format="%.4f")
    
    st.subheader("Scenario")
    crash_drop = st.number_input("Crash Drop %", value=0.25)
    
    st.divider()
    show_all_results = st.checkbox("Show All Candidates", value=False)
    
    run_btn = st.button("Run Batch Scan", type="primary")

if run_btn:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    master_results = []
    errors = []
    
    progress_bar = st.progress(0, text="Starting Batch Scan...")
    
    for idx, ticker in enumerate(tickers):
        progress_bar.progress(int((idx / len(tickers)) * 100), text=f"Scanning {ticker}...")
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if hist.empty:
                errors.append(f"Could not fetch price for {ticker}")
                continue
            current_price = hist['Close'].iloc[-1]
            
            raw_df, err, warn = get_options_data(ticker, min_dte, max_dte, max_dates)
            
            if err:
                continue
            
            df, crash_price = calculate_metrics(raw_df, current_price, crash_drop)
            
            mask_otm = df['otm_pct'] >= min_otm
            mask_prem = df['prem_frac'] <= max_prem_pct
            mask_vol = df['volume'] >= min_vol
            mask_oi = df['openInterest'] >= min_oi
            
            filtered = df[mask_otm & mask_prem & mask_vol & mask_oi].copy()
            
            if not filtered.empty:
                filtered.insert(0, "Ticker", ticker)
                master_results.append(filtered)
                
        except Exception as e:
            errors.append(f"Error scanning {ticker}: {str(e)}")
            
    progress_bar.empty()

    if not master_results:
        st.warning("No options found matching your criteria for any of the entered tickers.")
        if errors:
            with st.expander("View Errors"):
                st.write(errors)
    else:
        final_df = pd.concat(master_results, ignore_index=True)
        final_df = final_df.sort_values('crash_multiple', ascending=False)
        
        if not show_all_results:
            display_data = final_df.head(20)
            st.success(f"✅ Found {len(final_df)} total candidates. Showing Top 20.")
        else:
            display_data = final_df
            st.success(f"✅ Found {len(final_df)} candidates across {len(tickers)} tickers.")

        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Batch Results (CSV)",
            data=csv,
            file_name="convexity_batch_scan.csv",
            mime='text/csv',
        )
        
        view = display_data[['Ticker', 'expiration', 'strike', 'lastPrice', 'volume', 'openInterest', 'crash_value', 'crash_multiple', 'otm_pct']].copy()
        view.columns = ['Ticker', 'Expiration', 'Strike', 'Cost Now', 'Vol', 'Open Int', 'Value in Crash', 'Multiplier (x)', 'OTM %']

        def bold_top_rows(x):
            return ['font-weight: bold' if i < 3 else '' for i in range(len(x))]

        styler = view.style\
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
        
        st.info(f"**Scenario:** If market drops **{crash_drop:.0%}**, these options explode. 'Multiplier' = Payoff / Cost.")

        # --- MULTI-COLOR CHART (FIXED LEGEND) ---
        st.divider()
        
        chart_data = final_df.copy()
        chart_data["dte"] = chart_data["dte"].astype(float)
        chart_data["expiration_str"] = chart_data["expiration"].astype(str)

        summary = (
            chart_data.groupby(["Ticker", "expiration_str", "dte"])
            .agg(
                avg_multiple=("crash_multiple", "mean"),
                count=("crash_multiple", "size"),
            )
            .reset_index()
        )

        if not summary.empty:
            st.markdown("### 📆 Convexity Comparison (Color by Ticker)")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            unique_tickers = summary['Ticker'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tickers)))
            color_map = dict(zip(unique_tickers, colors))
            
            for ticker in unique_tickers:
                subset = summary[summary['Ticker'] == ticker]
                ax.scatter(
                    subset["dte"],
                    subset["avg_multiple"],
                    s=subset["count"] * 50.0,
                    alpha=0.7,
                    label=ticker,
                    color=color_map[ticker],
                    edgecolors='black'
                )

            ax.set_xlabel("Days to Expiration")
            ax.set_ylabel("Avg Multiplier (x)")
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # --- THE FIX IS HERE ---
            # Moves legend outside the plot area
            ax.legend(title="Ticker", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            st.pyplot(fig)

        st.divider()
        st.caption("Disclaimer: Educational use only. Theoretical values. Not financial advice.")
