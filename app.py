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
    """Calculates days until a specific date string (YYYY-MM-DD)."""
    d = datetime.strptime(date_str, '%Y-%m-%d').date()
    return (d - date.today()).days

@st.cache_data(ttl=3600, show_spinner=False)
def get_options_data(ticker_symbol, min_dte_days, max_dte_days):
    """
    Fetches real option chains using yfinance with rate-limit handling.
    
    1. Smart Filtering: Only fetches dates within the requested DTE range.
    2. Caching: Saves results for 1 hour to prevent re-fetching on every slider move.
    3. Throttling: Sleeps randomly between requests to avoid 429 errors.
    """
    tk = yf.Ticker(ticker_symbol)
    
    try:
        # Fetch all available expiration dates first
        expirations = tk.options
    except Exception:
        return None, "Could not fetch expirations. Rate limited or invalid ticker."
    
    if not expirations:
        return None, "No options data found."

    # 1. PRE-FILTER DATES: Only request data for dates we actually want
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
        return None, f"No expirations found between {min_dte_days} and {max_dte_days} days. Try widening your DTE range."

    all_puts = []
    
    # Progress bar setup
    progress_text = f"Fetching {len(relevant_dates)} expiration dates (throttled)..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, exp_date in enumerate(relevant_dates):
        # Update progress
        my_bar.progress(int(((i + 1) / len(relevant_dates)) * 100), text=f"Fetching {exp_date}...")
        
        # 2. THROTTLING: Sleep 0.5 - 1.5 seconds between calls
        time.sleep(random.uniform(0.5, 1.5))
        
        try:
            # Get chain for this expiration
            chain = tk.option_chain(exp_date)
            puts = chain.puts
            
            # Add expiration column
            puts['expiration'] = exp_date
            
            # Ensure critical columns exist
            if 'lastPrice' not in puts.columns or 'strike' not in puts.columns:
                continue
                
            all_puts.append(puts)
        except Exception as e:
            # If one date fails, skip it and keep going
            continue
            
    my_bar.empty()
    
    if not all_puts:
        return None, "No put contracts found."
        
    df = pd.concat(all_puts, ignore_index=True)
    return df, None

def calculate_metrics(df, underlying_price, crash_drop=0.35):
    """
    Adds screener metrics: DTE, OTM%, Premium%, Crash Multiple
    """
    # Calculate DTE
    df['dte'] = df['expiration'].apply(days_until)
    
    # Calculate OTM % (Moneyness)
    # We want Deep OTM Puts (Strike < Spot). 
    # OTM% = (Spot - Strike) / Spot.
    df['otm_pct'] = (underlying_price - df['strike']) / underlying_price
    
    # Premium Fraction
    df['prem_frac'] = df['lastPrice'] / underlying_price
    
    # Crash Payoff Multiple
    # Scenario: Underlying drops by crash_drop (e.g. 35%)
    crash_price = underlying_price * (1 - crash_drop)
    
    # Intrinsic value at crash
    df['crash_intrinsic'] = (df['strike'] - crash_price).clip(lower=0)
    
    # Multiple = Intrinsic / Current Cost
    # Avoid division by zero
    df['crash_multiple'] = np.where(
        df['lastPrice'] > 0.01, 
        df['crash_intrinsic'] / df['lastPrice'], 
        0
    )
    
    return df

# --- Main App Interface ---

st.title("🛡️ Tail-Risk Convexity Screener")
st.markdown("""
This tool scans **real option chains** (via Yahoo Finance) to find "lottery ticket" Puts:
1. **Deep OTM** (20-50% below spot)
2. **Cheap** (Low premium vs spot)
3. **Explosive Payoff** if the market crashes (e.g. 35% drop)
""")

# Sidebar inputs
with st.sidebar:
    st.header("Search Parameters")
    ticker = st.text_input("Ticker Symbol", value="SPY").upper()
    
    st.subheader("Filter Criteria")
    min_dte = st.number_input("Min Days to Expiration", value=30)
    max_dte = st.number_input("Max Days to Expiration", value=120)
    
    min_otm = st.slider("Min OTM % (Distance from Spot)", 0.10, 0.60, 0.20)
    max_otm = st.slider("Max OTM %", 0.10, 0.60, 0.50)
    
    max_prem_pct = st.number_input("Max Premium (% of Spot)", value=0.01, format="%.4f")
    
    st.subheader("Scenario")
    crash_drop = st.number_input("Hypothetical Crash Drop (%)", value=0.35)
    min_multiple = st.number_input("Min Payoff Multiple (x)", value=20)

    run_btn = st.button("Run Scanner", type="primary")
    
    st.info("Tip: If you get 'Rate Limited', try waiting a minute or narrowing your DTE range.")

if run_btn:
    try:
        # 1. Get Spot Price
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if history.empty:
            st.error(f"Could not fetch data for {ticker}")
            st.stop()
        
        current_price = history['Close'].iloc[-1]
        st.success(f"**{ticker}** Price: **${current_price:,.2f}**")
        
        # 2. Get Options (with smart filtering & caching)
        raw_df, error = get_options_data(ticker, min_dte, max_dte)
        
        if error:
            st.error(error)
            st.stop()
            
        # 3. Calculate Metrics
        df = calculate_metrics(raw_df, current_price, crash_drop)
        
        # 4. Filter Results
        mask_otm = (df['otm_pct'] >= min_otm) & (df['otm_pct'] <= max_otm)
        mask_prem = (df['prem_frac'] <= max_prem_pct)
        mask_mult = (df['crash_multiple'] >= min_multiple)
        
        # Note: DTE is already filtered in the fetch function, but good to be safe
        mask_dte = (df['dte'] >= min_dte) & (df['dte'] <= max_dte)
        
        filtered = df[mask_otm & mask_prem & mask_mult & mask_dte].copy()
        
        if filtered.empty:
            st.warning("No options found matching these strict criteria.")
        else:
            # Sort by Explosiveness (Crash Multiple)
            filtered = filtered.sort_values(by='crash_multiple', ascending=False).head(50)
            
            st.success(f"Found {len(filtered)} candidates!")
            
            # Prepare Display Table
            display_cols = [
                'contractSymbol', 'expiration', 'strike', 'lastPrice', 
                'dte', 'otm_pct', 'crash_multiple'
            ]
            
            display_df = filtered[display_cols].copy()
            
            # Formatting
            display_df['otm_pct'] = display_df['otm_pct'].apply(lambda x: f"{x:.1%}")
            display_df['crash_multiple'] = display_df['crash_multiple'].apply(lambda x: f"{x:.1f}x")
            display_df['lastPrice'] = display_df['lastPrice'].apply(lambda x: f"${x:.2f}")
            display_df['strike'] = display_df['strike'].apply(lambda x: f"${x:.0f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            st.caption(f"Note: 'Crash Multiple' = (Intrinsic Value if {ticker} drops {crash_drop:.0%}) / Current Price.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
