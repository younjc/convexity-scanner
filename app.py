import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date

# --- Page Configuration ---
st.set_page_config(page_title="Convexity Screener", layout="wide")

# --- Helper Functions ---
def get_options_data(ticker_symbol):
    """
    Fetches real option chains for a ticker using yfinance.
    Returns a DataFrame of puts.
    """
    tk = yf.Ticker(ticker_symbol)
    
    # yfinance sometimes fails to get expirations; handle gracefully
    try:
        expirations = tk.options
    except Exception:
        return None, "Could not fetch expirations. Ticker might be invalid or data unavailable."
    
    if not expirations:
        return None, "No options data found."

    all_puts = []
    
    # Progress bar for fetching data (it can be slow)
    progress_text = "Fetching option chains..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, exp_date in enumerate(expirations):
        # Update progress
        progress_percent = int(((i + 1) / len(expirations)) * 100)
        my_bar.progress(progress_percent, text=f"Fetching {exp_date}...")
        
        try:
            # Get chain for this expiration
            chain = tk.option_chain(exp_date)
            puts = chain.puts
            
            # Add expiration column
            puts['expiration'] = exp_date
            
            # Basic cleaning
            puts['lastPrice'] = puts['lastPrice']
            puts['strike'] = puts['strike']
            
            # Calculate simple Greeks/Metrics if missing
            # Note: yfinance often lacks Delta. We will approximate or skip.
            # We will use 'percentChange' or volatility as proxies if needed, 
            # but for this specific "Universa" screener, we focus on moneyness/price.
            
            all_puts.append(puts)
        except Exception as e:
            continue
            
    my_bar.empty()
    
    if not all_puts:
        return None, "No put contracts found."
        
    df = pd.concat(all_puts, ignore_index=True)
    return df, None

def days_until(date_str):
    d = datetime.strptime(date_str, '%Y-%m-%d').date()
    return (d - date.today()).days

def calculate_metrics(df, underlying_price, crash_drop=0.35):
    """
    Adds screener metrics: DTE, OTM%, Premium%, Crash Multiple
    """
    # Calculate DTE
    df['dte'] = df['expiration'].apply(days_until)
    
    # Calculate OTM % (Moneyness)
    # For puts: (Spot - Strike) / Spot. 
    # Positive result = ITM (bad for us), Negative result = OTM.
    # We want Deep OTM, so we look for Strike < Spot.
    # Let's define OTM% as how far DROP is needed. 
    # OTM% = (Spot - Strike) / Spot.
    # Example: Spot 500, Strike 300. (500-300)/500 = 0.40 (40% OTM)
    df['otm_pct'] = (underlying_price - df['strike']) / underlying_price
    
    # Premium Fraction
    df['prem_frac'] = df['lastPrice'] / underlying_price
    
    # Crash Payoff Multiple
    # Scenario: Underlying drops by crash_drop (e.g. 35%)
    crash_price = underlying_price * (1 - crash_drop)
    
    # Intrinsic value at crash
    # If Strike 300, Crash Price 325 (500 * .65), Intrinsic = 0.
    # If Strike 400, Crash Price 325, Intrinsic = 75.
    df['crash_intrinsic'] = (df['strike'] - crash_price).clip(lower=0)
    
    # Multiple = Intrinsic / Current Cost
    # Avoid division by zero
    df['crash_multiple'] = np.where(
        df['lastPrice'] > 0.01, 
        df['crash_intrinsic'] / df['lastPrice'], 
        0
    )
    
    return df

# --- Main App ---
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

if run_btn:
    try:
        # 1. Get Spot Price
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if history.empty:
            st.error(f"Could not fetch data for {ticker}")
            st.stop()
        
        current_price = history['Close'].iloc[-1]
        st.info(f"Current Price of {ticker}: **${current_price:,.2f}**")
        
        # 2. Get Options
        raw_df, error = get_options_data(ticker)
        if error:
            st.error(error)
            st.stop()
            
        # 3. Calculate Metrics
        df = calculate_metrics(raw_df, current_price, crash_drop)
        
        # 4. Filter
        # OTM Filter: We want strike to be 20% to 50% below spot.
        # This means (Spot - Strike)/Spot should be between 0.20 and 0.50
        mask_otm = (df['otm_pct'] >= min_otm) & (df['otm_pct'] <= max_otm)
        mask_dte = (df['dte'] >= min_dte) & (df['dte'] <= max_dte)
        mask_prem = (df['prem_frac'] <= max_prem_pct)
        mask_mult = (df['crash_multiple'] >= min_multiple)
        
        filtered = df[mask_otm & mask_dte & mask_prem & mask_mult].copy()
        
        if filtered.empty:
            st.warning("No options found matching these strict criteria.")
        else:
            # Sort by Explosiveness
            filtered = filtered.sort_values(by='crash_multiple', ascending=False).head(50)
            
            st.success(f"Found {len(filtered)} candidates!")
            
            # Display readable table
            display_cols = [
                'contractSymbol', 'expiration', 'strike', 'lastPrice', 
                'dte', 'otm_pct', 'crash_multiple'
            ]
            
            # Formatting for display
            display_df = filtered[display_cols].copy()
            display_df['otm_pct'] = display_df['otm_pct'].apply(lambda x: f"{x:.1%}")
            display_df['crash_multiple'] = display_df['crash_multiple'].apply(lambda x: f"{x:.1f}x")
            display_df['lastPrice'] = display_df['lastPrice'].apply(lambda x: f"${x:.2f}")
            display_df['strike'] = display_df['strike'].apply(lambda x: f"${x:.0f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            st.caption("Note: 'Crash Multiple' assumes the option holds intrinsic value at expiration if the market drops instantly. It does not account for IV spikes (Vega), so actual returns would likely be even higher.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
