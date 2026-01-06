import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from datetime import datetime, date
import time
import random
import sqlite3
import json

# --- Page Configuration ---
st.set_page_config(page_title="Convexity Screener", layout="wide")

# ==========================================
# DATABASE MANAGER (SQLite)
# ==========================================

DB_FILE = "convexity.db"

def get_connection():
    # check_same_thread=False is needed for Streamlit's threading model
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # 1. Scan Runs Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS scan_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts_utc TEXT,
            tickers TEXT,
            settings_json TEXT
        )
    """)
    
    # 2. Option Contracts (Static info)
    c.execute("""
        CREATE TABLE IF NOT EXISTS option_contracts (
            contract_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            expiration TEXT,
            strike REAL,
            option_type TEXT,
            UNIQUE(ticker, expiration, strike, option_type)
        )
    """)
    
    # 3. Snapshots (Time-series data)
    c.execute("""
        CREATE TABLE IF NOT EXISTS option_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            contract_id INTEGER,
            snapshot_ts_utc TEXT,
            underlying_price REAL,
            last_price REAL,
            volume INTEGER,
            open_interest INTEGER,
            dte INTEGER,
            otm_pct REAL,
            prem_frac REAL,
            crash_value REAL,
            crash_multiple REAL,
            UNIQUE(run_id, contract_id),
            FOREIGN KEY(run_id) REFERENCES scan_runs(run_id),
            FOREIGN KEY(contract_id) REFERENCES option_contracts(contract_id)
        )
    """)
    conn.commit()
    conn.close()

def log_scan_run(tickers_list, settings_dict):
    conn = get_connection()
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    tickers_str = ",".join(tickers_list)
    settings_json = json.dumps(settings_dict)
    
    c.execute("INSERT INTO scan_runs (run_ts_utc, tickers, settings_json) VALUES (?, ?, ?)",
              (ts, tickers_str, settings_json))
    run_id = c.lastrowid
    conn.commit()
    conn.close()
    return run_id, ts

def save_batch_results(run_id, run_ts, df_results):
    if df_results.empty:
        return

    conn = get_connection()
    c = conn.cursor()

    # 1. Upsert Contracts (Insert OR IGNORE)
    contracts_data = []
    for _, row in df_results.iterrows():
        contracts_data.append((row['Ticker'], row['expiration'], row['strike'], 'P'))
    
    c.executemany("""
        INSERT OR IGNORE INTO option_contracts (ticker, expiration, strike, option_type)
        VALUES (?, ?, ?, ?)
    """, contracts_data)
    
    # 2. Resolve Contract IDs
    tickers = df_results['Ticker'].unique()
    placeholders = ','.join(['?']*len(tickers))
    query = f"SELECT contract_id, ticker, expiration, strike FROM option_contracts WHERE ticker IN ({placeholders}) AND option_type='P'"
    c.execute(query, tuple(tickers))
    
    id_map = {}
    for row in c.fetchall():
        key = (row[1], row[2], row[3])
        id_map[key] = row[0]

    # 3. Prepare Snapshots
    snapshots_data = []
    for _, row in df_results.iterrows():
        key = (row['Ticker'], row['expiration'], row['strike'])
        cid = id_map.get(key)
        
        if cid:
            snapshots_data.append((
                run_id,
                cid,
                run_ts,
                row.get('underlying_price', 0),
                row['lastPrice'],
                row['volume'],
                row['openInterest'],
                row['dte'],
                row['otm_pct'],
                row['prem_frac'],
                row['crash_value'],
                row['crash_multiple']
            ))
            
    c.executemany("""
        INSERT OR IGNORE INTO option_snapshots 
        (run_id, contract_id, snapshot_ts_utc, underlying_price, last_price, volume, open_interest, dte, otm_pct, prem_frac, crash_value, crash_multiple)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, snapshots_data)
    
    conn.commit()
    conn.close()

def get_history_runs():
    conn = get_connection()
    df = pd.read_sql("SELECT run_id, run_ts_utc, tickers FROM scan_runs ORDER BY run_id DESC", conn)
    conn.close()
    return df

def get_repricing_data(run_id_new, run_id_old):
    conn = get_connection()
    query = """
        SELECT 
            c.ticker, c.expiration, c.strike,
            s1.last_price as price_new, s1.crash_multiple as mult_new, s1.volume as vol_new, s1.otm_pct as otm_new,
            s2.last_price as price_old, s2.crash_multiple as mult_old, s2.volume as vol_old,
            s1.underlying_price as und_new
        FROM option_snapshots s1
        JOIN option_snapshots s2 ON s1.contract_id = s2.contract_id
        JOIN option_contracts c ON s1.contract_id = c.contract_id
        WHERE s1.run_id = ? AND s2.run_id = ?
    """
    df = pd.read_sql(query, conn, params=(run_id_new, run_id_old))
    conn.close()
    return df

def get_contract_history(ticker, expiration, strike):
    """
    Updated to fetch Open Interest for Time Series Analysis
    """
    conn = get_connection()
    query = """
        SELECT 
            s.snapshot_ts_utc, 
            s.last_price, 
            s.volume, 
            s.open_interest,
            s.crash_multiple, 
            s.underlying_price
        FROM option_snapshots s
        JOIN option_contracts c ON s.contract_id = c.contract_id
        WHERE c.ticker = ? AND c.expiration = ? AND c.strike = ?
        ORDER BY s.snapshot_ts_utc ASC
    """
    df = pd.read_sql(query, conn, params=(ticker, expiration, strike))
    conn.close()
    return df

def get_available_contracts():
    conn = get_connection()
    query = "SELECT DISTINCT ticker, expiration, strike FROM option_contracts ORDER BY ticker, expiration, strike"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Initialize DB on import/startup
try:
    init_db()
except Exception as e:
    st.error(f"Database initialization failed: {e}")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def days_until(date_str):
    try:
        d = datetime.strptime(date_str, '%Y-%m-%d').date()
        return (d - date.today()).days
    except ValueError:
        return 0

@st.cache_data(ttl=3600, show_spinner=False)
def get_options_data(ticker_symbol, min_dte_days, max_dte_days, max_dates_to_scan):
    tk = yf.Ticker(ticker_symbol)
    
    try:
        expirations = tk.options
    except Exception:
        return None, f"Could not fetch expirations for {ticker_symbol}.", None
    
    if not expirations:
        return None, f"No options data found for {ticker_symbol}.", None

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

    warning_msg = None
    if len(valid_dates) > max_dates_to_scan:
        indices = np.linspace(0, len(valid_dates) - 1, max_dates_to_scan, dtype=int)
        unique_indices = sorted(list(set(indices)))
        subset = [valid_dates[i] for i in unique_indices]
        warning_msg = f"Scanning subset for {ticker_symbol}"
        valid_dates = subset

    all_puts = []
    for i, exp_date in enumerate(valid_dates):
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
    
    crash_price = underlying_price * (1 - crash_drop)
    df['crash_value'] = (df['strike'] - crash_price).clip(lower=0)
    
    df['crash_multiple'] = np.where(
        df['lastPrice'] > 0.01, 
        df['crash_value'] / df['lastPrice'], 
        0
    )
    return df, crash_price

def analyze_microstructure(df):
    """
    Performs time-series analysis on option snapshots to determine positioning behavior.
    """
    df = df.copy()
    df['snapshot_ts_utc'] = pd.to_datetime(df['snapshot_ts_utc'])
    df = df.sort_values('snapshot_ts_utc')
    
    # Calculate Deltas
    df['delta_oi'] = df['open_interest'].diff().fillna(0)
    df['delta_price'] = df['last_price'].diff().fillna(0)
    
    # Classify Daily Action
    def classify_day(row):
        if row['delta_oi'] > 0:
            return "Accumulation"
        elif row['delta_oi'] < 0:
            return "Distribution"
        else:
            return "Flat"
            
    df['action'] = df.apply(classify_day, axis=1)
    
    # Churn Detection: Volume > 5x abs(Delta OI) and Volume > 50
    df['is_churn'] = (df['volume'] > 50) & (df['volume'] > 5 * df['delta_oi'].abs())
    
    return df

def generate_verdict(df):
    """
    Generates a text verdict based on the trend of Open Interest and Volume.
    """
    if len(df) < 2:
        return "Insufficient Data", "Need at least 2 data points to analyze trend."
        
    start_oi = df['open_interest'].iloc[0]
    end_oi = df['open_interest'].iloc[-1]
    net_oi_change = end_oi - start_oi
    
    # Determine Trend
    if net_oi_change > (start_oi * 0.10) and net_oi_change > 50:
        trend = "Accreting Hedge"
        desc = "Consistent increase in Open Interest suggests net new positioning (risk adding)."
    elif net_oi_change < -(start_oi * 0.10):
        trend = "Active Unwind"
        desc = "Significant decrease in Open Interest suggests closing of positions."
    else:
        # Check for Churn vs Static
        churn_days = df['is_churn'].sum()
        if churn_days / len(df) > 0.5:
            trend = "High-Churn / Non-Directional"
            desc = "High volume relative to OI changes indicates intraday trading or rolling without net exposure shift."
        else:
            trend = "Static Hedge"
            desc = "Open Interest has remained relatively stable (Held Inventory)."
            
    return trend, desc

# ==========================================
# UI & MAIN LOGIC
# ==========================================

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Live Scanner", "History Analysis", "Contract Forensics"])

if app_mode == "Live Scanner":
    # --- Live Scanner UI ---
    st.title("🛡️ Tail-Risk Convexity Screener")
    st.caption("Compare crash protection across multiple assets simultaneously. Results are saved to DB.")

    with st.sidebar:
        st.header("Settings")
        ticker_input = st.text_input("Enter Tickers (comma-separated)", value="SPY, QQQ, IWM", help="Example: SPY, QQQ, IWM, HYG, FXI")
        
        st.subheader("Time Horizon & Speed")
        min_dte = st.number_input("Min DTE", value=30)
        max_dte = st.number_input("Max DTE", value=90)
        max_dates = st.slider("Max Dates per Ticker", 1, 10, 3)

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

        settings = {"min_dte": min_dte, "max_dte": max_dte, "min_otm": min_otm, "crash_drop": crash_drop}
        current_run_id, current_run_ts = log_scan_run(tickers, settings)
        
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
                if err: continue
                
                df, crash_price = calculate_metrics(raw_df, current_price, crash_drop)
                df['Ticker'] = ticker
                df['underlying_price'] = current_price

                try:
                    save_batch_results(current_run_id, current_run_ts, df)
                except Exception as e:
                    errors.append(f"DB Error for {ticker}: {e}")

                mask_otm = df['otm_pct'] >= min_otm
                mask_prem = df['prem_frac'] <= max_prem_pct
                mask_vol = df['volume'] >= min_vol
                mask_oi = df['openInterest'] >= min_oi
                filtered = df[mask_otm & mask_prem & mask_vol & mask_oi].copy()
                
                if not filtered.empty:
                    master_results.append(filtered)
            except Exception as e:
                errors.append(f"Error scanning {ticker}: {str(e)}")
        progress_bar.empty()

        if not master_results:
            st.warning("No options found matching your criteria. Raw data saved to History.")
            if errors: st.write(errors)
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
            st.download_button("📥 Download Results (CSV)", csv, "convexity_batch_scan.csv", "text/csv")
            
            view = display_data[['Ticker', 'expiration', 'strike', 'lastPrice', 'volume', 'openInterest', 'crash_value', 'crash_multiple', 'otm_pct']].copy()
            view.columns = ['Ticker', 'Expiration', 'Strike', 'Cost Now', 'Vol', 'Open Int', 'Value in Crash', 'Multiplier (x)', 'OTM %']

            def bold_top_rows(x):
                return ['font-weight: bold' if i < 3 else '' for i in range(len(x))]

            st.dataframe(view.style.format({
                'Strike': '${:,.0f}', 'Cost Now': '${:.2f}', 'Value in Crash': '${:.2f}',
                'Multiplier (x)': '{:.1f}x', 'OTM %': '{:.1%}', 'Vol': '{:,.0f}', 'Open Int': '{:,.0f}'
            }).background_gradient(subset=['Multiplier (x)'], cmap='Greens').apply(bold_top_rows, axis=0), use_container_width=True)

            # --- Chart ---
            st.divider()
            chart_data = final_df.copy()
            chart_data["dte"] = chart_data["dte"].astype(float)
            chart_data["expiration_str"] = chart_data["expiration"].astype(str)
            summary = chart_data.groupby(["Ticker", "expiration_str", "dte"]).agg(
                avg_multiple=("crash_multiple", "mean"), count=("crash_multiple", "size")).reset_index()

            if not summary.empty:
                st.markdown("### 📆 Convexity Comparison (Color by Ticker)")
                fig, ax = plt.subplots(figsize=(10, 5))
                unique_tickers = summary['Ticker'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tickers)))
                color_map = dict(zip(unique_tickers, colors))
                
                for ticker in unique_tickers:
                    subset = summary[summary['Ticker'] == ticker]
                    ax.scatter(subset["dte"], subset["avg_multiple"], s=subset["count"] * 50.0, alpha=0.7, label=ticker, color=color_map[ticker], edgecolors='black')

                ax.set_xlabel("Days to Expiration")
                ax.set_ylabel("Avg Multiplier (x)")
                ax.grid(True, linestyle='--', alpha=0.3)
                legend_elements = [Line2D([0], [0], marker='o', color='w', label=t, markerfacecolor=color_map[t], markersize=10, markeredgecolor='black') for t in unique_tickers]
                ax.legend(handles=legend_elements, title="Ticker", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)

elif app_mode == "History Analysis":
    st.title("📜 Market Memory & Repricing")
    runs = get_history_runs()
    
    if runs.empty:
        st.warning("No history found. Run a scan in 'Live Scanner' first.")
    else:
        st.subheader("Compare Scan Runs")
        col1, col2 = st.columns(2)
        run_opts = runs.apply(lambda x: f"ID {x['run_id']}: {x['run_ts_utc']} ({x['tickers']})", axis=1).tolist()
        
        with col1:
            new_run_str = st.selectbox("Newer Run", run_opts, index=0)
            new_run_id = int(new_run_str.split(":")[0].replace("ID ", ""))
        with col2:
            default_old = 1 if len(run_opts) > 1 else 0
            old_run_str = st.selectbox("Older Run", run_opts, index=default_old)
            old_run_id = int(old_run_str.split(":")[0].replace("ID ", ""))
            
        if st.button("Generate Comparison"):
            if new_run_id == old_run_id:
                st.error("Please select different runs.")
            else:
                df_comp = get_repricing_data(new_run_id, old_run_id)
                if df_comp.empty:
                    st.info("No overlapping contracts found.")
                else:
                    df_comp['price_chg_pct'] = (df_comp['price_new'] - df_comp['price_old']) / df_comp['price_old'].replace(0, np.nan)
                    df_comp['mult_chg_pct'] = (df_comp['mult_new'] - df_comp['mult_old']) / df_comp['mult_old'].replace(0, np.nan)
                    df_comp['repricing_score'] = (df_comp['price_chg_pct'].fillna(0) + 0.5 * df_comp['mult_chg_pct'].fillna(0) + 0.1 * np.log1p(df_comp['vol_new']))
                    df_comp = df_comp.sort_values('repricing_score', ascending=False)
                    
                    st.write(f"Comparing **{len(df_comp)}** matching contracts.")
                    st.dataframe(df_comp[['ticker', 'expiration', 'strike', 'price_new', 'price_chg_pct', 'mult_new', 'mult_chg_pct', 'vol_new', 'repricing_score']].head(20).style.format({
                        'strike': '{:,.1f}', 'price_new': '${:.2f}', 'price_chg_pct': '{:+.1%}', 'mult_new': '{:.1f}x', 'mult_chg_pct': '{:+.1%}', 'vol_new': '{:,.0f}', 'repricing_score': '{:.2f}'
                    }).background_gradient(subset=['repricing_score'], cmap='coolwarm'), use_container_width=True)

elif app_mode == "Contract Forensics":
    st.title("🕵️ Single-Contract Forensics")
    st.caption("Track positioning evolution: Accreting, Static, or Unwinding.")

    avail = get_available_contracts()
    
    if avail.empty:
        st.warning("No data found. Please run the 'Live Scanner' first to populate the database.")
    else:
        # --- SELECTION HEADER ---
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                sel_ticker = st.selectbox("Ticker", avail['ticker'].unique())
                avail_exp = avail[avail['ticker'] == sel_ticker]['expiration'].unique()
            with col2:
                sel_exp = st.selectbox("Expiration", sorted(avail_exp))
                avail_str = avail[(avail['ticker'] == sel_ticker) & (avail['expiration'] == sel_exp)]['strike'].unique()
            with col3:
                sel_strike = st.selectbox("Strike", sorted(avail_str))
            with col4:
                st.write("") # Spacer
                st.write("") # Spacer
                run_analysis = st.button("🔎 Run Forensics", type="primary")

        st.divider()

        if run_analysis:
            # 1. Fetch Data
            raw_df = get_contract_history(sel_ticker, sel_exp, sel_strike)
            
            if raw_df.empty:
                st.error("No history found for this contract.")
            elif len(raw_df) < 2:
                st.warning("Not enough data points (need at least 2 scan runs) to perform time-series analysis.")
                st.dataframe(raw_df)
            else:
                # 2. Run Analysis
                df = analyze_microstructure(raw_df)
                verdict, reason = generate_verdict(df)
                
                # --- VERDICT BANNER ---
                verdict_color = "#2b6cb0" # Blue
                if "Accreting" in verdict: verdict_color = "#2f855a" # Green
                if "Unwind" in verdict: verdict_color = "#c53030" # Red
                if "Churn" in verdict: verdict_color = "#dd6b20" # Orange
                
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: rgba(200, 200, 200, 0.1); border-left: 5px solid {verdict_color};">
                    <h3 style="margin:0; color: {verdict_color};">{verdict}</h3>
                    <p style="margin:0; font-size: 1.1em;">{reason}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.caption("⚠️ **Limitations:** Database does not currently store Bid/Ask spread or Implied Volatility. Analysis is based strictly on Price, Volume, and Open Interest dynamics.")
                
                # --- VISUALIZATION ---
                st.subheader("Time-Series Evolution")
                
                # Dual Axis Chart: Price vs Open Interest
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # OI Area Plot
                color_oi = 'tab:blue'
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Open Interest', color=color_oi, fontweight='bold')
                ax1.fill_between(df['snapshot_ts_utc'], df['open_interest'], color=color_oi, alpha=0.1)
                ax1.plot(df['snapshot_ts_utc'], df['open_interest'], color=color_oi, marker='o', label='Open Interest')
                ax1.tick_params(axis='y', labelcolor=color_oi)
                
                # Price Line Plot
                ax2 = ax1.twinx()
                color_price = 'tab:green'
                ax2.set_ylabel('Option Price ($)', color=color_price, fontweight='bold')
                ax2.plot(df['snapshot_ts_utc'], df['last_price'], color=color_price, linestyle='--', marker='x', linewidth=2, label='Price')
                ax2.tick_params(axis='y', labelcolor=color_price)
                
                plt.title(f"Positioning Structure: {sel_ticker} {sel_exp} ${sel_strike}P")
                ax1.grid(True, linestyle=':', alpha=0.6)
                
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                fig.autofmt_xdate()
                st.pyplot(fig)
                
                # --- SUMMARY TABLE ---
                st.subheader("Microstructure Ledger")
                
                display_cols = ['snapshot_ts_utc', 'last_price', 'volume', 'open_interest', 'delta_oi', 'action', 'is_churn', 'underlying_price']
                
                def color_action(val):
                    color = ''
                    if val == 'Accumulation': color = 'green'
                    elif val == 'Distribution': color = 'red'
                    return f'color: {color}; font-weight: bold' if color else ''

                styled_df = df[display_cols].sort_values('snapshot_ts_utc', ascending=False).style.format({
                    'snapshot_ts_utc': '{:%Y-%m-%d %H:%M}',
                    'last_price': '${:.2f}',
                    'volume': '{:,.0f}',
                    'open_interest': '{:,.0f}',
                    'delta_oi': '{:+,.0f}',
                    'underlying_price': '${:,.2f}'
                }).map(color_action, subset=['action'])
                
                st.dataframe(styled_df, use_container_width=True)
