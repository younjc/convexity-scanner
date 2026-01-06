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
# NAVIGATION HANDLER (Fixes API Exception)
# ==========================================
# This must run before the sidebar widget is instantiated
if 'nav_target' in st.session_state:
    st.session_state['nav_radio'] = st.session_state['nav_target']
    del st.session_state['nav_target']

# Callback for "Current Holdings" buttons (Safe Navigation)
def go_to_forensics(ticker, exp, strike):
    st.session_state['f_ticker'] = ticker
    st.session_state['f_exp'] = exp
    st.session_state['f_strike'] = strike
    st.session_state['nav_radio'] = "Contract Forensics"
    st.session_state['trigger_forensics'] = True

# ==========================================
# DATABASE MANAGER (SQLite)
# ==========================================

DB_FILE = "convexity.db"

def get_connection():
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
    try:
        c.execute("SELECT implied_volatility FROM option_snapshots LIMIT 1")
    except sqlite3.OperationalError:
        try:
            c.execute("ALTER TABLE option_snapshots ADD COLUMN implied_volatility REAL")
        except:
            pass 

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
            implied_volatility REAL,
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

    # 4. Watchlist Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            expiration TEXT,
            strike REAL,
            date_added TEXT,
            UNIQUE(ticker, expiration, strike)
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

    contracts_data = []
    for _, row in df_results.iterrows():
        contracts_data.append((row['Ticker'], row['expiration'], row['strike'], 'P'))
    
    c.executemany("""
        INSERT OR IGNORE INTO option_contracts (ticker, expiration, strike, option_type)
        VALUES (?, ?, ?, ?)
    """, contracts_data)
    
    tickers = df_results['Ticker'].unique()
    placeholders = ','.join(['?']*len(tickers))
    query = f"SELECT contract_id, ticker, expiration, strike FROM option_contracts WHERE ticker IN ({placeholders}) AND option_type='P'"
    c.execute(query, tuple(tickers))
    
    id_map = {}
    for row in c.fetchall():
        key = (row[1], row[2], row[3])
        id_map[key] = row[0]

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
                row.get('impliedVolatility', 0),
                row['dte'],
                row['otm_pct'],
                row['prem_frac'],
                row['crash_value'],
                row['crash_multiple']
            ))
            
    c.executemany("""
        INSERT OR IGNORE INTO option_snapshots 
        (run_id, contract_id, snapshot_ts_utc, underlying_price, last_price, volume, open_interest, implied_volatility, dte, otm_pct, prem_frac, crash_value, crash_multiple)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, snapshots_data)
    
    conn.commit()
    conn.close()

# --- Watchlist DB Functions ---
def add_to_watchlist(ticker, exp, strike):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO watchlist (ticker, expiration, strike, date_added) VALUES (?, ?, ?, ?)",
                  (ticker.upper(), exp, strike, datetime.utcnow().isoformat()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_watchlist():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM watchlist", conn)
    conn.close()
    return df

def delete_from_watchlist(item_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()

def get_watchlist_data_history():
    conn = get_connection()
    query = """
        SELECT 
            w.ticker, w.expiration, w.strike,
            s.snapshot_ts_utc, s.last_price, s.volume, s.open_interest, s.implied_volatility, s.underlying_price, s.otm_pct
        FROM watchlist w
        JOIN option_contracts c ON w.ticker = c.ticker AND w.expiration = c.expiration AND w.strike = c.strike
        JOIN option_snapshots s ON c.contract_id = s.contract_id
        ORDER BY s.snapshot_ts_utc ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- Standard DB Functions ---
def get_history_runs():
    conn = get_connection()
    df = pd.read_sql("SELECT run_id, run_ts_utc, tickers FROM scan_runs ORDER BY run_id DESC", conn)
    conn.close()
    return df

def get_contract_history(ticker, expiration, strike):
    conn = get_connection()
    query = """
        SELECT 
            s.snapshot_ts_utc, s.last_price, s.volume, s.open_interest, s.implied_volatility, s.crash_multiple, s.underlying_price
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

# --- Functions to Load Previous Run ---
def get_last_run_id():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT run_id FROM scan_runs ORDER BY run_id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def get_run_data(run_id):
    conn = get_connection()
    query = """
        SELECT 
            c.ticker as Ticker, c.expiration, c.strike, 
            s.last_price as lastPrice, s.volume, s.open_interest as openInterest, 
            s.crash_value, s.crash_multiple, s.otm_pct, s.dte, s.underlying_price
        FROM option_snapshots s
        JOIN option_contracts c ON s.contract_id = c.contract_id
        WHERE s.run_id = ?
    """
    df = pd.read_sql(query, conn, params=(run_id,))
    conn.close()
    return df

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
                if 'impliedVolatility' not in puts.columns:
                    puts['impliedVolatility'] = 0
                all_puts.append(puts)
        except Exception:
            continue
            
    if not all_puts:
        return None, f"No puts found for {ticker_symbol}.", warning_msg
        
    df = pd.concat(all_puts, ignore_index=True)
    return df, None, warning_msg

def get_closest_expiration(ticker_obj, target_date_str, debug_log=None):
    try:
        valid_dates = ticker_obj.options
        if not valid_dates:
            if debug_log is not None: debug_log.append("⚠️ Exchange returned NO expiration dates.")
            return None
        
        if target_date_str in valid_dates:
            return target_date_str
            
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        closest_date = None
        min_diff = 999
        
        for d_str in valid_dates:
            try:
                d_obj = datetime.strptime(d_str, '%Y-%m-%d').date()
                diff = abs((d_obj - target_date).days)
                if diff < min_diff:
                    min_diff = diff
                    closest_date = d_str
            except:
                continue
        
        if min_diff < 15:
            if debug_log is not None: debug_log.append(f"✅ Auto-matched: {target_date_str} -> {closest_date}")
            return closest_date
        else:
            if debug_log is not None: debug_log.append(f"❌ No date found near {target_date_str}.")
            return None

    except Exception as e:
        if debug_log is not None: debug_log.append(f"Date Logic Error: {e}")
        return None

def smart_fetch_contract(ticker, user_exp, user_strike, debug_log=None):
    tk = yf.Ticker(ticker)
    
    # 1. Resolve Expiration
    best_exp = get_closest_expiration(tk, user_exp, debug_log)
    if not best_exp:
        return None, None, None
        
    try:
        chain = tk.option_chain(best_exp)
        puts = chain.puts
        
        if puts.empty:
            if debug_log is not None: debug_log.append(f"Chain empty for {ticker} {best_exp}")
            return None, None, None

        # 2. Resolve Strike
        puts['diff'] = abs(puts['strike'] - user_strike)
        closest_row = puts.loc[puts['diff'].idxmin()]
        max_allowed_diff = max(2.5, user_strike * 0.015)
        
        if closest_row['diff'] > max_allowed_diff:
            if debug_log is not None: debug_log.append(f"Strike Mismatch: Wanted {user_strike}, found {closest_row['strike']}")
            return None, best_exp, None
            
        contract = closest_row.copy()
        contract['expiration'] = best_exp
        if 'impliedVolatility' not in contract:
            contract['impliedVolatility'] = 0
            
        return contract, best_exp, closest_row['strike']
    except Exception as e:
        if debug_log is not None: debug_log.append(f"Fetch Error {ticker}: {e}")
        return None, None, None

def calculate_metrics(df, underlying_price, crash_drop=0.35):
    df['dte'] = df['expiration'].apply(days_until)
    
    if underlying_price > 0:
        df['otm_pct'] = (underlying_price - df['strike']) / underlying_price
        df['prem_frac'] = df['lastPrice'] / underlying_price
        crash_price = underlying_price * (1 - crash_drop)
        df['crash_value'] = (df['strike'] - crash_price).clip(lower=0)
    else:
        df['otm_pct'] = 0
        df['prem_frac'] = 0
        df['crash_value'] = 0
    
    df['crash_multiple'] = np.where(
        df['lastPrice'] > 0.01, 
        df['crash_value'] / df['lastPrice'], 
        0
    )
    return df, 0

def analyze_microstructure(df):
    df = df.copy()
    df['snapshot_ts_utc'] = pd.to_datetime(df['snapshot_ts_utc'])
    df = df.sort_values('snapshot_ts_utc')
    
    df['delta_oi'] = df['open_interest'].diff().fillna(0)
    df['delta_price'] = df['last_price'].diff().fillna(0)
    
    def classify_day(row):
        if row['delta_oi'] > 0: return "Accretion"
        elif row['delta_oi'] < 0: return "Unwind"
        else: return "Hold"
            
    df['action'] = df.apply(classify_day, axis=1)
    
    mask_churn = (df['volume'] > 50) & (df['volume'] > 5 * df['delta_oi'].abs())
    df.loc[mask_churn, 'action'] = "Churn"
    
    return df

def generate_verdict(df):
    if len(df) < 2:
        return "Insufficient Data", "Need at least 2 data points to analyze trend."
        
    start_oi = df['open_interest'].iloc[0]
    end_oi = df['open_interest'].iloc[-1]
    net_oi_change = end_oi - start_oi
    
    if net_oi_change > (start_oi * 0.10) and net_oi_change > 50:
        trend = "Accreting Hedge"
        desc = "Consistent increase in Open Interest suggests net new positioning (risk adding)."
    elif net_oi_change < -(start_oi * 0.10):
        trend = "Active Unwind"
        desc = "Significant decrease in Open Interest suggests closing of positions."
    else:
        churn_days = (df['action'] == "Churn").sum()
        if churn_days / len(df) > 0.5:
            trend = "High-Churn / Non-Directional"
            desc = "High volume relative to OI changes indicates intraday trading or rolling."
        else:
            trend = "Static Hedge"
            desc = "Open Interest has remained relatively stable (Held Inventory)."
            
    return trend, desc

# ==========================================
# UI & MAIN LOGIC
# ==========================================

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Live Scanner", "Put Watchlist", "History Analysis", "Contract Forensics"], key="nav_radio")

# -----------------------------------------------------------------------------
# MODE: LIVE SCANNER
# -----------------------------------------------------------------------------
if app_mode == "Live Scanner":
    st.title("🛡️ Tail-Risk Convexity Screener")
    st.caption("Scan entire chains to find cheap crash protection.")

    with st.sidebar:
        st.header("Settings")
        ticker_input = st.text_input("Enter Tickers (comma-separated)", value="SPY, QQQ, IWM")
        
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
        
        # --- BUTTONS ---
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_btn = st.button("Run Batch Scan", type="primary")
        with col_btn2:
            load_last_btn = st.button("📂 Load Last Run")

    # --- ACTION HANDLERS ---
    
    # 1. LOAD LAST RUN HANDLER
    if load_last_btn:
        last_id = get_last_run_id()
        if last_id:
            with st.spinner("Retrieving last run from database..."):
                df_last = get_run_data(last_id)
                if not df_last.empty:
                    df_last = df_last.sort_values('crash_multiple', ascending=False)
                    st.session_state['scan_results'] = df_last
                    st.session_state['force_full_view'] = True
                    st.success(f"Loaded {len(df_last)} contracts from Run ID {last_id}")
        else:
            st.warning("No previous runs found in database.")

    # 2. NEW SCAN HANDLER
    if run_btn:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        if not tickers:
            st.error("Please enter at least one ticker.")
        else:
            settings = {"min_dte": min_dte, "max_dte": max_dte, "min_otm": min_otm, "crash_drop": crash_drop}
            current_run_id, current_run_ts = log_scan_run(tickers, settings)
            
            master_results = []
            errors = []
            progress_bar = st.progress(0, text="Starting Batch Scan...")
            
            for idx, ticker in enumerate(tickers):
                progress_bar.progress(int((idx / len(tickers)) * 100), text=f"Scanning {ticker}...")
                try:
                    stock = yf.Ticker(ticker)
                    try:
                        hist = stock.history(period="1d")
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                        else:
                            current_price = stock.fast_info['last_price']
                    except:
                        current_price = 0
                    
                    if current_price == 0:
                        errors.append(f"Could not get price for {ticker}")
                        continue

                    raw_df, err, warn = get_options_data(ticker, min_dte, max_dte, max_dates)
                    if err: continue
                    
                    df, _ = calculate_metrics(raw_df, current_price, crash_drop)
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
                st.warning("No options found. Raw data saved to History.")
                if errors: st.write(errors)
            else:
                final_df = pd.concat(master_results, ignore_index=True)
                final_df = final_df.sort_values('crash_multiple', ascending=False)
                
                st.session_state['scan_results'] = final_df
                st.session_state['force_full_view'] = False
                st.rerun()

    # --- DISPLAY LOGIC (PERSISTENT) ---
    if 'scan_results' in st.session_state and not st.session_state['scan_results'].empty:
        final_df = st.session_state['scan_results']
        
        show_full = show_all_results or st.session_state.get('force_full_view', False)
        
        if not show_full:
            display_data = final_df.head(20).copy()
            st.info(f"Showing Top 20 of {len(final_df)} candidates. (Live Scan Mode)")
        else:
            display_data = final_df.copy()
            st.info(f"Showing all {len(final_df)} candidates.")

        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results (CSV)", csv, "convexity_batch_scan.csv", "text/csv")
        
        # --- INTERACTIVE RESULTS TABLE ---
        st.write("### 🎯 Scan Results (Select to Add to Watchlist)")
        
        view = display_data[['Ticker', 'expiration', 'strike', 'lastPrice', 'volume', 'openInterest', 'crash_value', 'crash_multiple', 'otm_pct']].copy()
        view.insert(0, "Add", False) 
        
        edited_df = st.data_editor(
            view,
            column_config={
                "Add": st.column_config.CheckboxColumn("Add?", help="Check to add to Watchlist", default=False),
                "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                "expiration": st.column_config.TextColumn("Expiration", disabled=True),
                "strike": st.column_config.NumberColumn("Strike", format="$%.0f", disabled=True),
                "lastPrice": st.column_config.NumberColumn("Cost Now", format="$%.2f", disabled=True),
                "volume": st.column_config.NumberColumn("Vol", format="%.0f", disabled=True),
                "openInterest": st.column_config.NumberColumn("Open Int", format="%.0f", disabled=True),
                "crash_value": st.column_config.NumberColumn("Value in Crash", format="$%.2f", disabled=True),
                "crash_multiple": st.column_config.NumberColumn("Multiplier", format="%.1fx", disabled=True),
                "otm_pct": st.column_config.NumberColumn("OTM %", format="%.1f%%", disabled=True),
            },
            hide_index=True,
            use_container_width=True
        )

        if st.button("➕ Add Selected to Watchlist", type="primary"):
            selected_rows = edited_df[edited_df["Add"] == True]
            
            if selected_rows.empty:
                st.warning("No contracts selected.")
            else:
                count = 0
                for index, row in selected_rows.iterrows():
                    if add_to_watchlist(row['Ticker'], row['expiration'], row['strike']):
                        count += 1
                
                if count > 0:
                    st.success(f"✅ Successfully added {count} contracts to Watchlist!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("Selected contracts were already in the Watchlist.")

# -----------------------------------------------------------------------------
# MODE: PUT WATCHLIST
# -----------------------------------------------------------------------------
elif app_mode == "Put Watchlist":
    st.title("📋 Put Option Watchlist")
    st.caption("Monitor specific contracts. Auto-corrects dates if misaligned.")

    with st.expander("Manage Watchlist", expanded=True):
        col_in1, col_in2, col_in3, col_in4 = st.columns(4)
        with col_in1:
            w_ticker = st.text_input("Ticker", value="SPY")
        with col_in2:
            w_exp = st.text_input("Expiration (YYYY-MM-DD)", value="2025-06-20")
        with col_in3:
            w_strike = st.number_input("Strike", value=450.0, step=1.0)
        with col_in4:
            st.write("")
            st.write("")
            if st.button("Add to Watchlist"):
                if add_to_watchlist(w_ticker, w_exp, w_strike):
                    st.success(f"Added {w_ticker} {w_exp} {w_strike}P")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Could not add. Duplicate entry?")

        watchlist_df = get_watchlist()
        if not watchlist_df.empty:
            st.subheader("Current Holdings")
            cols_header = st.columns([1, 2, 1, 1, 1])
            cols_header[0].write("**Ticker**")
            cols_header[1].write("**Contract**")
            cols_header[2].write("**Added**")
            cols_header[3].write("**Analyze**")
            cols_header[4].write("**Delete**")
            
            st.divider()

            for idx, row in watchlist_df.iterrows():
                c1, c2, c3, c4, c5 = st.columns([1, 2, 1, 1, 1])
                with c1: st.write(f"**{row['ticker']}**")
                with c2: st.write(f"{row['expiration']} **${row['strike']}P**")
                with c3: st.write(f"{row['date_added'][:10]}")
                
                # FIX: Use Callback to prevent "modified after instantiated" error
                with c4:
                    st.button("🔎", key=f"analyze_{row['id']}", 
                              on_click=go_to_forensics, 
                              args=(row['ticker'], row['expiration'], row['strike']))
                        
                with c5: 
                    if st.button("🗑️", key=f"del_{row['id']}"):
                        delete_from_watchlist(row['id'])
                        st.rerun()
        else:
            st.info("Watchlist is empty. Add contracts above.")

    st.divider()

    if not watchlist_df.empty:
        col_act1, col_act2 = st.columns([1, 4])
        with col_act1:
            if st.button("🔄 Update Watchlist Data", type="primary"):
                progress_bar = st.progress(0, text="Updating Watchlist...")
                results = []
                updates_made = []
                debug_logs = []
                
                run_id, run_ts = log_scan_run(watchlist_df['ticker'].unique().tolist(), {"type": "watchlist_update"})
                total_items = len(watchlist_df)
                
                conn = get_connection()
                c = conn.cursor()
                
                for i, row in watchlist_df.iterrows():
                    progress_bar.progress(int((i) / total_items * 100), text=f"Fetching {row['ticker']}...")
                    try:
                        tk_obj = yf.Ticker(row['ticker'])
                        
                        price = 0
                        try:
                            hist = tk_obj.history(period="1d")
                            if not hist.empty:
                                price = hist['Close'].iloc[-1]
                            else:
                                price = tk_obj.fast_info['last_price']
                        except:
                            debug_logs.append(f"{row['ticker']}: Failed to get underlying price.")
                            price = 0
                        
                        opt_data, real_exp, real_strike = smart_fetch_contract(row['ticker'], row['expiration'], row['strike'], debug_logs)
                        
                        if opt_data is not None:
                            if real_exp != row['expiration'] or abs(real_strike - row['strike']) > 0.01:
                                c.execute("UPDATE watchlist SET expiration = ?, strike = ? WHERE id = ?", (real_exp, real_strike, row['id']))
                                updates_made.append(f"{row['ticker']}: {row['expiration']} -> {real_exp}")

                            d = opt_data.to_dict()
                            d['Ticker'] = row['ticker']
                            d['expiration'] = real_exp 
                            
                            processed, _ = calculate_metrics(pd.DataFrame([d]), price, 0.25)
                            processed['underlying_price'] = price
                            results.append(processed)
                    except Exception as e:
                        debug_logs.append(f"CRITICAL Error on {row['ticker']}: {e}")
                
                conn.commit()
                conn.close()
                progress_bar.empty()
                
                with st.expander("View Debug Logs (If data is missing)", expanded=True):
                    for log in debug_logs:
                        st.text(log)

                if updates_made:
                    st.info(f"ℹ️ Auto-corrected {len(updates_made)} contracts to valid dates.")
                
                if results:
                    final_df = pd.concat(results, ignore_index=True)
                    save_batch_results(run_id, run_ts, final_df)
                    st.success("Watchlist updated!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("Could not fetch data. Check 'View Debug Logs' below.")

    st.subheader("📊 Watchlist Dashboard")
    hist_data = get_watchlist_data_history()
    
    if hist_data.empty:
        st.warning("No data found. Click 'Update Watchlist Data'.")
    else:
        hist_data['snapshot_ts_utc'] = pd.to_datetime(hist_data['snapshot_ts_utc'])
        hist_data = hist_data.sort_values(['ticker', 'expiration', 'strike', 'snapshot_ts_utc'])
        
        analyzed_frames = []
        for (t, e, s), group in hist_data.groupby(['ticker', 'expiration', 'strike']):
            analyzed_frames.append(analyze_microstructure(group))
        
        if analyzed_frames:
            full_analysis = pd.concat(analyzed_frames)
            latest_view = full_analysis.sort_values('snapshot_ts_utc').groupby(['ticker', 'expiration', 'strike']).tail(1).copy()
            
            # --- AGGREGATE STATS ---
            total_oi = latest_view['open_interest'].sum()
            net_delta_oi = latest_view['delta_oi'].sum()
            weighted_otm = (latest_view['otm_pct'] * latest_view['open_interest']).sum() / total_oi if total_oi > 0 else 0
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Watchlist OI", f"{total_oi:,.0f}", delta=f"{net_delta_oi:,.0f}")
            m2.metric("Weighted OTM %", f"{weighted_otm:.1%}")
            
            # Reflexivity Check
            reflexive_count = 0
            for i, row in latest_view.iterrows():
                prev_rows = full_analysis[(full_analysis['ticker']==row['ticker']) & 
                                          (full_analysis['expiration']==row['expiration']) & 
                                          (full_analysis['strike']==row['strike']) &
                                          (full_analysis['snapshot_ts_utc'] < row['snapshot_ts_utc'])]
                if not prev_rows.empty:
                    prev = prev_rows.iloc[-1]
                    if (row['underlying_price'] < prev['underlying_price'] * 0.995) and \
                       (row['implied_volatility'] > prev['implied_volatility']) and \
                       (row['delta_oi'] > 0):
                        reflexive_count += 1

            if reflexive_count >= 2:
                m3.error(f"⚠️ REFLEXIVITY: Active ({reflexive_count})")
            elif reflexive_count == 1:
                m3.warning("⚠️ REFLEXIVITY: Early")
            else:
                m3.success("Reflexivity: None")

            def color_verdict(val):
                if val == 'Accretion': return 'color: green; font-weight: bold'
                if val == 'Unwind': return 'color: red; font-weight: bold'
                if val == 'Churn': return 'color: orange; font-weight: bold'
                return ''
                
            # --- INTERACTIVE DASHBOARD TABLE ---
            st.write("### Contract Status (Select to Analyze)")
            
            display = latest_view[['ticker', 'expiration', 'strike', 'last_price', 'volume', 'open_interest', 'delta_oi', 'implied_volatility', 'otm_pct', 'action']].copy()
            display.insert(0, "Select", False)
            
            edited_dash = st.data_editor(
                display,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=False),
                    "ticker": "Ticker",
                    "expiration": "Exp",
                    "strike": st.column_config.NumberColumn("Strike", format="$%.1f"),
                    "last_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "volume": st.column_config.NumberColumn("Vol", format="%.0f"),
                    "open_interest": st.column_config.NumberColumn("OI", format="%.0f"),
                    "delta_oi": st.column_config.NumberColumn("ΔOI", format="%+.0f"),
                    "implied_volatility": st.column_config.NumberColumn("IV", format="%.1%"),
                    "otm_pct": st.column_config.NumberColumn("OTM", format="%.1%"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("🔎 Analyze Selected Contract"):
                selected_rows = edited_dash[edited_dash["Select"] == True]
                if selected_rows.empty:
                    st.warning("Please select a row first.")
                else:
                    # Take the first selected row
                    row = selected_rows.iloc[0]
                    # Use nav_target logic because we are in the script body, not a callback
                    st.session_state['f_ticker'] = row['ticker']
                    st.session_state['f_exp'] = row['expiration']
                    st.session_state['f_strike'] = row['strike']
                    st.session_state['nav_target'] = "Contract Forensics"
                    st.session_state['trigger_forensics'] = True
                    st.rerun()

# -----------------------------------------------------------------------------
# MODE: HISTORY ANALYSIS
# -----------------------------------------------------------------------------
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
                    
                    st.dataframe(df_comp[['ticker', 'expiration', 'strike', 'price_new', 'price_chg_pct', 'mult_new', 'mult_chg_pct', 'vol_new', 'repricing_score']].head(20).style.format({
                        'strike': '{:,.1f}', 'price_new': '${:.2f}', 'price_chg_pct': '{:+.1%}', 'mult_new': '{:.1f}x', 'mult_chg_pct': '{:+.1%}', 'vol_new': '{:,.0f}', 'repricing_score': '{:.2f}'
                    }).background_gradient(subset=['repricing_score'], cmap='coolwarm'), use_container_width=True)

# -----------------------------------------------------------------------------
# MODE: CONTRACT FORENSICS
# -----------------------------------------------------------------------------
elif app_mode == "Contract Forensics":
    st.title("🕵️ Single-Contract Forensics")
    st.caption("Track positioning evolution: Accreting, Static, or Unwinding.")

    avail = get_available_contracts()
    
    if avail.empty:
        st.warning("No data found. Please run the 'Live Scanner' first.")
    else:
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            # Pre-selection Logic
            def_ticker_ix = 0
            if 'f_ticker' in st.session_state and st.session_state['f_ticker'] in avail['ticker'].unique():
                def_ticker_ix = list(avail['ticker'].unique()).index(st.session_state['f_ticker'])
            
            with col1:
                sel_ticker = st.selectbox("Ticker", avail['ticker'].unique(), index=def_ticker_ix)
                avail_exp = avail[avail['ticker'] == sel_ticker]['expiration'].unique()
            
            def_exp_ix = 0
            if 'f_exp' in st.session_state and st.session_state['f_exp'] in avail_exp:
                def_exp_ix = list(sorted(avail_exp)).index(st.session_state['f_exp'])
            
            with col2:
                sel_exp = st.selectbox("Expiration", sorted(avail_exp), index=def_exp_ix)
                avail_str = avail[(avail['ticker'] == sel_ticker) & (avail['expiration'] == sel_exp)]['strike'].unique()
            
            def_strike_ix = 0
            if 'f_strike' in st.session_state and st.session_state['f_strike'] in avail_str:
                def_strike_ix = list(sorted(avail_str)).index(st.session_state['f_strike'])
            
            with col3:
                sel_strike = st.selectbox("Strike", sorted(avail_str), index=def_strike_ix)
            
            with col4:
                st.write("")
                st.write("")
                run_analysis = st.button("🔎 Run Forensics", type="primary")

        st.divider()

        if run_analysis or st.session_state.get('trigger_forensics', False):
            # Reset Trigger
            st.session_state['trigger_forensics'] = False
            
            raw_df = get_contract_history(sel_ticker, sel_exp, sel_strike)
            if raw_df.empty:
                st.error("No history found for this contract.")
            elif len(raw_df) < 2:
                st.warning("Not enough data points to analyze trend.")
                st.dataframe(raw_df)
            else:
                df = analyze_microstructure(raw_df)
                verdict, reason = generate_verdict(df)
                
                verdict_color = "#2b6cb0"
                if "Accreting" in verdict: verdict_color = "#2f855a"
                if "Unwind" in verdict: verdict_color = "#c53030"
                if "Churn" in verdict: verdict_color = "#dd6b20"
                
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: rgba(200, 200, 200, 0.1); border-left: 5px solid {verdict_color};">
                    <h3 style="margin:0; color: {verdict_color};">{verdict}</h3>
                    <p style="margin:0; font-size: 1.1em;">{reason}</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig, ax1 = plt.subplots(figsize=(12, 6))
                color_oi = 'tab:blue'
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Open Interest', color=color_oi, fontweight='bold')
                ax1.fill_between(df['snapshot_ts_utc'], df['open_interest'], color=color_oi, alpha=0.1)
                ax1.plot(df['snapshot_ts_utc'], df['open_interest'], color=color_oi, marker='o', label='Open Interest')
                ax1.tick_params(axis='y', labelcolor=color_oi)
                
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
                
                display_cols = ['snapshot_ts_utc', 'last_price', 'volume', 'open_interest', 'delta_oi', 'action', 'underlying_price']
                def color_action(val):
                    color = ''
                    if val == 'Accumulation': color = 'green'
                    elif val == 'Distribution': color = 'red'
                    return f'color: {color}; font-weight: bold' if color else ''

                styled_df = df[display_cols].sort_values('snapshot_ts_utc', ascending=False).style.format({
                    'snapshot_ts_utc': '{:%Y-%m-%d %H:%M}', 'last_price': '${:.2f}', 'volume': '{:,.0f}',
                    'open_interest': '{:,.0f}', 'delta_oi': '{:+,.0f}', 'underlying_price': '${:,.2f}'
                }).map(color_action, subset=['action'])
                st.dataframe(styled_df, use_container_width=True)
