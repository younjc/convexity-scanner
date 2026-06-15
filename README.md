# 🛡️ Convexity Scanner

A Streamlit app for scanning options chains to identify cheap tail-risk protection (deep OTM puts with high crash multiples), tracking contracts over time, and analyzing positioning patterns via open interest microstructure.

## What It Does

The app has four modes accessible from the sidebar:

**Live Scanner** — Scans options chains for one or more tickers and filters puts by DTE window, OTM%, premium cost, volume, and open interest. For each candidate, it computes a crash multiple: how many times your premium back you'd receive in a defined crash scenario (default: 25% drop). Results are saved to a local SQLite database and can be exported as CSV.

**Put Watchlist** — Manually track specific put contracts. The app auto-corrects expiration dates and strikes if they've rolled off the exchange (smart fuzzy matching). Running an update fetches current market data and logs a new snapshot to the database.

**History Analysis** — Compares two historical scan runs side by side. Shows repricing scores across overlapping contracts, letting you see which puts have gotten more or less expensive between runs.

**Contract Forensics** — Pulls the full snapshot history for any scanned contract and classifies each data point as Accreting (rising OI), Unwinding (falling OI), Static, or Churn (high volume but flat OI). Generates a positioning verdict and checks for reflexivity signals (price drop + rising IV + rising OI occurring simultaneously).

## Getting Started

### Prerequisites

- Python 3.8+
- No API keys required — all market data is fetched via `yfinance`

### Installation

```bash
git clone https://github.com/younjc/convexity-scanner.git
cd convexity-scanner
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.

A `convexity.db` SQLite file will be created in the project directory on first run. This file stores all scan history and is required for History Analysis and Contract Forensics to work.

## Usage

### Live Scanner

1. Enter one or more tickers (comma-separated) in the sidebar, e.g. `SPY, QQQ, IWM`.
2. Configure your DTE window, OTM% floor, max premium%, liquidity filters, and crash drop assumption.
3. Click **Run Batch Scan**. Results are sorted by crash multiple descending.
4. Check the **Add?** boxes on any rows and click **Add Selected to Watchlist** to track them.
5. Use **Load Last Run** to reload the most recent scan without hitting the API again.

### Put Watchlist

1. Add contracts manually or via the Live Scanner.
2. Click **Update Watchlist Data** to fetch current prices and log a new snapshot.
3. The dashboard shows aggregate open interest, weighted OTM%, and a reflexivity indicator.

### History Analysis

1. Run at least two scans on separate occasions.
2. Select a newer and an older run from the dropdowns and click **Generate Comparison**.
3. The repricing score ranks contracts by how much they've moved (price change + multiplier change + volume).

### Contract Forensics

1. Select a ticker, expiration, and strike from the dropdowns (populated from scan history).
2. Click **Run Forensics** to see the full OI timeline, microstructure classification, and a positioning verdict.

You can also navigate directly to Contract Forensics from a Watchlist row using the 🔎 button.

## Key Concepts

**Crash Multiple** — `crash_value / premium_paid`, where crash value is the intrinsic value of the put if the underlying drops by the configured percentage. A multiple of 10x means the put pays back 10× your cost in the crash scenario.

**OTM%** — How far the strike is below the current price as a fraction of the current price. A 15% OTM put on a $500 stock has a strike around $425.

**Reflexivity Signal** — A condition where the underlying price is falling, implied volatility is rising, and open interest is increasing simultaneously. This pattern can indicate feedback-loop dynamics in hedging demand.

**Microstructure Actions** — Each snapshot is classified as:
- *Accretion*: OI increased → net new positions opened
- *Unwind*: OI decreased → positions being closed
- *Churn*: High volume but little OI change → intraday trading or rolling
- *Hold*: No significant change

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework
- [yfinance](https://github.com/ranaroussi/yfinance) — Options chain and price data
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — Data processing
- [Matplotlib](https://matplotlib.org/) — Charts in Contract Forensics
- SQLite (stdlib) — Local persistence for scan history and watchlist

## Limitations

- Data is sourced from Yahoo Finance via `yfinance`, which has rate limits and occasional gaps in options data. The scanner includes randomized delays between requests to reduce throttling.
- The `convexity.db` file is local to wherever you run the app. If you redeploy (e.g., to Streamlit Cloud), history will not persist unless you commit the database or swap in a hosted database backend.
- Crash multiples are hypothetical and assume you can exit at intrinsic value at the moment of the crash. Real execution will differ.

## Disclaimer

This tool is for informational and research purposes only. Nothing here constitutes financial advice. Options trading involves significant risk of loss.
