# Deep Dive: Oil & Gas Pricing Forecasting System

# 1) What the system is trying to do

* Build a time-series–driven **pricing & demand view** for gasoline/diesel.
* Support **either real market inputs** (crude, gas, USD index) or **synthetic scenarios** (when real data is missing/insufficient).
* Train multiple models (ARIMA / SARIMAX / Exponential Smoothing / Prophet), **compare performance**, and produce a **30-day outlook**.
* Quantify **sensitivity** of prices to key drivers.
* Use the forecast, costs, and competition to **optimize retail price** for profit, then visualize everything in a React UI (and optionally summarize news).

---

# 2) Data: real vs synthetic

## 2.1 Real data pipeline

* Data comes from **Yahoo Finance** via `yfinance`:

  * `CL=F` → WTI crude front-month (Close).
  * `NG=F` → Henry Hub natural gas (Close).
  * `DX-Y.NYB` → U.S. dollar index (Close), used as a macro proxy.
* Steps in `load_real_data()`:

  1. **Download** each series for the requested window.
  2. **Forward-fill gaps** (`ffill`) because markets have holidays/weekends and some series miss days.
  3. **Align dates**: intersect the date indices across all successfully fetched series so they share the same calendar.
  4. **Reset index** and ensure a **`date` column** of dtype `datetime64[ns]`.
  5. Call `_add_synthetic_enhancements(df)` to **augment real prices** with:

     * **Retail product proxies** (gasoline/diesel) derived from crude + noise.
     * Macro indicators (gdp_growth, inflation, interest_rate, consumer_confidence).
     * Supply/demand proxies (inventory_levels, refinery_utilization, industrial_activity, temperature seasonality).
     * Cost components: variable (driven by crude) + fixed (refining + distribution).
     * Competitive prices: `competitor_gasoline`, `competitor_diesel`.
     * Demand for each product, with **price elasticity** and macro factors.

**Assumptions** (real data augmentation):

* Retail prices ≈ **linear function of crude** + idiosyncratic noise. This isn’t a refinery margin model; it’s a transparent proxy for demonstration.
* Demand is **downward sloping** with price and boosted by GDP/industrial activity and seasonality.
* Competitors move broadly with your price plus noise.

---

## 2.2 Synthetic data pipeline

When real data is unavailable or you want sandbox scenarios, `generate_enhanced_synthetic_data()` creates a **richer simulated environment**:

### 2.2.1 Building blocks

* **Dates**: daily frequency from a start date (e.g., 1 Jan 2020).
* **Crude price** = trend (45→85) + **annual seasonality** + **volatility** noise.
* **Macro**: `gdp_growth`, `inflation_rate`, `interest_rate`, `unemployment_rate`.
* **Supply**: `inventory_levels`, `refinery_utilization`, `production_capacity`.
* **Demand drivers**: `temperature_deviation` (seasonality + noise), `industrial_activity`, `consumer_confidence`.
* **Risk/volatility**: `geopolitical_risk` (regime switching), `market_volatility` (lognormal).
* **Costs**:

  * `crude_oil_cost_ratio` (e.g., 65%) → variable component.
  * refining_cost (per barrel), distribution_cost (per barrel), both with mild noise → fixed component.
* **Competitive**: `competitor_gasoline`, `competitor_diesel`, `market_share`, `brand_premium`, `location_premium`, `service_quality_score`.

### 2.2.2 Prices (retail proxies)

* `gasoline_price` and `diesel_price` = variable cost (crude × ratio) + fixed cost (refining + distribution)

  * **refining margin** (persistent series) + **demand/supply noises** (e.g., seasonality in temperature, industrial activity) + **risk premia**.

### 2.2.3 Demand functions

\[
\text{demand}_t = \text{base} + \text{seasonality} - \alpha(p_t - \bar p) + \beta_{macro} + \gamma (\text{competitor} - p_t) + \varepsilon_t
\]

* `alpha` encodes **price elasticity** (negative slope).
* Competitors’ price above yours raises demand (substitution).
* Macro terms (GDP growth, industrial/consumer indicators) lift demand.

---

# 3) Derived features

`_add_derived_features()` adds indicators commonly used for forecasting and analytics:

* **Calendar**: month, quarter, day-of-year, seasonal flags (winter/summer).
* **Technical**: ma7, ma30, momentum, volatility, spreads, advantage.
* **Cost & margin**: gross_margin, margin_percentage.
* **Utilization proxy**: demand / production_capacity.

---

# 4) Stationarity & modeling setup

* Use **ADF test** for stationarity.
* Train/test split (80/20).
* Ensure alignment of exogenous regressors with date index.

---

# 5) Forecast models

* **ARIMA** (grid over (p,d,q), AIC selection).
* **SARIMAX** with exogenous regressors.
* **Exponential Smoothing** (Holt-Winters).
* **Prophet** with yearly seasonality and optional regressors.

---

# 6) Model evaluation

Metrics:

* MAE, RMSE, MAPE, Directional Accuracy.
* Lowest MAPE model = “best”.
* Store **aligned test predictions** for visualization.

---

# 7) Future forecasts

* Forecast `h` days ahead with each model.
* Export per-model forecasts for UI overlay.

---

# 8) Cost model & margins

* Variable cost = crude × ratio.
* Fixed = refining + distribution.
* Margin = Price − Cost; Margin % = Margin / Price × 100.

---

# 9) Optimization logic

* Elastic demand function with price elasticity.
* Competitor and market share effects.
* Optimize price per day by maximizing profit:

\[
\Pi_t(P) = (P - C_t) \times Q_t(P) \times MS(P)
\]

Bound price between cost+5% and forecast+30%.

---

# 10) Sensitivity analysis

* Shock each factor ±20%.
* Retrain or approximate effect.
* Compute elasticity = (Impact+ − Impact−)/(2Δ).
* Interpret large negatives (inventory) or positives (GDP).

---

# 11) Profiles & scales

* **Corporate**: high demand, less elastic.
* **Refinery**: mid demand, medium elasticity.
* **Regional**: smaller, more elastic, higher volatility.

---

# 12) News ingestion & LLM

* Fetches RSS/HTML (EIA, IEA, OPEC, Google).
* Filters by keywords, deduplicates, time-clips.
* Summarizes with OpenAI into structured Markdown.

---

# 13) React UI & API

* **Backend**: FastAPI endpoints for `/run`, `/data`, `/news/run`, `/news/data`.
* **Frontend**: React TypeScript (`recharts`) shows:

  * Forecasts vs actuals.
  * Model performance.
  * Sensitivity results.
  * Optimization paths.
  * News summaries.

---

# 14) Key assumptions

* Retail prices linear w/ crude.
* Demand isoelastic.
* Competitor modeled as one rival price.
* Exogs extrapolated linearly.
* Fixed vs variable cost simplified.
* Optimization day-by-day.

---

# 15) Pitfalls & caveats

* Watch data leakage in feature engineering.
* Seasonal orders must match daily data.
* Prophet needs calibrated regressors.
* Elasticity assumptions dominate results.
* Competitor modeling oversimplified.
* Consider confidence intervals and risk-adjusted optimization.

---

# 16) Calibration tips

* Replace synthetic with real OPIS/AAA or internal rack data.
* Add taxes explicitly.
* Use actual competitor panels.
* Econometrically estimate demand elasticity.
* Use futures/weather forecasts for exogs.
* Add inventory/capacity constraints for multi-day optimization.

---

## TL;DR

* Ingest **real or synthetic** data → augment with macro, costs, competitors.
* Train 4 models → pick best by MAPE.
* Forecast 30 days → sensitivity ±20%.
* Optimize price daily with elasticity + competition.
* Export JSON → UI shows history, forecasts, margins, profit.
* Optional **news summarizer** adds qualitative context.

