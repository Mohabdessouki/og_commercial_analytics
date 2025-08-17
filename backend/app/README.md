# Oil & Gas Commercial Analytics API

A comprehensive API for analyzing, forecasting, and optimizing oil & gas pricing, demand, and supply, with news aggregation and advanced sensitivity analysis. Powered by FastAPI, this backend is tailored for commercial and market analysts, refineries, and energy strategists.

---

## Features

- **Time Series Forecasting**: 
  - ARIMA, SARIMAX, Exponential Smoothing, and Prophet (if available).
  - Supports gasoline and diesel with highly realistic, profile-scaled synthetic or real (Yahoo Finance) data.

- **Market Simulation**:
  - Generate enhanced synthetic datasets with refinery profiles ("corporate", "typical_refinery", "regional_refinery").
  - Incorporates economic, industrial, and competitive factors.

- **Model Evaluation**:
  - Automatic performance metrics (MAPE, RMSE, directional accuracy).
  - Model selection based on best performance.

- **Sensitivity Analysis**:
  - Local elasticity computation for exogenous variables (e.g., crude price, inventory, utilization).
  - Clean, robust outputs for scenario planning.

- **Dynamic Price Optimization**:
  - Profit-driven price band recommendations.
  - Considers cost scenarios ("optimistic", "base", "pessimistic") and competitive landscape.

- **Comprehensive Reports & Plots**:
  - Text and visual summaries of forecasts, sensitivities, costs, and optimization.
  - Strategic recommendations for commercial actions.

- **Energy News Aggregation**:
  - Aggregates, filters, and summarizes headlines from EIA, IEA, OPEC, and Google News.
  - Uses OpenAI API for intelligent summarization (if configured).

---

## API Endpoints

### Health Check

```
GET /health
```
Returns `{ "ok": true, "time": ... }`.

---

### Run Analysis

```
POST /api/run-analysis
```
**Request Body** (`application/json`):

```json
{
  "target_product": "gasoline", // or "diesel"
  "use_real_data": true,
  "start_date": "2022-01-01",
  "train_ratio": 0.8,
  "forecast_horizon": 30,
  "profile": "typical_refinery" // or "corporate", "regional_refinery"
}
```

**Returns:**
- KPIs (current price, demand, cost)
- Model performance and forecasts
- Sensitivity analysis results
- Optimization recommendations
- Recent data preview

---

### Pricing Optimization

```
POST /api/optimize
```
**Request Body**:

```json
{
  "target_product": "gasoline",
  "cost_scenario": "base",        // "optimistic", "base", "pessimistic"
  "competitive_scenario": "base", // "weak", "base", "aggressive"
  "horizon": 30,
  "band_pct": 0.06
}
```

**Returns:** Optimal pricing and projected profit metrics for the requested horizon.

---

### News Aggregation & Summary

```
POST /api/news
```
**Request Body**:

```json
{
  "hours": 168,        // Look-back window in hours (default: 7 days)
  "max_items": 30,     // Max news items
  "model": "gpt-4o"    // Optional: OpenAI model for summarization
}
```

**Returns:** Filtered news items and Markdown summary.

---

## Installation & Running

### Requirements

- Python 3.10+
- `pip install -r requirements.txt`
  - **Main packages:** `fastapi`, `uvicorn`, `scikit-learn`, `statsmodels`, `pandas`, `matplotlib`, `yfinance`, `feedparser`, `beautifulsoup4`, `openai` (optional for news summaries)

### Running the API

```bash
uvicorn app.main:app --reload
```

The server will be available at `http://localhost:8000`.

---

## Configuration

- **OPENAI_API_KEY**:
  - Set as an environment variable for advanced news summarization.
  - If not set, summaries default to simple headline lists.

---

## Usage Notes

- **Data Source**: By default, tries to fetch real market data via Yahoo Finance (`yfinance`). If unavailable, uses high-fidelity synthetic data.
- **Profiles**: Select different refinery/market profiles to simulate various throughput scales and competitive scenarios.
- **Visualization**: Plots are generated inline and shown if running in an environment with GUI support.

---

## Example: Quick Analysis

```python
import requests

resp = requests.post("http://localhost:8000/api/run-analysis", json={
    "target_product": "gasoline",
    "use_real_data": false,
    "profile": "typical_refinery"
})
print(resp.json())
```

---
