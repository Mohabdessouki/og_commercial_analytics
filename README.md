# Exxon Oil & Gas Pricing Forecasting & Optimization

## Project Overview

This project builds an **AI-driven decision-support system** for oil & gas pricing, demand forecasting, and profit optimization.  
It combines **real market data**, **synthetic scenario modeling**, **time-series forecasting models**, **optimization algorithms**, and **LLM-powered news insights** into an interactive **React-based UI** with a Python backend.

The goal is to support **refinery and trading teams** in:

- Predicting future oil & gas prices
- Optimizing product prices to maximize profit margins
- Performing sensitivity analysis on key cost drivers
- Interpreting real-world market news with AI

---

## System Architecture

### 1. **Backend (Python)**

- Framework: **FastAPI**
- Modules:
  - **Data Layer**
    - Fetches historical oil & gas prices (EIA, OPEC, CME futures, synthetic data fallback)
    - Creates synthetic demand curves (based on price elasticity assumptions)
  - **Forecasting Models**
    - ARIMA
    - SARIMAX
    - Exponential Smoothing
    - Prophet
    - Performance measured using **MAPE**
  - **Optimization Engine**
    - Uses forecasted demand & cost data
    - Solves for **profit-maximizing prices** under different scenarios (baseline, pessimistic, optimistic)
  - **Sensitivity Analysis**
    - Tests elasticity of demand wrt crude price, inventory levels, and external shocks
  - **Profit Margin & Cost Calculation**
    - Computes cost-to-price ratios
    - Simulates refinery margins under different supply/demand shocks

### 2. **Frontend (React + TypeScript)**

- UI Components:
  - `Controls.tsx`: Run analysis parameters (product, horizon, start date, profile)
  - `PriceForecastChart.tsx`: Plots forecasts across models
  - `OptimizationCharts.tsx`: Shows optimized vs forecasted price
  - `PerformanceBar.tsx`: Displays MAPE values per model
  - `SensitivityBar.tsx`: Elasticity impact visualization
  - `Kpis.tsx`: Displays current cost, price, demand, and margin KPIs
  - `NewsCard.tsx`: Market news + LLM insights

- Visualizations: **Recharts + D3.js**
- Deployment-ready with **Docker**.

### 3. **LLM Integration**

- Fetches **real-time oil & gas news** (EIA, OPEC, TradingView, CME futures feeds).
- Summarizes headlines into **market intelligence**:
  - Supply outlook
  - Demand signals
  - Prices & margins
  - Geopolitics & logistics impacts
- Classifies into **Bullish / Bearish** implications.
- Integrated with dashboard for **contextual decision support**.

---

## Data Sources

### **Real Data**

- **EIA**: Today in Energy, crude oil prices, refinery stats
- **CME Group**: RBOB Gasoline, NY Harbor ULSD, WTI Futures
- **OPEC Press**: Production, basket price, supply outlook

### **Synthetic Data**

- Used when real data is missing or for scenario stress testing.
- Example assumptions:
  - Gasoline ~35% markup over crude
  - Diesel ~28% markup over crude
  - Demand curve: price elasticity around -0.3 to -0.5
  - Random shocks added via Gaussian noise

---

## Key Calculations

- **Forecast Price:** From ARIMA, SARIMAX, Prophet, etc.
- **Optimal Price:** Solved by maximizing `(Price - Cost) × Demand`.
- **Profit Margin:** `(Price - Cost) / Price`.
- **Elasticity:** %ΔDemand / %ΔPrice, measured locally for crude and inventory.
- **Scenario Analysis:** Baseline, Optimistic, Pessimistic demand & supply shocks.

---

## 📈 Sample Results (UI Output)

From the dashboard (example screenshot):

- Current Price: **$94.72**
- Current Demand: **135,902 bpd**
- Current Cost: **$54.32**
- Margin: **42.7%**
- Model Performance: Prophet best (MAPE ~2.47)
- Sensitivity: Demand most sensitive to **crude oil price**
- Projected Daily Profit: **declining over horizon due to demand erosion**

---

## LLM Market Insights Example

- **Bearish**: UBS cuts Brent outlook due to oversupply
- **Bullish**: WTI Midland strengthens on regional demand
- **Bearish**: Oil prices fall as supply builds
- **Bullish**: Cash grades improve on WTI/Brent spread

**Summary:**  
Models forecast **stable-to-declining prices**, while news signals **oversupply risk** but **localized bullish spreads**. Together, the system recommends **cautious pricing optimization** to preserve margins.

---

## Running the Project

## Running with Makefile (Docker)

For containerized setup, the following **Makefile** commands are available:

* **Start containers**

```bash
make up
```

* **Stop containers**

```bash
make down
```

* **View logs**

```bash
make logs
```

This will bring up both the **frontend** and **backend** services (default: frontend at `http://localhost:3000`, backend at `http://localhost:8000`).

### Using Docker

```bash
# Build and run backend
cd backend
docker build -t exxon-backend .
docker run -p 8000:8000 exxon-backend

# Build and run frontend
cd frontend
docker build -t exxon-frontend .
docker run -p 3000:3000 exxon-frontend
```
---


## Configuration for LLM Integration

To enable the **LLM-powered news summarization and market intelligence features**, you need to create either a `.env` file or a `.config.yml` file at the project root.

### Option 1: `.env` file

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Option 2: `.config.yml` file

```yaml
OPENAI_API_KEY: your_api_key_here
OPENAI_BASE_URL: https://api.openai.com/v1
```

* The `OPENAI_API_KEY` is your personal API key from OpenAI.
* The `OPENAI_BASE_URL` can be left as default (`https://api.openai.com/v1`) unless you are routing requests through a private endpoint.

⚠️ **Important:** Do not commit your `.env` or `.config.yml` file to source control. Add them to your `.gitignore` to keep secrets safe.

---
