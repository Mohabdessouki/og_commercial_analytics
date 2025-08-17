from __future__ import annotations
import os
import io
import json
from datetime import datetime
from typing import Literal, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- import your existing system code ----------
# For simplicity, we embed the class here. If you already have it as a module,
# replace this block with `from your_module import TimeSeriesOilGasPricingSystem`.
# === BEGIN: pasted class ===
# (Shortened: only minor edits – logging removed – logic intact.)
# NOTE: If you have the long class in a file, paste it here 1:1.
#       For brevity in this scaffold, we assume you paste the full class exactly
#       as provided earlier.

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import scipy.optimize as sp_opt

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

DEFAULT_PARAMS = {
    'us_daily_gasoline_demand_bpd': 9_000_000,
    'exxon_refining_capacity_bpd': 2_000_000,
    'typical_refinery_bpd': 600_000,
    'regional_refinery_bpd': 275_000,
    'gasoline_yield': 0.45,
    'diesel_yield': 0.30
}

class TimeSeriesOilGasPricingSystem:
    """
    Enhanced Time Series Forecasting System for Oil & Gas Pricing
    with Sensitivity Analysis, Cost Analysis, Competition Factors,
    and ExxonMobil-like default volumes/demand scaling.
    """

    def __init__(self, forecast_horizon=30, profile='typical_refinery'):
        """
        profile: one of {'corporate', 'typical_refinery', 'regional_refinery'}
        - 'corporate' -> ~2.0M bpd throughput
        - 'typical_refinery' -> ~600k bpd throughput
        - 'regional_refinery' -> ~275k bpd throughput
        """
        assert profile in {'corporate', 'typical_refinery', 'regional_refinery'}
        self.profile = profile

        self.forecast_horizon = forecast_horizon
        self.data = None
        self.models = {}
        self.models_meta = {}       # store metadata like ARIMA orders per target
        self.forecasts = {}
        self.model_performance = {}
        self.sensitivity_results = {}
        self.optimization_results = {}
        self.scaler = MinMaxScaler()

        # Cost structure (per barrel)
        self.cost_structure = {
            'crude_oil_cost_ratio': 0.65,
            'refining_cost': 8.0,
            'distribution_cost': 5.0,
            'retail_margin': 0.15,
            'tax_rate': 0.18
        }

        self.competitive_weights = {
            'competitor_price': 0.4,
            'market_share': 0.2,
            'brand_premium': 0.15,
            'location_premium': 0.1,
            'service_quality': 0.15
        }

        print("Enhanced Time Series Oil & Gas Pricing System Initialized")
        print(f"Forecast horizon: {forecast_horizon} days")
        print("Models available: ARIMA, SARIMAX, ExponentialSmoothing" + (", Prophet" if PROPHET_AVAILABLE else ""))
        print(f"Volume profile: {profile.replace('_',' ').title()}")

    # ----------------------- Small helpers -----------------------

    def _trend(self, series, window=30):
        s = pd.Series(series).dropna()
        if len(s) < 2:
            return 0.0
        diff = s.tail(window).diff().dropna()
        return float(diff.mean()) if not diff.empty else 0.0

    def _best_model_name(self, target_product):
        perf = self.model_performance.get(target_product, {})
        if not perf:
            return None
        return min(perf.keys(), key=lambda k: perf[k]['MAPE'])

    def _future_exog_frame(self, cols, horizon):
        rows = []
        for i in range(1, horizon + 1):
            row = {}
            for c in cols:
                last_val = self.data[c].iloc[-1]
                trend = self.data[c].iloc[-30:].diff().mean()
                row[c] = float(last_val + (trend if np.isfinite(trend) else 0.0) * i)
            rows.append(row)
        return pd.DataFrame(rows, columns=cols)

    def _profile_throughput(self):
        if self.profile == 'corporate':
            return DEFAULT_PARAMS['exxon_refining_capacity_bpd']
        elif self.profile == 'regional_refinery':
            return DEFAULT_PARAMS['regional_refinery_bpd']
        return DEFAULT_PARAMS['typical_refinery_bpd']

    def _profile_base_demands(self):
        """Return base gasoline/diesel demand (bpd) and production capacity baseline (bpd)."""
        capacity = self._profile_throughput()
        gas = capacity * DEFAULT_PARAMS['gasoline_yield']   # gasoline share
        diesel = capacity * DEFAULT_PARAMS['diesel_yield']  # diesel share
        return gas, diesel, capacity

    # -------------------- Data Generation/Loading --------------------

    def generate_enhanced_synthetic_data(self, n_samples=1000):
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

        oil_trend = np.linspace(45, 85, n_samples)
        seasonal_oil = 8 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
        volatility = np.random.normal(0, 5, n_samples)
        crude_oil_price = oil_trend + seasonal_oil + volatility

        gdp_growth = np.random.normal(2.5, 1.2, n_samples)
        inflation_rate = np.random.normal(2.8, 0.8, n_samples)
        interest_rate = np.random.normal(3.5, 1.0, n_samples)
        unemployment_rate = np.random.normal(5.2, 1.5, n_samples)

        # Scale capacity and demands to realistic Exxon-like levels
        base_gasoline_bpd, base_diesel_bpd, base_capacity_bpd = self._profile_base_demands()

        inventory_levels = self._create_persistent_series(base_capacity_bpd * 0.7,  # barrels in storage (rough)
                                                          base_capacity_bpd * 0.12, n_samples, 0.92)
        refinery_utilization = np.clip(
            self._create_persistent_series(0.90, 0.06, n_samples, 0.88), 0.6, 1.02
        )
        production_capacity = self._create_persistent_series(base_capacity_bpd,
                                                             base_capacity_bpd * 0.04, n_samples, 0.95)

        temperature_deviation = 8 * np.sin(2 * np.pi * (np.arange(n_samples) + 90) / 365.25) + \
                                np.random.normal(0, 3, n_samples)
        industrial_activity = self._create_persistent_series(88, 12, n_samples, 0.91)
        consumer_confidence = self._create_persistent_series(65, 15, n_samples, 0.85)

        geopolitical_risk = self._create_regime_series(n_samples, [0.7, 0.2, 0.1], 0.92)
        market_volatility = np.random.lognormal(0, 0.5, n_samples)

        refining_margin = self._create_persistent_series(12, 4, n_samples, 0.8)
        distribution_costs = self.cost_structure['distribution_cost'] + np.random.normal(0, 1, n_samples)

        gasoline_price = (crude_oil_price * self.cost_structure['crude_oil_cost_ratio'] +
                          self.cost_structure['refining_cost'] +
                          refining_margin + distribution_costs +
                          geopolitical_risk * 2 + temperature_deviation * 0.3)

        diesel_price = (crude_oil_price * self.cost_structure['crude_oil_cost_ratio'] +
                        self.cost_structure['refining_cost'] * 0.95 +
                        refining_margin * 0.9 + distribution_costs * 0.95 +
                        industrial_activity * 0.05)

        competitor_gasoline = gasoline_price + np.random.normal(2, 3, n_samples)
        competitor_diesel = diesel_price + np.random.normal(1.5, 2.5, n_samples)
        market_share = np.clip(np.random.normal(0.25, 0.08, n_samples), 0.05, 0.6)
        brand_premium = np.random.normal(3, 1.5, n_samples)
        location_premium = np.random.normal(1, 2, n_samples)
        service_quality_score = np.random.normal(7.5, 1.2, n_samples)

        # Demand series in barrels/day — seasonal ±8–10% around base, elastic to price
        gasoline_demand = (
            base_gasoline_bpd * (1
                                 + 0.08 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25))
            + (-0.12) * (gasoline_price - np.mean(gasoline_price)) / np.maximum(np.mean(gasoline_price), 1e-6) * base_gasoline_bpd
            + 0.02 * gdp_growth / np.maximum(np.abs(gdp_growth).mean(), 1) * base_gasoline_bpd
            + 0.01 * consumer_confidence / 100 * base_gasoline_bpd
            + (competitor_gasoline - gasoline_price) * 1000.0   # competitive effect (scaled)
            + np.random.normal(0, base_gasoline_bpd * 0.01, n_samples)
        )

        diesel_demand = (
            base_diesel_bpd * (1
                               + 0.06 * np.sin(2 * np.pi * (np.arange(n_samples) + 120) / 365.25))
            + (-0.10) * (diesel_price - np.mean(diesel_price)) / np.maximum(np.mean(diesel_price), 1e-6) * base_diesel_bpd
            + 0.03 * industrial_activity / 100 * base_diesel_bpd
            + 0.02 * gdp_growth / np.maximum(np.abs(gdp_growth).mean(), 1) * base_diesel_bpd
            + (competitor_diesel - diesel_price) * 800.0
            + np.random.normal(0, base_diesel_bpd * 0.01, n_samples)
        )

        gasoline_demand = np.maximum(gasoline_demand, 0.0)
        diesel_demand = np.maximum(diesel_demand, 0.0)

        variable_costs = crude_oil_price * self.cost_structure['crude_oil_cost_ratio']
        fixed_costs = self.cost_structure['refining_cost'] + distribution_costs
        total_costs = variable_costs + fixed_costs

        self.data = pd.DataFrame({
            'date': dates,
            'crude_oil_price': crude_oil_price,
            'gasoline_price': gasoline_price,
            'diesel_price': diesel_price,
            'gasoline_demand': gasoline_demand,
            'diesel_demand': diesel_demand,
            'gdp_growth': gdp_growth,
            'inflation_rate': inflation_rate,
            'interest_rate': interest_rate,
            'unemployment_rate': unemployment_rate,
            'consumer_confidence': consumer_confidence,
            'inventory_levels': inventory_levels,
            'refinery_utilization': refinery_utilization,
            'production_capacity': production_capacity,
            'temperature_deviation': temperature_deviation,
            'industrial_activity': industrial_activity,
            'geopolitical_risk': geopolitical_risk,
            'market_volatility': market_volatility,
            'variable_costs': variable_costs,
            'fixed_costs': fixed_costs,
            'total_costs': total_costs,
            'refining_margin': refining_margin,
            'competitor_gasoline': competitor_gasoline,
            'competitor_diesel': competitor_diesel,
            'market_share': market_share,
            'brand_premium': brand_premium,
            'location_premium': location_premium,
            'service_quality_score': service_quality_score,
        })

        self._add_derived_features()
        self.data = self.data.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)

        print(f"Generated {len(self.data)} samples with {len(self.data.columns)} features (profile-scaled volumes)")
        return self.data

    def load_real_data(self, start_date='2020-01-01', end_date=None):
        if not YFINANCE_AVAILABLE:
            print("Yahoo Finance not available. Using synthetic data.")
            return self.generate_enhanced_synthetic_data()

        print("Loading real market data...")
        try:
            real_data = {}

            oil = yf.download('CL=F', start=start_date, end=end_date, progress=False)
            if not oil.empty:
                real_data['crude_oil_price'] = oil['Close'].ffill()
                print(f"  ✓ Crude oil: {len(oil)} records")

            gas = yf.download('NG=F', start=start_date, end=end_date, progress=False)
            if not gas.empty:
                real_data['natural_gas_price'] = gas['Close'].ffill()
                print(f"  ✓ Natural gas: {len(gas)} records")

            dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date, progress=False)
            if not dxy.empty:
                real_data['usd_index'] = dxy['Close'].ffill()

            if not real_data:
                raise RuntimeError("No instruments returned data")

            idx = None
            for s in real_data.values():
                idx = s.index if idx is None else idx.intersection(s.index)
            if idx is None or len(idx) == 0:
                raise RuntimeError("No overlapping dates across series")

            aligned = [ser.reindex(idx).ffill() for ser in real_data.values()]
            df = pd.concat(aligned, axis=1)
            df.columns = list(real_data.keys())
            df = df.sort_index()

            df = df.reset_index()
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
            else:
                df.rename(columns={'index': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)

            # Enhance with realistic, profile-scaled volumes
            df = self._add_synthetic_enhancements(df)
            self.data = df

            self._add_derived_features()
            self.data = self.data.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)

            print(f"Successfully loaded {len(df)} days of real market data (profile-scaled volumes)")
            return df

        except Exception as e:
            print(f"Error loading real data: {e}")
            import traceback; traceback.print_exc()
            print("Falling back to synthetic data...")
            return self.generate_enhanced_synthetic_data()

    def _add_synthetic_enhancements(self, df):
        n_samples = len(df)
        np.random.seed(42)

        base_present = 'crude_oil_price' in df.columns
        base_price = df['crude_oil_price'].values if base_present else np.linspace(50, 80, n_samples)

        df['gasoline_price'] = base_price * 1.35 + np.random.normal(8, 3, n_samples)
        df['diesel_price'] = base_price * 1.28 + np.random.normal(6, 2.5, n_samples)

        df['gdp_growth'] = self._create_persistent_series(2.5, 1.2, n_samples, 0.9)
        df['inflation_rate'] = self._create_persistent_series(2.8, 0.8, n_samples, 0.85)
        df['interest_rate'] = self._create_persistent_series(3.5, 1.0, n_samples, 0.88)
        df['consumer_confidence'] = self._create_persistent_series(65, 15, n_samples, 0.82)

        base_gasoline_bpd, base_diesel_bpd, base_capacity_bpd = self._profile_base_demands()

        df['inventory_levels'] = self._create_persistent_series(base_capacity_bpd * 0.7,
                                                                base_capacity_bpd * 0.12, n_samples, 0.92)
        df['refinery_utilization'] = np.clip(
            self._create_persistent_series(0.90, 0.06, n_samples, 0.88), 0.6, 1.02)
        df['industrial_activity'] = self._create_persistent_series(88, 12, n_samples, 0.91)

        df['temperature_deviation'] = (8 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25) +
                                       np.random.normal(0, 3, n_samples))

        df['variable_costs'] = base_price * self.cost_structure['crude_oil_cost_ratio']
        df['fixed_costs'] = (self.cost_structure['refining_cost'] +
                             self.cost_structure['distribution_cost'] +
                             np.random.normal(0, 1, n_samples))
        df['total_costs'] = df['variable_costs'] + df['fixed_costs']

        df['competitor_gasoline'] = df['gasoline_price'] + np.random.normal(2, 3, n_samples)
        df['competitor_diesel'] = df['diesel_price'] + np.random.normal(1.5, 2.5, n_samples)
        df['market_share'] = np.clip(np.random.normal(0.25, 0.08, n_samples), 0.05, 0.6)
        df['brand_premium'] = np.random.normal(3, 1.5, n_samples)

        # Profile-scaled demands (bpd)
        df['gasoline_demand'] = np.maximum(
            base_gasoline_bpd * (1 + 0.08 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)) +
            (-0.12) * (df['gasoline_price'] - df['gasoline_price'].mean()) /
            max(df['gasoline_price'].mean(), 1e-6) * base_gasoline_bpd +
            df['gdp_growth'] * 0.02 / max(abs(df['gdp_growth']).mean(), 1) * base_gasoline_bpd +
            df['consumer_confidence'] * 0.01 / 100 * base_gasoline_bpd +
            (df['competitor_gasoline'] - df['gasoline_price']) * 1000.0 +
            np.random.normal(0, base_gasoline_bpd * 0.01, n_samples), 0)

        df['diesel_demand'] = np.maximum(
            base_diesel_bpd * (1 + 0.06 * np.sin(2 * np.pi * (np.arange(n_samples) + 120) / 365.25)) +
            (-0.10) * (df['diesel_price'] - df['diesel_price'].mean()) /
            max(df['diesel_price'].mean(), 1e-6) * base_diesel_bpd +
            df['industrial_activity'] * 0.03 / 100 * base_diesel_bpd +
            df['gdp_growth'] * 0.02 / max(abs(df['gdp_growth']).mean(), 1) * base_diesel_bpd +
            (df['competitor_diesel'] - df['diesel_price']) * 800.0 +
            np.random.normal(0, base_diesel_bpd * 0.01, n_samples), 0)

        df['production_capacity'] = self._create_persistent_series(base_capacity_bpd,
                                                                   base_capacity_bpd * 0.04, n_samples, 0.95)

        return df

    def _add_derived_features(self):
        if self.data is None or 'date' not in self.data.columns:
            return
        self.data['month'] = self.data['date'].dt.month
        self.data['quarter'] = self.data['date'].dt.quarter
        self.data['day_of_year'] = self.data['date'].dt.dayofyear
        self.data['is_winter'] = self.data['month'].isin([12, 1, 2]).astype(int)
        self.data['is_summer'] = self.data['month'].isin([6, 7, 8]).astype(int)

        for product in ['gasoline', 'diesel']:
            price_col = f'{product}_price'
            if price_col in self.data.columns:
                self.data[f'{product}_ma7'] = self.data[price_col].rolling(7, min_periods=1).mean()
                self.data[f'{product}_ma30'] = self.data[price_col].rolling(30, min_periods=1).mean()
                self.data[f'{product}_momentum'] = (self.data[f'{product}_ma7'] /
                                                    self.data[f'{product}_ma30'] - 1) * 100
                self.data[f'{product}_volatility'] = self.data[price_col].rolling(20, min_periods=1).std()
                if 'crude_oil_price' in self.data.columns:
                    self.data[f'{product}_crude_spread'] = (self.data[price_col] -
                                                            self.data['crude_oil_price'])
                comp_col = f'competitor_{product}'
                if comp_col in self.data.columns:
                    self.data[f'{product}_competitive_advantage'] = (self.data[comp_col] -
                                                                     self.data[price_col])

        if all(col in self.data.columns for col in ['gasoline_price', 'total_costs']):
            self.data['gross_margin'] = self.data['gasoline_price'] - self.data['total_costs']
            self.data['margin_percentage'] = (self.data['gross_margin'] /
                                              self.data['gasoline_price'] * 100)

        if 'production_capacity' in self.data.columns:
            for product in ['gasoline', 'diesel']:
                demand_col = f'{product}_demand'
                if demand_col in self.data.columns:
                    self.data[f'{product}_utilization'] = (self.data[demand_col] /
                                                           self.data['production_capacity'])

    def _create_persistent_series(self, mean, std, length, persistence):
        series = np.zeros(length)
        series[0] = np.random.normal(mean, std)
        for i in range(1, length):
            series[i] = (persistence * series[i - 1] +
                         (1 - persistence) * mean +
                         np.random.normal(0, std * np.sqrt(1 - persistence ** 2)))
        return series

    def _create_regime_series(self, length, probs, persistence):
        series = np.zeros(length, dtype=int)
        series[0] = np.random.choice(len(probs), p=probs)
        for i in range(1, length):
            if np.random.random() < persistence:
                series[i] = series[i - 1]
            else:
                series[i] = np.random.choice(len(probs), p=probs)
        return series

    # ----------------------- Modeling -----------------------

    def check_stationarity(self, series, name):
        result = adfuller(series.dropna())
        print(f'{name} Stationarity Test:')
        print(f'  ADF Statistic: {result[0]:.6f}')
        print(f'  p-value: {result[1]:.6f}')
        if result[1] <= 0.05:
            print(f'  ✓ {name} is stationary')
            return True
        else:
            print(f'  ⚠ {name} is non-stationary')
            return False

    def train_models(self, target_product='gasoline', train_ratio=0.8):
        print(f"\n=== Training Models for {target_product.title()} Price ===")

        price_column = f'{target_product}_price'
        if price_column not in self.data.columns:
            print(f"Error: {price_column} not found in data")
            return

        ts_data = self.data[price_column].dropna()
        train_size = int(len(ts_data) * train_ratio)
        train_data = ts_data[:train_size]
        test_data = ts_data[train_size:]

        print(f"Training size: {len(train_data)}, Test size: {len(test_data)}")
        _ = self.check_stationarity(train_data, f'{target_product.title()} Price')

        models = {}
        predictions = {}

        try:
            print("\nTraining ARIMA model...")
            best_aic = float('inf')
            best_order = (1, 1, 1)
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(train_data, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except Exception:
                            continue
            print(f"  Best ARIMA order: {best_order}")
            arima_model = ARIMA(train_data, order=best_order)
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=len(test_data))
            self.models_meta.setdefault(target_product, {})['ARIMA_order'] = best_order
            models['ARIMA'] = arima_fit
            predictions['ARIMA'] = arima_forecast
        except Exception as e:
            print(f"  ARIMA failed: {e}")

        try:
            print("\nTraining SARIMAX model...")
            exog_cols = ['crude_oil_price', 'inventory_levels', 'refinery_utilization']
            available_exog = [col for col in exog_cols if col in self.data.columns]
            if available_exog:
                exog_df = self.data[['date'] + available_exog].dropna().copy()
                target_df = self.data[['date', price_column]].dropna().copy()
                merged = target_df.merge(exog_df, on='date', how='inner').dropna()
                y = merged[price_column]
                X = merged[available_exog]
                train_len = int(len(y) * train_ratio)
                y_train, y_test = y.iloc[:train_len], y.iloc[train_len:]
                X_train, X_test = X.iloc[:train_len], X.iloc[train_len:]
                sarimax_model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
                sarimax_fit = sarimax_model.fit(disp=False)
                sarimax_forecast = sarimax_fit.forecast(steps=len(y_test), exog=X_test)
                models['SARIMAX'] = sarimax_fit
                predictions['SARIMAX'] = sarimax_forecast
            else:
                print("  No suitable exogenous variables found for SARIMAX")
        except Exception as e:
            print(f"  SARIMAX failed: {e}")

        try:
            print("\nTraining Exponential Smoothing model...")
            exp_model = ExponentialSmoothing(
                train_data,
                trend='add',
                seasonal='add' if len(train_data) > 60 else None,
                seasonal_periods=7 if len(train_data) > 60 else None
            )
            exp_fit = exp_model.fit()
            exp_forecast = exp_fit.forecast(len(test_data))
            models['ExponentialSmoothing'] = exp_fit
            predictions['ExponentialSmoothing'] = exp_forecast
        except Exception as e:
            print(f"  Exponential Smoothing failed: {e}")

        if PROPHET_AVAILABLE:
            try:
                print("\nTraining Prophet model...")
                prophet_df = pd.DataFrame({
                    'ds': self.data['date'][:train_size].values,
                    'y': train_data.values
                })
                regressor_cols = ['crude_oil_price', 'inventory_levels']
                for col in regressor_cols:
                    if col in self.data.columns:
                        prophet_df[col] = self.data[col][:train_size].values

                prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                for col in regressor_cols:
                    if col in self.data.columns:
                        prophet_model.add_regressor(col)
                prophet_model.fit(prophet_df)

                future_df = pd.DataFrame({'ds': self.data['date'][train_size:train_size + len(test_data)].values})
                for col in regressor_cols:
                    if col in self.data.columns:
                        future_df[col] = self.data[col][train_size:train_size + len(test_data)].values
                prophet_forecast = prophet_model.predict(future_df)

                models['Prophet'] = prophet_model
                predictions['Prophet'] = prophet_forecast['yhat'].values
            except Exception as e:
                print(f"  Prophet failed: {e}")

        self._evaluate_models(predictions, test_data, target_product)
        self.models[target_product] = models
        self.forecasts[target_product] = predictions
        return models, predictions

    def _evaluate_models(self, predictions, test_data, target_product):
        print(f"\n=== Model Performance for {target_product.title()} ===")
        performance = {}
        for model_name, pred in predictions.items():
            try:
                if len(pred) != len(test_data):
                    min_len = min(len(pred), len(test_data))
                    pred = np.array(pred)[:min_len]
                    test_subset = test_data.iloc[:min_len]
                else:
                    test_subset = test_data
                mae = mean_absolute_error(test_subset, pred)
                mse = mean_squared_error(test_subset, pred)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(test_subset, pred) * 100
                actual_direction = np.diff(test_subset) > 0
                pred_direction = np.diff(pred) > 0
                directional_accuracy = np.mean(actual_direction == pred_direction) * 100
                performance[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape,
                                           'Directional_Accuracy': directional_accuracy}
                print(f"\n{model_name}:")
                print(f"  MAE: ${mae:.3f}")
                print(f"  RMSE: ${rmse:.3f}")
                print(f"  MAPE: {mape:.2f}%")
                print(f"  Directional Accuracy: {directional_accuracy:.1f}%")
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")

        self.model_performance[target_product] = performance
        if performance:
            best_model = min(performance.keys(), key=lambda x: performance[x]['MAPE'])
            print(f"\n🏆 Best performing model: {best_model} (MAPE: {performance[best_model]['MAPE']:.2f}%)")
        return performance

    # ----------------------- Forecasting -----------------------

    def forecast_future(self, target_product='gasoline', forecast_days=None):
        if forecast_days is None:
            forecast_days = self.forecast_horizon
        print(f"\n=== Generating {forecast_days}-day forecast for {target_product.title()} ===")
        if target_product not in self.models:
            print(f"No trained models found for {target_product}. Please train models first.")
            return {}
        models = self.models[target_product]
        future_forecasts = {}
        for model_name, model in models.items():
            try:
                if model_name == 'ARIMA':
                    forecast = model.forecast(steps=forecast_days)
                    future_forecasts[model_name] = forecast
                elif model_name == 'SARIMAX':
                    exog_cols = ['crude_oil_price', 'inventory_levels', 'refinery_utilization']
                    available_exog = [c for c in exog_cols if c in self.data.columns]
                    if available_exog:
                        future_rows = []
                        for i in range(1, forecast_days + 1):
                            row = {}
                            for col in available_exog:
                                last_val = self.data[col].iloc[-1]
                                trend = self._trend(self.data[col].iloc[-30:], window=30)
                                row[col] = float(last_val + trend * i)
                            future_rows.append(row)
                        future_exog = pd.DataFrame(future_rows, columns=available_exog)
                        forecast = model.forecast(steps=forecast_days, exog=future_exog)
                        future_forecasts[model_name] = forecast
                elif model_name == 'ExponentialSmoothing':
                    forecast = model.forecast(forecast_days)
                    future_forecasts[model_name] = forecast
                elif model_name == 'Prophet':
                    future_dates = pd.date_range(
                        start=self.data['date'].iloc[-1] + pd.Timedelta(days=1),
                        periods=forecast_days, freq='D')
                    future_df = pd.DataFrame({'ds': future_dates})
                    regressor_cols = ['crude_oil_price', 'inventory_levels']
                    for col in regressor_cols:
                        if col in self.data.columns:
                            last_val = self.data[col].iloc[-1]
                            trend = self._trend(self.data[col].iloc[-30:], window=30)
                            future_df[col] = [float(last_val + trend * (i + 1)) for i in range(forecast_days)]
                    prophet_forecast = model.predict(future_df)
                    future_forecasts[model_name] = prophet_forecast['yhat'].values
                print(f"  ✓ {model_name}: Generated {forecast_days}-day forecast")
            except Exception as e:
                print(f"  ✗ {model_name}: Failed to generate forecast - {e}")
        return future_forecasts

    # -------------------- Clean local sensitivity --------------------

    def sensitivity_local(self, target_product='gasoline', factors=None, horizon=7, shock=0.01):
        if factors is None:
            factors = ['crude_oil_price', 'inventory_levels', 'refinery_utilization',
                       'gdp_growth', 'competitor_gasoline', 'industrial_activity',
                       'consumer_confidence', 'market_share']

        best = self._best_model_name(target_product)
        if best is None:
            print("Run training first.")
            return pd.DataFrame()

        base_forecasts = self.forecast_future(target_product, forecast_days=horizon)
        if not base_forecasts or best not in base_forecasts:
            print("No base forecast available.")
            return pd.DataFrame()
        base = np.array(base_forecasts[best], dtype=float)

        if best == 'SARIMAX':
            used = [c for c in ['crude_oil_price', 'inventory_levels', 'refinery_utilization']
                    if c in self.data.columns]
        elif best == 'Prophet':
            used = [c for c in ['crude_oil_price', 'inventory_levels'] if c in self.data.columns]
        else:
            print(f"{best} has no exogenous regressors; local sensitivity not applicable.")
            return pd.DataFrame()

        eval_factors = [f for f in factors if f in used]
        if not eval_factors:
            print(f"No exogenous factors used by {best}.")
            return pd.DataFrame()

        rows = []
        for f in eval_factors:
            if best == 'SARIMAX':
                future_exog = self._future_exog_frame(used, horizon)
                exog_pos = future_exog.copy(); exog_pos[f] = exog_pos[f] * (1 + shock)
                exog_neg = future_exog.copy(); exog_neg[f] = exog_neg[f] * (1 - shock)
                model = self.models[target_product]['SARIMAX']
                y_pos = np.array(model.forecast(steps=horizon, exog=exog_pos), dtype=float)
                y_neg = np.array(model.forecast(steps=horizon, exog=exog_neg), dtype=float)
            else:
                model = self.models[target_product]['Prophet']
                future_dates = pd.date_range(self.data['date'].iloc[-1] + pd.Timedelta(days=1),
                                             periods=horizon, freq='D')
                pos_df = pd.DataFrame({'ds': future_dates})
                neg_df = pd.DataFrame({'ds': future_dates})
                for c in used:
                    last_val = self.data[c].iloc[-1]
                    trend = self.data[c].iloc[-30:].diff().mean()
                    seq = [float(last_val + (trend if np.isfinite(trend) else 0.0) * (i+1))
                           for i in range(horizon)]
                    if c == f:
                        pos_df[c] = [v*(1+shock) for v in seq]
                        neg_df[c] = [v*(1-shock) for v in seq]
                    else:
                        pos_df[c] = seq; neg_df[c] = seq
                y_pos = model.predict(pos_df)['yhat'].values
                y_neg = model.predict(neg_df)['yhat'].values

            pct_base = np.maximum(base, 1e-6)
            el_pos = ((y_pos - base) / pct_base) / shock
            el_neg = ((y_neg - base) / pct_base) / (-shock)
            el = 0.5 * (el_pos + el_neg)

            mean_el = float(np.mean(el))
            se = float(np.std(el, ddof=1) / np.sqrt(len(el))) if len(el) > 1 else 0.0
            ci_low, ci_high = mean_el - 1.96*se, mean_el + 1.96*se

            rows.append({'factor': f, 'elasticity_mean': mean_el,
                         'ci_low': ci_low, 'ci_high': ci_high})

        out = pd.DataFrame(rows).sort_values(by='elasticity_mean',
                                             key=lambda s: np.abs(s), ascending=False)
        self.sensitivity_results.setdefault(target_product, {})['local'] = out.to_dict('records')
        print("\nClean local sensitivities (7-day, ±1%):")
        for r in out.itertuples(index=False):
            print(f"  {r.factor:>20s}: {r.elasticity_mean:+.3f}  (95% CI {r.ci_low:+.3f}..{r.ci_high:+.3f})")
        return out

    # -------------------- Dynamic optimizer --------------------

    def _forecast_daily_demand(self, target_product, horizon=30):
        demand_col = f'{target_product}_demand'
        d = self.data[demand_col].dropna()
        if len(d) < 60:
            base = float(d.tail(30).mean()) if len(d) else 1000.0
            return np.full(horizon, base, dtype=float)
        tail = d.tail(60)
        trend = tail.diff().mean()
        base = tail.mean()
        dow = self.data['date'].tail(60).dt.dayofweek.values
        sf = {k: 1.0 for k in range(7)}
        for k in range(7):
            vals = tail[dow == k]
            if len(vals) > 3:
                sf[k] = float(vals.mean() / base)
        start_dow = (self.data['date'].iloc[-1] + pd.Timedelta(days=1)).dayofweek
        out = []
        for i in range(horizon):
            day_idx = (start_dow + i) % 7
            val = (base + (trend if np.isfinite(trend) else 0.0) * i) * sf.get(day_idx, 1.0)
            out.append(max(val, 1.0))
        return np.array(out, dtype=float)

    def optimize_pricing_strategy_dynamic(self, target_product='gasoline',
                                          cost_scenario='base', competitive_scenario='base',
                                          horizon=30, band_pct=0.06):
        print(f"\n=== Dynamic Pricing Optimization for {target_product.title()} ===")
        print(f"Cost scenario: {cost_scenario}, Competitive scenario: {competitive_scenario}")

        forecasts = self.forecast_future(target_product, horizon)
        best = self._best_model_name(target_product)
        if not forecasts or best not in forecasts:
            print("No forecasts available for optimization.")
            return pd.DataFrame()
        price_ref = np.array(forecasts[best], dtype=float)

        cost_adj = {'optimistic': 0.95, 'base': 1.0, 'pessimistic': 1.10}.get(cost_scenario, 1.0)
        comp_adj = {'weak': 1.05, 'base': 1.0, 'aggressive': 0.95}.get(competitive_scenario, 1.0)

        current_cost = float(self.data['total_costs'].iloc[-1]) * cost_adj
        comp0 = float(self.data[f'competitor_{target_product}'].iloc[-1]) * comp_adj

        comp_trend = self.data[f'competitor_{target_product}'].iloc[-30:].diff().mean()
        comp_path = np.array([comp0 + (comp_trend if np.isfinite(comp_trend) else 0.0) * (i+1)
                              for i in range(horizon)], dtype=float)

        demand_fc = self._forecast_daily_demand(target_product, horizon=horizon)

        results = []
        price_elasticity = -1.8
        ms_base = float(self.data['market_share'].iloc[-1])

        for t in range(horizon):
            base_demand_t = demand_fc[t]
            comp_t = comp_path[t]
            ref_t = price_ref[t]

            def neg_profit(p):
                p = float(p)
                demand_mult = (p / max(ref_t, 1e-6)) ** price_elasticity
                d = base_demand_t * demand_mult
                comp_effect = 1 + 0.3 * (comp_t - p) / max(p, 1e-6)
                d *= comp_effect
                ms_effect = np.clip(ms_base * (1 + 0.1 * (comp_t - p) / max(p, 1e-6)), 0.05, 0.6)
                return -(p - current_cost) * d * ms_effect

            lower_band = comp_t * (1 - band_pct)
            upper_band = comp_t * (1 + band_pct)
            lb = max(current_cost * 1.05, lower_band)
            ub = max(lb + 1e-3, upper_band)

            res = sp_opt.minimize_scalar(neg_profit, bounds=(lb, ub), method='bounded')
            p_opt = float(res.x)
            prof = float(-res.fun)

            demand_mult = (p_opt / max(ref_t, 1e-6)) ** price_elasticity
            d = base_demand_t * demand_mult
            comp_effect = 1 + 0.3 * (comp_t - p_opt) / max(p_opt, 1e-6)
            d *= comp_effect
            ms_effect = np.clip(ms_base * (1 + 0.1 * (comp_t - p_opt) / max(p_opt, 1e-6)), 0.05, 0.6)

            results.append({
                'day': t+1,
                'forecast_ref_price': ref_t,
                'competitor_price': comp_t,
                'optimal_price': p_opt,
                'price_band_low': lb, 'price_band_high': ub,
                'profit_per_unit': p_opt - current_cost,
                'expected_demand': float(d),
                'expected_market_share': float(ms_effect),
                'total_profit': prof,
                'margin_percentage': (p_opt - current_cost) / p_opt * 100
            })

        df = pd.DataFrame(results)
        self.optimization_results[target_product] = df
        print(f"\nDynamic optimization summary ({horizon} days):")
        print(f"  Avg optimal price: ${df['optimal_price'].mean():.2f}")
        print(f"  Total profit: ${df['total_profit'].sum():,.2f}")
        print(f"  Avg margin: {df['margin_percentage'].mean():.2f}%")
        print(f"  Price band hits: "
              f"{np.mean(np.isclose(df['optimal_price'], df['price_band_low']))*100:.1f}% at floor, "
              f"{np.mean(np.isclose(df['optimal_price'], df['price_band_high']))*100:.1f}% at ceiling.")
        return df

    # -------------------- Reporting & Plots --------------------

    def generate_comprehensive_report(self, target_product='gasoline'):
        print("\n" + "=" * 80)
        print(f"COMPREHENSIVE TIME SERIES ANALYSIS REPORT")
        print(f"Product: {target_product.upper()}")
        print("=" * 80)

        current_price = self.data[f'{target_product}_price'].iloc[-1]
        current_demand = self.data[f'{target_product}_demand'].iloc[-1]
        current_cost = self.data['total_costs'].iloc[-1]
        current_margin = (current_price - current_cost) / current_price * 100

        print(f"\n CURRENT MARKET STATUS")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Current Demand: {current_demand:,.0f} bpd")
        print(f"Current Cost: ${current_cost:.2f}")
        print(f"Current Margin: {current_margin:.1f}%")

        if target_product in self.model_performance:
            print(f"\n MODEL PERFORMANCE SUMMARY")
            performance = self.model_performance[target_product]
            for model, metrics in performance.items():
                print(f"\n{model}:")
                print(f"  MAPE: {metrics['MAPE']:.2f}%")
                print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.1f}%")
            best_model = min(performance.keys(), key=lambda x: performance[x]['MAPE'])
            print(f"\n Best Model: {best_model} (MAPE: {performance[best_model]['MAPE']:.2f}%)")

        forecasts = self.forecast_future(target_product, 30)
        if forecasts:
            print(f"\n 30-DAY FORECAST SUMMARY")
            for model_name, forecast in forecasts.items():
                avg_forecast = np.mean(forecast)
                price_change = (avg_forecast - current_price) / current_price * 100
                print(f"{model_name}: ${avg_forecast:.2f} avg ({price_change:+.2f}%)")

        if target_product in self.sensitivity_results and 'local' in self.sensitivity_results[target_product]:
            print(f"\n LOCAL SENSITIVITY (7-day, ±1%)")
            for r in self.sensitivity_results[target_product]['local']:
                print(f"  {r['factor']:>20s}: {r['elasticity_mean']:+.3f} "
                      f"(95% CI {r['ci_low']:+.3f}..{r['ci_high']:+.3f})")

        if target_product in self.optimization_results:
            opt_results = self.optimization_results[target_product]
            print(f"\n PRICING OPTIMIZATION")
            print(f"Recommended avg price: ${opt_results['optimal_price'].mean():.2f}")
            print(f"Expected total profit ({len(opt_results)} days): ${opt_results['total_profit'].sum():,.2f}")
            print(f"Average recommended margin: {opt_results['margin_percentage'].mean():.1f}%")

        print(f"\nSTRATEGIC RECOMMENDATIONS")
        if forecasts:
            first_model = next(iter(forecasts))
            avg_forecast = np.mean(forecasts[first_model])
            if avg_forecast > current_price * 1.02:
                print("BULLISH: Prices expected to rise - consider inventory buildup")
            elif avg_forecast < current_price * 0.98:
                print("BEARISH: Prices expected to fall - consider price competitiveness")
            else:
                print("STABLE: Prices expected to remain stable - focus on efficiency")

        price_volatility = self.data[f'{target_product}_price'].rolling(30).std().iloc[-1]
        if price_volatility > self.data[f'{target_product}_price'].std():
            print("HIGH VOLATILITY: Consider dynamic pricing and hedging strategies")
        else:
            print("NORMAL VOLATILITY: Standard pricing strategies applicable")

        competitor_col = f'competitor_{target_product}'
        if competitor_col in self.data.columns:
            competitive_position = current_price - self.data[competitor_col].iloc[-1]
            if competitive_position > 2:
                print("PREMIUM PRICING: Higher than competitors - justify with value")
            elif competitive_position < -2:
                print("COMPETITIVE PRICING: Lower than competitors - opportunity for increase")
            else:
                print("MARKET PRICING: Aligned with competitors - monitor closely")

        print("\n" + "=" * 80)

    def plot_comprehensive_analysis(self, target_product='gasoline'):
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Comprehensive Analysis Dashboard - {target_product.title()}', fontsize=16)

        price_col = f'{target_product}_price'
        historical_dates = self.data['date']
        historical_prices = self.data[price_col]
        axes[0, 0].plot(historical_dates, historical_prices, label='Historical', linewidth=2)

        if target_product in self.forecasts:
            future_dates = pd.date_range(
                start=historical_dates.iloc[-1] + pd.Timedelta(days=1),
                periods=30, freq='D'
            )
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            for i, (model_name, forecast) in enumerate(self.forecasts[target_product].items()):
                f = np.array(forecast)
                axes[0, 0].plot(future_dates[:len(f)], f[:30],
                                label=f'{model_name} Forecast',
                                linestyle='--', color=colors[i % len(colors)])
        axes[0, 0].set_title('Price History and Forecasts')
        axes[0, 0].set_xlabel('Date'); axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend(); axes[0, 0].tick_params(axis='x', rotation=45)

        if target_product in self.model_performance:
            performance = self.model_performance[target_product]
            models = list(performance.keys())
            mapes = [performance[model]['MAPE'] for model in models]
            bars = axes[0, 1].bar(models, mapes, alpha=0.7)
            axes[0, 1].set_title('Model Performance (MAPE)')
            axes[0, 1].set_ylabel('MAPE (%)'); axes[0, 1].tick_params(axis='x', rotation=45)
            best_idx = mapes.index(min(mapes)); bars[best_idx].set_color('gold')

        # Sensitivity panel: prefer local clean sensitivities if available
        if target_product in self.sensitivity_results and 'local' in self.sensitivity_results[target_product]:
            sens = pd.DataFrame(self.sensitivity_results[target_product]['local'])
            sens = sens.sort_values(by='elasticity_mean', key=lambda s: np.abs(s), ascending=True)
            axes[0, 2].barh(sens['factor'], sens['elasticity_mean'])
            axes[0, 2].set_title('Price Sensitivity (Elasticity, clean local)')
            axes[0, 2].set_xlabel('Elasticity')

        current_costs = self.data['total_costs'].iloc[-1]
        variable_costs = self.data['variable_costs'].iloc[-1]
        fixed_costs = self.data['fixed_costs'].iloc[-1]
        axes[1, 0].pie([variable_costs, fixed_costs], labels=['Variable Costs', 'Fixed Costs'],
                       autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title(f'Cost Structure (Total: ${current_costs:.2f})')

        axes[1, 1].scatter(self.data[price_col], self.data[f'{target_product}_demand'], alpha=0.6)
        axes[1, 1].set_title('Price vs Demand Relationship'); axes[1, 1].set_xlabel('Price ($)'); axes[1, 1].set_ylabel('Demand (bpd)')
        z = np.polyfit(self.data[price_col], self.data[f'{target_product}_demand'], 1); p = np.poly1d(z)
        axes[1, 1].plot(self.data[price_col], p(self.data[price_col]), "r--", alpha=0.8)

        competitor_col = f'competitor_{target_product}'
        if competitor_col in self.data.columns:
            axes[1, 2].plot(self.data['date'], self.data[price_col], label='Our Price', linewidth=2)
            axes[1, 2].plot(self.data['date'], self.data[competitor_col], label='Competitor Price', linewidth=2, alpha=0.7)
            axes[1, 2].set_title('Competitive Price Comparison')
            axes[1, 2].set_xlabel('Date'); axes[1, 2].set_ylabel('Price ($)')
            axes[1, 2].legend(); axes[1, 2].tick_params(axis='x', rotation=45)

        if target_product in self.optimization_results:
            opt_results = self.optimization_results[target_product]
            axes[2, 0].plot(opt_results['day'], opt_results['forecast_ref_price'], label='Forecasted', marker='o')
            axes[2, 0].plot(opt_results['day'], opt_results['optimal_price'], label='Optimized', marker='s')
            axes[2, 0].set_title('Price Optimization Results'); axes[2, 0].set_xlabel('Days Ahead'); axes[2, 0].set_ylabel('Price ($)')
            axes[2, 0].legend()
            axes[2, 1].bar(opt_results['day'], opt_results['total_profit'])
            axes[2, 1].set_title('Daily Profit Projection'); axes[2, 1].set_xlabel('Days Ahead'); axes[2, 1].set_ylabel('Profit ($)')
            axes[2, 2].plot(opt_results['day'], opt_results['margin_percentage'], marker='o')
            axes[2, 2].set_title('Projected Profit Margins'); axes[2, 2].set_xlabel('Days Ahead'); axes[2, 2].set_ylabel('Margin (%)')
            axes[2, 2].grid(True, alpha=0.3)
        else:
            monthly_data = self.data.groupby(self.data['date'].dt.month)[price_col].mean()
            axes[2, 1].bar(monthly_data.index, monthly_data.values)
            axes[2, 1].set_title('Seasonal Price Patterns'); axes[2, 1].set_xlabel('Month'); axes[2, 1].set_ylabel('Average Price ($)')
            volatility = self.data[price_col].rolling(30).std()
            axes[2, 2].plot(self.data['date'], volatility)
            axes[2, 2].set_title('30-Day Price Volatility'); axes[2, 2].set_xlabel('Date'); axes[2, 2].set_ylabel('Volatility ($)')
            axes[2, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    # -------------------- Workflow Orchestration --------------------

    def run_complete_analysis(self, target_product='gasoline', use_real_data=False,
                              start_date='2020-01-01', train_ratio=0.8):
        print("=" * 80)
        print("ENHANCED TIME SERIES OIL & GAS PRICING ANALYSIS")
        print("=" * 80)

        print("\n1. Loading market data...")
        if use_real_data:
            self.load_real_data(start_date=start_date)
        else:
            self.generate_enhanced_synthetic_data()
        print(f"   Loaded {len(self.data)} data points with {len(self.data.columns)} features")

        print(f"\n2. Training time series models...")
        models, predictions = self.train_models(target_product, train_ratio)

        print(f"\n3. Generating future forecasts...")
        future_forecasts = self.forecast_future(target_product, 30)

        print(f"\n4. Performing clean local sensitivity...")
        _ = self.sensitivity_local(target_product, horizon=7, shock=0.01)

        print(f"\n5. Dynamic pricing optimization with bands...")
        _ = self.optimize_pricing_strategy_dynamic(
            target_product=target_product, cost_scenario='optimistic',
            competitive_scenario='weak', horizon=30, band_pct=0.06
        )

        print(f"\n6. Generating comprehensive report...")
        self.generate_comprehensive_report(target_product)

        print(f"\n7. Creating visualization dashboard...")
        self.plot_comprehensive_analysis(target_product)

        print("\n" + "=" * 80)
        print("COMPLETE ANALYSIS FINISHED!")
        print("=" * 80)

        return {
            'models': models,
            'forecasts': future_forecasts,
            'model_performance': self.model_performance.get(target_product, {}),
            'sensitivity_results': self.sensitivity_results.get(target_product, {}),
            'optimization_results': self.optimization_results.get(target_product, pd.DataFrame()),
            'data': self.data
        }

    # -------------------- Export --------------------

    def export_results(self, target_product='gasoline', filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"oil_gas_analysis_{target_product}_{timestamp}"

        results = {
            'analysis_date': datetime.now().isoformat(),
            'target_product': target_product,
            'data_summary': {
                'total_records': len(self.data),
                'date_range': f"{self.data['date'].min()} to {self.data['date'].max()}",
                'current_price': float(self.data[f'{target_product}_price'].iloc[-1]),
                'current_demand': float(self.data[f'{target_product}_demand'].iloc[-1])
            },
            'model_performance': self.model_performance.get(target_product, {}),
            'sensitivity_results': self.sensitivity_results.get(target_product, {}),
            'forecasts': {}
        }

        if target_product in self.forecasts:
            for model_name, forecast in self.forecasts[target_product].items():
                if hasattr(forecast, 'tolist'):
                    results['forecasts'][model_name] = forecast.tolist()
                else:
                    results['forecasts'][model_name] = list(forecast)

        if target_product in self.optimization_results:
            opt_df = self.optimization_results[target_product]
            if isinstance(opt_df, pd.DataFrame):
                results['optimization_results'] = opt_df.to_dict('records')

        json_filename = f"{filename}.json"
        try:
            with open(json_filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✓ Results exported to {json_filename}")
        except Exception as e:
            print(f"Error exporting JSON: {e}")

        csv_filename = f"{filename}_data.csv"
        try:
            self.data.to_csv(csv_filename, index=False)
            print(f"✓ Data exported to {csv_filename}")
        except Exception as e:
            print(f"Error exporting CSV: {e}")

        return json_filename, csv_filename

# --------------- API models ---------------
class RunRequest(BaseModel):
    target_product: Literal['gasoline','diesel'] = 'gasoline'
    use_real_data: bool = True
    start_date: str = '2022-01-01'
    train_ratio: float = 0.8
    forecast_horizon: int = 30
    profile: Literal['corporate','typical_refinery','regional_refinery'] = 'typical_refinery'

class OptimizeRequest(BaseModel):
    target_product: Literal['gasoline','diesel'] = 'gasoline'
    cost_scenario: Literal['optimistic','base','pessimistic'] = 'base'
    competitive_scenario: Literal['weak','base','aggressive'] = 'base'
    horizon: int = 30
    band_pct: float = 0.06

class NewsRequest(BaseModel):
    hours: int = 168
    max_items: int = 30
    model: str | None = None  # falls back to env/DEFAULT_MODEL

# -------------------- NEWS: sources, filters, helpers --------------------
import re, time, requests as _requests
import feedparser as _feedparser
from bs4 import BeautifulSoup as _BS
from openai import OpenAI as _OpenAI

DEFAULT_USER_AGENT = "oilgas-news-bot/1.0 (+https://example.com)"

SOURCES: dict[str, dict] = {
    "EIA_TodayInEnergy": {"type": "rss", "url": "https://www.eia.gov/tools/rssfeeds/todayinenergy.xml"},
    "EIA_TWIP": {"type": "rss", "url": "https://www.eia.gov/tools/rssfeeds/twip.xml"},
    "EIA_NaturalGasWeekly": {"type": "rss", "url": "https://www.eia.gov/tools/rssfeeds/natural_gas.xml"},
    "IEA_Press": {"type": "rss", "url": "https://www.iea.org/press/pressreleases.rss"},
    "IEA_News": {"type": "html", "url": "https://www.iea.org/news", "selectors": {"item": "a.sc-1t83c9q-0.hKkXUx", "href_attr": "href", "title_attr": None}},
    "OPEC_Press": {"type": "html", "url": "https://www.opec.org/press-releases.html", "selectors": {"item": "a", "href_attr": "href", "title_attr": None}},
    "GoogleNews_OilGas": {"type": "rss", "url": "https://news.google.com/rss/search?q=(OPEC+OR+Brent+OR+WTI+OR+diesel+OR+inventories+OR+refinery+OR+sanctions+OR+LNG+OR+gasoline+OR+distillate+OR+crack+spread)&hl=en-US&gl=US&ceid=US:en"},
}

_KEYWORDS = [
    r"\bOPEC\+?\b", r"\bquota\b", r"\bproduction (?:cut|increase|target)s?\b",
    r"\bexport (?:ban|curb|quota|halt)s?\b", r"\bsanction(s)?\b", r"\bembargo\b",
    r"\binventor(?:y|ies)\b", r"\bstock(?:s)?\b", r"\bdraw(?:down)?\b", r"\bbuildup\b",
    r"\brefiner(?:y|ies)\b", r"\bmaintenance\b", r"\bturnaround\b", r"\boutage\b",
    r"\bdiesel\b", r"\bULSD\b", r"\bgaso?line\b", r"\bjet fuel\b", r"\bdistillate\b",
    r"\btariff(s)?\b", r"\bChina\b", r"\bIndia\b", r"\bGDP\b", r"\bindustrial output\b",
    r"\bRed Sea\b", r"\bStrait of Hormuz\b", r"\bPanama Canal\b", r"\bBlack Sea\b",
    r"\bdemand\b", r"\bconsumption\b", r"\bheatwave\b", r"\bcold snap\b", r"\bpower demand\b",
    r"\bBrent\b", r"\bWTI\b", r"\bprice cap\b", r"\bfutures\b", r"\bspread\b", r"\bcrack\b", r"\bLNG\b",
]
_KEYWORD_RE = re.compile("|".join(_KEYWORDS), flags=re.I)


def _fetch_rss(url: str) -> list[dict]:
    feed = _feedparser.parse(url)
    items = []
    for e in feed.entries:
        title = getattr(e, "title", "") or ""
        link = getattr(e, "link", "") or ""
        summary = getattr(e, "summary", "") or ""
        published = None
        if getattr(e, "published_parsed", None):
            published = datetime.fromtimestamp(time.mktime(e.published_parsed))
        items.append({"title": title.strip(), "url": link, "summary": summary.strip(), "published": published})
    return items


def _absolute_url(base: str, href: str) -> str:
    if not href:
        return base
    if href.startswith("http"):
        return href
    from urllib.parse import urljoin
    return urljoin(base, href)


def _fetch_html_list(url: str, selectors: dict | None) -> list[dict]:
    resp = _requests.get(url, timeout=20, headers={"User-Agent": DEFAULT_USER_AGENT})
    resp.raise_for_status()
    soup = _BS(resp.text, "html.parser")
    items: list[dict] = []
    anchors = soup.select(selectors["item"]) if selectors and selectors.get("item") else soup.find_all("a")
    for a in anchors:
        href = a.get(selectors.get("href_attr", "href"), None) if selectors else a.get("href")
        text = (a.get(selectors.get("title_attr")) if selectors and selectors.get("title_attr") else a.get_text()).strip()
        if not text or not href:
            continue
        link = _absolute_url(url, href)
        if any(domain in link for domain in [".iea.org/", "opec.org"]):
            items.append({"title": text, "url": link, "summary": "", "published": None})
    return items


def _fetch_all(verbose: bool = False) -> list[dict]:
    all_items: list[dict] = []
    for name, cfg in SOURCES.items():
        try:
            items = _fetch_rss(cfg["url"]) if cfg["type"] == "rss" else _fetch_html_list(cfg["url"], cfg.get("selectors", {}))
            for it in items:
                it["source"] = name
            if verbose:
                print(f"[fetch] {name}: {len(items)} items")
            all_items.extend(items)
        except Exception as e:
            print(f"[WARN] {name} fetch failed: {e}")
    return all_items


def _dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        key = (it.get("title", "").lower(), (it.get("url", "").split("?")[0]))
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _within_hours(dt_obj: datetime | None, hours: int) -> bool:
    if dt_obj is None:
        return True
    now = datetime.utcnow()
    delta = now - dt_obj.replace(tzinfo=None)
    return delta.total_seconds() <= hours * 3600


def _clip_time(items: list[dict], hours: int) -> list[dict]:
    return [it for it in items if _within_hours(it.get("published"), hours)]


def _keyword_filter(items: list[dict]) -> list[dict]:
    return [it for it in items if _KEYWORD_RE.search(((it.get("title") or "") + " " + (it.get("summary") or "")).strip())]



def _summarize_with_openai(items: list[dict], model: str | None) -> str:
    """Return Markdown. If OPENAI_API_KEY missing or any error occurs, return a simple headline list."""
    try:
        lines = [f"- [{it.get('source')}] {it.get('title')}  {it.get('url')}" for it in (items or [])]
        key = os.getenv("OPENAI_API_KEY") or ""
        base_url = os.getenv("OPENAI_BASE_URL") or None
        if not key:
            return "\n".join(lines)

        try:
            # Prefer Responses API; if not available, fall back to Chat Completions
            from openai import OpenAI as _OpenAI
            client = _OpenAI(api_key=key, base_url=base_url)
            today = datetime.utcnow().strftime("%Y-%m-%d")
            sources_snippet = [{"source": it.get("source"), "title": it.get("title"), "url": it.get("url")} for it in (items or [])[:80]]
            prompt = f"""
You are an energy markets analyst. Today is {today}.
You will receive a JSON list of headlines/links that affect oil & gas prices, demand, and supply.
1) Group items into Supply / Demand / Prices & Margins / Geopolitics-Logistics.
2) Under each, give crisp bullet points (≤12 words each).
3) Then write "Implications" with 3–5 bullets (bullish/bearish + why).
4) End with a one-line summary.
Return clean Markdown.
JSON:
{json.dumps(sources_snippet, ensure_ascii=False)}
"""
            try:
                resp = client.responses.create(
                    model=model or os.getenv("DEFAULT_MODEL", "gpt-4o"),
                    input=[
                        {"role": "system", "content": "Be accurate, non-hypey, and concise. Cite sources inline as [SourceName]."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                out = ""
                for item in getattr(resp, "output", []) or []:
                    if getattr(item, "type", "") == "message":
                        for c in getattr(item, "content", []) or []:
                            if getattr(c, "type", "") == "output_text":
                                out += c.text
                return out.strip() or "\n".join(lines)
            except Exception:
                # Fallback to chat.completions if Responses API isn't available
                chat = client.chat.completions.create(
                    model=model or os.getenv("DEFAULT_MODEL", "gpt-4o"),
                    messages=[
                        {"role": "system", "content": "Be accurate, non-hypey, and concise. Cite sources inline as [SourceName]."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                return (chat.choices[0].message.content or "").strip() or "\n".join(lines)
        except Exception:
            return "\n".join(lines)
    except Exception:
        # Absolute last resort
        return ""


# Global instance (lightweight until run)
system: Optional[TimeSeriesOilGasPricingSystem] = None

app = FastAPI(title="Oil & Gas Pricing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}

@app.post("/api/run-analysis")
def run_analysis(req: RunRequest):
    global system
    system = TimeSeriesOilGasPricingSystem(forecast_horizon=req.forecast_horizon, profile=req.profile)
    results = system.run_complete_analysis(
        target_product=req.target_product,
        use_real_data=req.use_real_data,
        start_date=req.start_date,
        train_ratio=req.train_ratio,
    )

    # Reduce payload sizes where possible
    df = results['data']
    last_rows = df.tail(365)  # cap
    data_preview = last_rows.to_dict(orient='records')

    return {
        "kpis": {
            "current_price": float(df[f"{req.target_product}_price"].iloc[-1]),
            "current_demand": float(df[f"{req.target_product}_demand"].iloc[-1]),
            "current_cost": float(df['total_costs'].iloc[-1])
        },
        "model_performance": results['model_performance'],
        "forecasts": {k: (v.tolist() if hasattr(v, 'tolist') else list(v)) for k,v in results['forecasts'].items()},
        "sensitivity": results['sensitivity_results'].get('local', []),
        "optimization": results['optimization_results'].to_dict('records') if hasattr(results['optimization_results'], 'to_dict') else [],
        "data": data_preview
    }

@app.post("/api/optimize")
def api_optimize(req: OptimizeRequest):
    global system
    if system is None:
        raise HTTPException(400, "Run analysis first")
    df = system.optimize_pricing_strategy_dynamic(
        target_product=req.target_product,
        cost_scenario=req.cost_scenario,
        competitive_scenario=req.competitive_scenario,
        horizon=req.horizon,
        band_pct=req.band_pct
    )
    return {"optimization": df.to_dict('records')}

@app.post("/api/news")
def api_news(req: NewsRequest):
    try:
        items = _fetch_all(verbose=False)
        items = _dedupe(items)
        items = _keyword_filter(items)
        items = _clip_time(items, req.hours)

        # Prioritize first-party sources, then Google News
        priority = {
            "EIA_TodayInEnergy": 0,
            "EIA_TWIP": 1,
            "EIA_NaturalGasWeekly": 2,
            "IEA_Press": 3,
            "IEA_News": 4,
            "OPEC_Press": 5,
            "GoogleNews_OilGas": 6,
        }
        items.sort(key=lambda it: priority.get(it.get("source"), 9))
        top = items[: req.max_items]

        summary_md = _summarize_with_openai(top, model=req.model)
        return {"summary_md": summary_md, "items": top}
    except Exception as e:
        raise HTTPException(500, f"news failed: {e}")
