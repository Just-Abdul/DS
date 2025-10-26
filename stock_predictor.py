"""
Canadian Stock Day Trading ML Predictor
=====================================

This module provides machine learning capabilities to predict top-performing 
Canadian stocks for day trading based on technical analysis and historical data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

import ta
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import VolumeSMAIndicator

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime, timedelta
import time
import joblib
import os


class CanadianStockPredictor:
    """
    A machine learning system for predicting top-performing Canadian stocks
    suitable for day trading based on technical analysis.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.top_canadian_stocks = []
        
    def get_top_canadian_stocks(self, limit=100):
        """
        Scrape top Canadian stocks from various sources and return a list of symbols.
        """
        # Common Canadian stocks from TSX
        tsx_stocks = [
            'SHOP.TO', 'CNR.TO', 'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',
            'ENB.TO', 'TRP.TO', 'SU.TO', 'CNQ.TO', 'IMO.TO', 'CVE.TO', 'ARX.TO',
            'WCP.TO', 'MEG.TO', 'BTE.TO', 'CPG.TO', 'VET.TO', 'WFG.TO',
            'ABX.TO', 'K.TO', 'NEM.TO', 'FNV.TO', 'AEM.TO', 'KL.TO', 'ELD.TO',
            'NGT.TO', 'CG.TO', 'FSZ.TO', 'SMU.TO', 'MG.TO', 'TKO.TO',
            'WEED.TO', 'ACB.TO', 'HEXO.TO', 'OGI.TO', 'FIRE.TO', 'ZENA.TO',
            'BB.TO', 'NTAR.TO', 'WELL.TO', 'DOC.TO', 'GDNP.TO', 'PKK.TO',
            'CSU.TO', 'ATD.TO', 'L.TO', 'MFC.TO', 'SLF.TO', 'GWO.TO',
            'BAM.TO', 'BPY.UN.TO', 'REI.UN.TO', 'HR.UN.TO', 'CT.TO'
        ]
        
        # Add mining and resource stocks
        mining_stocks = [
            'TOU.TO', 'POU.TO', 'KEL.TO', 'BIR.TO', 'NVA.TO', 'GXE.TO',
            'TVE.TO', 'SGY.TO', 'ERF.TO', 'CPX.TO', 'OBE.TO', 'PNE.TO',
            'YGR.TO', 'BXE.TO', 'CJ.TO', 'GTE.TO', 'HWX.TO', 'IPO.TO'
        ]
        
        # Technology and growth stocks
        tech_stocks = [
            'LSPD.TO', 'NVEI.TO', 'DCBO.TO', 'TOI.TO', 'QTRH.TO', 'MTLO.TO',
            'CTS.TO', 'ATS.TO', 'CGX.TO', 'GIB.A.TO', 'WSP.TO', 'STC.TO'
        ]
        
        all_stocks = tsx_stocks + mining_stocks + tech_stocks
        self.top_canadian_stocks = list(set(all_stocks))[:limit]
        
        print(f"Loaded {len(self.top_canadian_stocks)} Canadian stocks for analysis")
        return self.top_canadian_stocks
    
    def fetch_stock_data(self, symbol, period='1y'):
        """
        Fetch historical stock data for a given symbol.
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                print(f"No data found for {symbol}")
                return None
                
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """
        Calculate comprehensive technical indicators for the stock data.
        """
        if data is None or data.empty:
            return None
            
        df = data.copy()
        
        # Clean data
        df = dropna(df)
        
        # Price-based indicators
        df['SMA_5'] = SMAIndicator(close=df['Close'], window=5).sma_indicator()
        df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        
        df['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
        df['EMA_26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()
        
        # RSI
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Volume indicators
        df['Volume_SMA'] = VolumeSMAIndicator(close=df['Close'], volume=df['Volume'], window=20).volume_sma()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum and volatility
        df['Price_change'] = df['Close'].pct_change()
        df['Price_change_5d'] = df['Close'].pct_change(periods=5)
        df['Volatility_20d'] = df['Price_change'].rolling(window=20).std()
        
        # High-Low spread
        df['HL_spread'] = (df['High'] - df['Low']) / df['Close']
        df['HL_spread_ma'] = df['HL_spread'].rolling(window=10).mean()
        
        # Gap analysis
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Trend strength
        df['Trend_strength'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        
        # Support and resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support_distance'] = (df['Close'] - df['Support']) / df['Close']
        df['Resistance_distance'] = (df['Resistance'] - df['Close']) / df['Close']
        
        return df
    
    def create_features(self, data):
        """
        Create feature matrix for machine learning model.
        """
        if data is None or data.empty:
            return None, None
            
        # Calculate next day return as target
        data['Next_day_return'] = data['Close'].shift(-1) / data['Close'] - 1
        
        # Feature columns
        feature_cols = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'MACD_histogram', 'RSI',
            'BB_width', 'BB_position', 'Stoch_K', 'Stoch_D',
            'Volume_ratio', 'Price_change', 'Price_change_5d', 'Volatility_20d',
            'HL_spread', 'HL_spread_ma', 'Gap', 'Trend_strength',
            'Support_distance', 'Resistance_distance'
        ]
        
        # Additional derived features
        data['RSI_oversold'] = (data['RSI'] < 30).astype(int)
        data['RSI_overbought'] = (data['RSI'] > 70).astype(int)
        data['MACD_bullish'] = (data['MACD'] > data['MACD_signal']).astype(int)
        data['Above_SMA20'] = (data['Close'] > data['SMA_20']).astype(int)
        data['High_volume'] = (data['Volume_ratio'] > 1.5).astype(int)
        
        feature_cols.extend(['RSI_oversold', 'RSI_overbought', 'MACD_bullish', 
                           'Above_SMA20', 'High_volume'])
        
        # Remove rows with NaN values
        data_clean = data.dropna()
        
        if data_clean.empty:
            return None, None
            
        X = data_clean[feature_cols]
        y = data_clean['Next_day_return']
        
        return X, y
    
    def prepare_training_data(self, symbols_list=None, lookback_period='2y'):
        """
        Prepare training data from multiple stocks.
        """
        if symbols_list is None:
            symbols_list = self.get_top_canadian_stocks()
        
        all_features = []
        all_targets = []
        
        print(f"Preparing training data for {len(symbols_list)} stocks...")
        
        for i, symbol in enumerate(symbols_list):
            if i % 10 == 0:
                print(f"Processing {i+1}/{len(symbols_list)}: {symbol}")
            
            # Fetch and process data
            stock_data = self.fetch_stock_data(symbol, period=lookback_period)
            if stock_data is None:
                continue
                
            tech_data = self.calculate_technical_indicators(stock_data)
            if tech_data is None:
                continue
                
            X, y = self.create_features(tech_data)
            if X is None or y is None:
                continue
            
            # Add stock identifier
            X['Symbol'] = symbol
            
            all_features.append(X)
            all_targets.append(y)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        if not all_features:
            raise ValueError("No valid data found for any stocks")
        
        # Combine all data
        X_combined = pd.concat(all_features, ignore_index=True)
        y_combined = pd.concat(all_targets, ignore_index=True)
        
        # Remove symbol column for training (but keep for reference)
        symbols = X_combined['Symbol']
        X_combined = X_combined.drop('Symbol', axis=1)
        
        self.feature_columns = X_combined.columns.tolist()
        
        print(f"Training data prepared: {len(X_combined)} samples, {len(X_combined.columns)} features")
        
        return X_combined, y_combined, symbols
    
    def train_models(self, X, y):
        """
        Train multiple ML models for ensemble prediction.
        """
        print("Training machine learning models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=6, random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbose=-1
            )
        }
        
        # Train and evaluate models
        model_scores = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Use scaled data for some models
            if name in ['RandomForest', 'GradientBoosting']:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_scores[name] = {'MSE': mse, 'R2': r2, 'model': model}
            
            print(f"{name} - MSE: {mse:.6f}, R2: {r2:.4f}")
        
        # Store best models
        self.models = {name: info['model'] for name, info in model_scores.items()}
        
        # Feature importance (using RandomForest)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.models['RandomForest'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return model_scores, feature_importance
    
    def predict_stock_performance(self, symbol, days_ahead=1):
        """
        Predict stock performance for the next day(s).
        """
        # Fetch recent data
        stock_data = self.fetch_stock_data(symbol, period='6mo')
        if stock_data is None:
            return None
        
        # Calculate technical indicators
        tech_data = self.calculate_technical_indicators(stock_data)
        if tech_data is None:
            return None
        
        # Create features for the latest data point
        X, _ = self.create_features(tech_data)
        if X is None:
            return None
        
        # Get the most recent features
        latest_features = X.iloc[-1:][self.feature_columns]
        
        # Make predictions with all models
        predictions = {}
        
        # Scale features for models that need it
        latest_scaled = self.scaler.transform(latest_features)
        
        for name, model in self.models.items():
            if name in ['RandomForest', 'GradientBoosting']:
                pred = model.predict(latest_features)[0]
            else:
                pred = model.predict(latest_scaled)[0]
            
            predictions[name] = pred
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean(list(predictions.values()))
        
        # Get current price and calculate predicted price
        current_price = stock_data['Close'].iloc[-1]
        predicted_return = ensemble_pred
        predicted_price = current_price * (1 + predicted_return)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_return': predicted_return,
            'predicted_price': predicted_price,
            'predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'confidence_score': 1 - np.std(list(predictions.values()))  # Lower std = higher confidence
        }
    
    def get_top_predictions(self, n_top=10, min_return_threshold=0.02):
        """
        Get top N stock predictions for tomorrow's trading.
        """
        print(f"Analyzing stocks for top {n_top} predictions...")
        
        predictions = []
        
        for i, symbol in enumerate(self.top_canadian_stocks):
            if i % 20 == 0:
                print(f"Analyzing {i+1}/{len(self.top_canadian_stocks)}: {symbol}")
            
            pred = self.predict_stock_performance(symbol)
            if pred is not None:
                predictions.append(pred)
            
            time.sleep(0.1)  # Rate limiting
        
        # Filter and sort predictions
        valid_predictions = [
            p for p in predictions 
            if p['predicted_return'] > min_return_threshold and p['confidence_score'] > 0.3
        ]
        
        # Sort by predicted return
        top_predictions = sorted(
            valid_predictions, 
            key=lambda x: x['predicted_return'], 
            reverse=True
        )[:n_top]
        
        return top_predictions
    
    def save_model(self, filepath='canadian_stock_predictor.joblib'):
        """
        Save the trained model and scaler.
        """
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'top_stocks': self.top_canadian_stocks
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='canadian_stock_predictor.joblib'):
        """
        Load a previously trained model.
        """
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.top_canadian_stocks = model_data['top_stocks']
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"Model file {filepath} not found")
            return False
    
    def plot_predictions(self, predictions):
        """
        Create visualization of top predictions.
        """
        if not predictions:
            print("No predictions to plot")
            return
        
        # Create DataFrame for plotting
        df_plot = pd.DataFrame(predictions)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Predicted Returns', 'Confidence Scores', 
                          'Current vs Predicted Price', 'Model Agreement'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Plot 1: Predicted Returns
        fig.add_trace(
            go.Bar(x=df_plot['symbol'], y=df_plot['predicted_return'],
                   name='Predicted Return', marker_color='green'),
            row=1, col=1
        )
        
        # Plot 2: Confidence Scores
        fig.add_trace(
            go.Bar(x=df_plot['symbol'], y=df_plot['confidence_score'],
                   name='Confidence', marker_color='blue'),
            row=1, col=2
        )
        
        # Plot 3: Current vs Predicted Price
        fig.add_trace(
            go.Scatter(x=df_plot['symbol'], y=df_plot['current_price'],
                      mode='markers', name='Current Price', marker_color='red'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_plot['symbol'], y=df_plot['predicted_price'],
                      mode='markers', name='Predicted Price', marker_color='green'),
            row=2, col=1, secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Canadian Stock Trading Predictions",
            height=800,
            showlegend=True
        )
        
        fig.show()
        
        return fig


def main():
    """
    Main execution function for the Canadian Stock Predictor.
    """
    print("=== Canadian Stock Day Trading ML Predictor ===")
    print("Initializing predictor...")
    
    predictor = CanadianStockPredictor()
    
    # Try to load existing model
    if predictor.load_model():
        print("Using existing trained model...")
    else:
        print("Training new model...")
        
        # Get Canadian stocks
        stocks = predictor.get_top_canadian_stocks(limit=80)
        
        # Prepare training data
        X, y, symbols = predictor.prepare_training_data(stocks)
        
        # Train models
        model_scores, feature_importance = predictor.train_models(X, y)
        
        # Save model
        predictor.save_model()
    
    # Get predictions for tomorrow
    print("\nGenerating predictions for tomorrow's trading...")
    top_predictions = predictor.get_top_predictions(n_top=15, min_return_threshold=0.01)
    
    # Display results
    print(f"\n=== TOP {len(top_predictions)} STOCK PREDICTIONS FOR TOMORROW ===")
    print("-" * 80)
    
    for i, pred in enumerate(top_predictions, 1):
        print(f"{i:2d}. {pred['symbol']:10s} | "
              f"Current: ${pred['current_price']:6.2f} | "
              f"Predicted: ${pred['predicted_price']:6.2f} | "
              f"Return: {pred['predicted_return']:6.2%} | "
              f"Confidence: {pred['confidence_score']:4.2f}")
    
    # Create visualization
    if top_predictions:
        print("\nGenerating visualization...")
        predictor.plot_predictions(top_predictions[:10])
    
    return predictor, top_predictions


if __name__ == "__main__":
    predictor, predictions = main()