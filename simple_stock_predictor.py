#!/usr/bin/env python3
"""
Simplified Canadian Stock Day Trading Predictor
===============================================

A simplified version that doesn't require TA-Lib installation.
Uses basic technical indicators calculated with pandas and numpy.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class SimpleStockPredictor:
    """
    Simplified version of the Canadian stock predictor.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.tickers = []
        
    def get_top_canadian_stocks(self, limit=30):
        """
        Get a list of major Canadian stocks for analysis.
        """
        logger.info("Loading Canadian stock tickers...")
        
        # Major Canadian stocks
        major_canadian_stocks = [
            'SHOP.TO', 'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'NA.TO',
            'CNR.TO', 'CP.TO', 'ENB.TO', 'TRP.TO', 'SU.TO', 'CNQ.TO', 'IMO.TO',
            'MFC.TO', 'SLF.TO', 'POW.TO', 'GWO.TO', 'IAG.TO', 'WCN.TO',
            'ATD.TO', 'L.TO', 'CTC.A.TO', 'MRU.TO', 'LULU', 'CSU.TO',
            'WSP.TO', 'STN.TO', 'AC.TO', 'WJA.TO', 'CHR.TO', 'QSR.TO',
            'TFII.TO', 'CCO.TO', 'NTR.TO', 'AGU.TO', 'POT.TO', 'WEED.TO',
            'ACB.TO', 'HEXO.TO', 'APHA.TO', 'CRON.TO', 'OGI.TO', 'TLRY.TO',
            'BB.TO', 'DOO.TO', 'MAG.TO', 'K.TO', 'FNV.TO', 'ABX.TO'
        ]
        
        self.tickers = major_canadian_stocks[:limit]
        logger.info(f"Using {len(self.tickers)} Canadian stocks for analysis")
        return self.tickers
    
    def get_stock_data(self, ticker, period='2y'):
        """
        Fetch historical stock data using yfinance.
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                logger.warning(f"No data available for {ticker}")
                return None
                
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def calculate_simple_indicators(self, data):
        """
        Calculate simple technical indicators using pandas.
        """
        if data is None or len(data) < 50:
            return None
            
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Price and Volume changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving average crossovers
        df['SMA_20_50_Cross'] = np.where(df['SMA_20'] > df['SMA_50'], 1, 0)
        df['Price_SMA_20_Above'] = np.where(df['Close'] > df['SMA_20'], 1, 0)
        df['Price_SMA_50_Above'] = np.where(df['Close'] > df['SMA_50'], 1, 0)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility (ATR approximation)
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
        df['True_Range'] = np.maximum(df['High_Low'], np.maximum(df['High_Close'], df['Low_Close']))
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
        return df
    
    def create_features(self, df):
        """
        Create features for machine learning model.
        """
        if df is None or len(df) < 50:
            return None
            
        features = []
        
        # Technical indicator features
        tech_indicators = [
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'ATR',
            'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
            'BB_upper', 'BB_middle', 'BB_lower', 'Price_Change',
            'Volume_Change', 'High_Low_Pct', 'Close_Open_Pct',
            'SMA_20_50_Cross', 'Price_SMA_20_Above', 'Price_SMA_50_Above',
            'Volume_Ratio'
        ]
        
        # Get the latest values for each indicator
        latest_data = df.iloc[-1]
        for indicator in tech_indicators:
            if indicator in df.columns and not pd.isna(latest_data[indicator]):
                features.append(latest_data[indicator])
            else:
                features.append(0)
        
        # Historical performance features
        for days in [1, 3, 5, 10, 20]:
            if len(df) > days:
                pct_change = (df['Close'].iloc[-1] - df['Close'].iloc[-days-1]) / df['Close'].iloc[-days-1]
                features.append(pct_change)
            else:
                features.append(0)
        
        # Additional momentum features
        if len(df) >= 10:
            # 10-day momentum
            momentum_10 = (df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10]
            features.append(momentum_10)
        else:
            features.append(0)
        
        # Price position in Bollinger Bands
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            bb_position = (df['Close'].iloc[-1] - df['BB_lower'].iloc[-1]) / (df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1])
            features.append(bb_position)
        else:
            features.append(0.5)
        
        return np.array(features)
    
    def prepare_training_data(self):
        """
        Prepare training data from all available stocks.
        """
        logger.info("Preparing training data...")
        
        X = []
        y = []
        
        for i, ticker in enumerate(self.tickers):
            logger.info(f"Processing {ticker} ({i+1}/{len(self.tickers)})...")
            
            # Get stock data
            data = self.get_stock_data(ticker)
            if data is None:
                continue
                
            # Calculate technical indicators
            df = self.calculate_simple_indicators(data)
            if df is None:
                continue
            
            # Create features for each day (except the last 5 days for validation)
            for j in range(50, len(df) - 5):
                # Create features for day j
                features = self.create_features(df.iloc[:j+1])
                if features is None:
                    continue
                
                # Calculate target: did the stock perform well in the next 1-5 days?
                future_prices = df['Close'].iloc[j+1:j+6]
                if len(future_prices) > 0:
                    max_future_return = (future_prices.max() - df['Close'].iloc[j]) / df['Close'].iloc[j]
                    # Consider "excellent performance" as >3% gain (more realistic for day trading)
                    target = 1 if max_future_return > 0.03 else 0
                    
                    X.append(features)
                    y.append(target)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared {len(X)} training samples")
        return X, y
    
    def train_model(self, X, y):
        """
        Train the machine learning model.
        """
        logger.info("Training machine learning model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and choose the best one
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3)
            avg_score = cv_scores.mean()
            
            logger.info(f"{name} CV Score: {avg_score:.4f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # Train the best model
        best_model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Best model accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        self.model = best_model
        return best_model
    
    def predict_top_stocks(self, top_n=10):
        """
        Predict top performing stocks for the next day.
        """
        if self.model is None:
            logger.error("Model not trained yet!")
            return []
        
        logger.info("Predicting top performing stocks...")
        
        predictions = []
        
        for ticker in self.tickers:
            # Get recent data
            data = self.get_stock_data(ticker, period='6mo')
            if data is None:
                continue
                
            # Calculate technical indicators
            df = self.calculate_simple_indicators(data)
            if df is None:
                continue
            
            # Create features
            features = self.create_features(df)
            if features is None:
                continue
            
            # Make prediction
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            # Get current price and recent performance
            current_price = df['Close'].iloc[-1]
            recent_change = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100
            
            predictions.append({
                'ticker': ticker,
                'prediction': prediction,
                'probability': probability,
                'current_price': current_price,
                'recent_change': recent_change
            })
        
        # Sort by probability and return top N
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return predictions[:top_n]
    
    def run_prediction_pipeline(self, top_n=10):
        """
        Run the complete prediction pipeline.
        """
        logger.info("Starting Simplified Canadian Stock Prediction Pipeline...")
        
        # Step 1: Get Canadian stocks
        self.get_top_canadian_stocks()
        
        # Step 2: Prepare training data
        X, y = self.prepare_training_data()
        
        if len(X) == 0:
            logger.error("No training data available!")
            return []
        
        # Step 3: Train model
        self.train_model(X, y)
        
        # Step 4: Make predictions
        top_stocks = self.predict_top_stocks(top_n)
        
        return top_stocks

def main():
    """
    Main function to run the stock prediction system.
    """
    print("=" * 70)
    print("Simplified Canadian Stock Day Trading Predictor")
    print("=" * 70)
    print("This version uses basic technical indicators and doesn't require TA-Lib")
    print("=" * 70)
    
    # Create predictor instance
    predictor = SimpleStockPredictor()
    
    # Run prediction pipeline
    top_stocks = predictor.run_prediction_pipeline(top_n=10)
    
    if top_stocks:
        print("\n" + "=" * 70)
        print("TOP PREDICTED STOCKS FOR DAY TRADING")
        print("=" * 70)
        print(f"{'Rank':<5} {'Ticker':<12} {'Probability':<12} {'Current Price':<15} {'Recent Change':<15}")
        print("-" * 70)
        
        for i, stock in enumerate(top_stocks, 1):
            print(f"{i:<5} {stock['ticker']:<12} {stock['probability']:.4f}      ${stock['current_price']:.2f}        {stock['recent_change']:+.2f}%")
        
        print("\n" + "=" * 70)
        print("DISCLAIMER: This is for educational purposes only.")
        print("Always do your own research before making investment decisions.")
        print("Past performance does not guarantee future results.")
        print("Trading involves risk and you may lose money.")
        print("=" * 70)
    else:
        print("No predictions available. Please check your data sources.")

if __name__ == "__main__":
    main()