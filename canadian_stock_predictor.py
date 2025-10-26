#!/usr/bin/env python3
"""
Canadian Stock Day Trading Predictor
====================================

A machine learning algorithm to predict top-performing Canadian stocks for day trading.
Analyzes price history and technical indicators to suggest stocks likely to perform
excellently the following day.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import talib
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

class CanadianStockPredictor:
    """
    Main class for predicting top-performing Canadian stocks for day trading.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = []
        self.tickers = []
        
    def get_top_canadian_stocks(self, limit=50):
        """
        Scrape top performing Canadian stocks from Barchart.
        Falls back to a predefined list of major Canadian stocks if scraping fails.
        """
        logger.info("Fetching top Canadian stocks...")
        
        # Predefined list of major Canadian stocks as fallback
        major_canadian_stocks = [
            'SHOP.TO', 'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'NA.TO',
            'CNR.TO', 'CP.TO', 'ENB.TO', 'TRP.TO', 'SU.TO', 'CNQ.TO', 'IMO.TO',
            'MFC.TO', 'SLF.TO', 'POW.TO', 'GWO.TO', 'IAG.TO', 'WCN.TO',
            'ATD.TO', 'L.TO', 'CTC.A.TO', 'MRU.TO', 'LULU', 'CSU.TO',
            'WSP.TO', 'STN.TO', 'AC.TO', 'WJA.TO', 'CHR.TO', 'QSR.TO',
            'TFII.TO', 'CCO.TO', 'NTR.TO', 'AGU.TO', 'POT.TO', 'WEED.TO',
            'ACB.TO', 'HEXO.TO', 'APHA.TO', 'CRON.TO', 'OGI.TO', 'TLRY.TO'
        ]
        
        try:
            # Try to scrape from Barchart
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            url = "https://www.barchart.com/ca/stocks/top-100-stocks?viewName=main&orderBy=percentChange&orderDir=desc"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Look for stock symbols in the page
                # This is a simplified approach - in practice, you'd need to inspect the actual HTML structure
                stock_elements = soup.find_all(['a', 'span'], class_=lambda x: x and 'symbol' in x.lower())
                
                scraped_stocks = []
                for element in stock_elements:
                    text = element.get_text().strip()
                    if '.TO' in text or len(text) <= 5:  # Canadian stocks typically end with .TO
                        scraped_stocks.append(text)
                
                if scraped_stocks:
                    self.tickers = scraped_stocks[:limit]
                    logger.info(f"Successfully scraped {len(self.tickers)} stocks from Barchart")
                else:
                    raise Exception("No stocks found in scraped data")
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Failed to scrape Barchart: {e}")
            logger.info("Using predefined list of major Canadian stocks")
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
    
    def calculate_technical_indicators(self, data):
        """
        Calculate various technical indicators for the stock data.
        """
        if data is None or len(data) < 50:
            return None
            
        df = data.copy()
        
        # Price-based indicators
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
        
        df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
        
        # Momentum indicators
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
        
        # Volatility indicators
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'])
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Volume indicators
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        # Price patterns
        df['DOJI'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['HAMMER'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        df['ENGULFING'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        
        # Additional custom indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving average crossovers
        df['SMA_20_50_Cross'] = np.where(df['SMA_20'] > df['SMA_50'], 1, 0)
        df['Price_SMA_20_Above'] = np.where(df['Close'] > df['SMA_20'], 1, 0)
        df['Price_SMA_50_Above'] = np.where(df['Close'] > df['SMA_50'], 1, 0)
        
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
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'ATR', 'OBV',
            'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
            'BB_upper', 'BB_middle', 'BB_lower', 'Price_Change',
            'Volume_Change', 'High_Low_Pct', 'Close_Open_Pct',
            'SMA_20_50_Cross', 'Price_SMA_20_Above', 'Price_SMA_50_Above'
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
        
        # Volume features
        if len(df) >= 20:
            avg_volume = df['Volume'].tail(20).mean()
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            features.append(volume_ratio)
        else:
            features.append(1)
        
        return np.array(features)
    
    def prepare_training_data(self):
        """
        Prepare training data from all available stocks.
        """
        logger.info("Preparing training data...")
        
        X = []
        y = []
        
        for ticker in self.tickers:
            logger.info(f"Processing {ticker}...")
            
            # Get stock data
            data = self.get_stock_data(ticker)
            if data is None:
                continue
                
            # Calculate technical indicators
            df = self.calculate_technical_indicators(data)
            if df is None:
                continue
            
            # Create features for each day (except the last 5 days for validation)
            for i in range(50, len(df) - 5):
                # Create features for day i
                features = self.create_features(df.iloc[:i+1])
                if features is None:
                    continue
                
                # Calculate target: did the stock perform well in the next 1-5 days?
                future_prices = df['Close'].iloc[i+1:i+6]
                if len(future_prices) > 0:
                    max_future_return = (future_prices.max() - df['Close'].iloc[i]) / df['Close'].iloc[i]
                    # Consider "excellent performance" as >5% gain
                    target = 1 if max_future_return > 0.05 else 0
                    
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
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
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
            df = self.calculate_technical_indicators(data)
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
        logger.info("Starting Canadian Stock Prediction Pipeline...")
        
        # Step 1: Get top Canadian stocks
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
    print("=" * 60)
    print("Canadian Stock Day Trading Predictor")
    print("=" * 60)
    
    # Create predictor instance
    predictor = CanadianStockPredictor()
    
    # Run prediction pipeline
    top_stocks = predictor.run_prediction_pipeline(top_n=10)
    
    if top_stocks:
        print("\n" + "=" * 60)
        print("TOP PREDICTED STOCKS FOR DAY TRADING")
        print("=" * 60)
        print(f"{'Rank':<5} {'Ticker':<10} {'Probability':<12} {'Current Price':<15} {'Recent Change':<15}")
        print("-" * 60)
        
        for i, stock in enumerate(top_stocks, 1):
            print(f"{i:<5} {stock['ticker']:<10} {stock['probability']:.4f}      ${stock['current_price']:.2f}        {stock['recent_change']:+.2f}%")
        
        print("\n" + "=" * 60)
        print("DISCLAIMER: This is for educational purposes only.")
        print("Always do your own research before making investment decisions.")
        print("Past performance does not guarantee future results.")
        print("=" * 60)
    else:
        print("No predictions available. Please check your data sources.")

if __name__ == "__main__":
    main()