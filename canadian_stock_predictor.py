#!/usr/bin/env python3
"""
Canadian Stock Day Trading Predictor
=====================================
This script analyzes top-performing Canadian stocks and predicts which stocks
are likely to perform well the next trading day using machine learning.

Features:
- Fetches historical data for Canadian stocks
- Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Trains ML model on historical patterns of top performers
- Predicts next-day performers and provides trading recommendations
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Technical Indicators
import ta


class CanadianStockPredictor:
    """Main class for predicting top-performing Canadian stocks."""
    
    def __init__(self, lookback_days=90, prediction_threshold=5.0):
        """
        Initialize the predictor.
        
        Args:
            lookback_days: Number of days to look back for historical data
            prediction_threshold: % gain threshold to classify as "top performer"
        """
        self.lookback_days = lookback_days
        self.prediction_threshold = prediction_threshold
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def get_canadian_stock_list(self):
        """
        Get a list of popular Canadian stocks to analyze.
        You can expand this list or fetch from an API.
        """
        # Major Canadian stocks across various sectors
        canadian_stocks = [
            # Major Banks
            'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',
            # Energy
            'ENB.TO', 'CNQ.TO', 'SU.TO', 'TRP.TO', 'IMO.TO', 'CVE.TO',
            # Mining & Materials
            'ABX.TO', 'SHOP.TO', 'NTR.TO', 'FM.TO', 'K.TO', 'CCO.TO',
            # Tech
            'SHOP.TO', 'BB.TO', 'LSPD.TO', 'DOO.TO',
            # Telecom
            'T.TO', 'RCI-B.TO', 'BCE.TO',
            # Retail & Consumer
            'L.TO', 'ATD.TO', 'MG.TO', 'DOL.TO',
            # Utilities
            'FTS.TO', 'EMA.TO', 'AQN.TO', 'H.TO',
            # Transportation
            'CP.TO', 'CNR.TO', 'AC.TO',
            # REITs
            'REI-UN.TO', 'AP-UN.TO', 'GRT-UN.TO',
            # Insurance
            'MFC.TO', 'SLF.TO', 'IFC.TO',
            # Healthcare/Cannabis
            'WEED.TO', 'ACB.TO', 'TLRY.TO',
            # Junior Mining (volatile but popular for day trading)
            'GQC.VN', 'MKA.VN', 'NURS.VN', 'QNC.VN', 'LEM.VN',
            'NCX.VN', 'VST.CN', 'ONE.VN', 'PPX.VN', 'SLG.VN',
            'TUNG.CN', 'PHOS.CN', 'URM.CN', 'VMI.VN', 'BOLT.CN',
            'NMI.VN', 'WERX.CN', 'NEXU.CN', 'PLUG.CN', 'METX.CN',
            'BPAG.VN', 'DPF.VN', 'BAR.CN', 'CDN.CN', 'MERG.VN',
            'SCPE.CN', 'CUAU.CN', 'RARE.VN', 'NBRK.CN', 'SPAI.CN',
            'HUT.TO', 'ARA.TO', 'TMQ.TO', 'ICS.CN', 'HG.CN',
            'GURU.TO', 'AUMB.VN', 'FOR.VN', 'NDM.TO', 'ISTK.CN',
            'ADYA.VN', 'MLPN.VN', 'MU.NE', 'IVS.VN', 'SFTB.NE',
            'ELVA.TO', 'CLS.TO', 'III.TO'
        ]
        return canadian_stocks
    
    def download_stock_data(self, symbol, period='6mo'):
        """Download historical stock data."""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if df.empty or len(df) < 50:  # Need minimum data
                return None
            return df
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """
        Calculate comprehensive technical indicators for the stock.
        
        Returns a dataframe with all technical indicators added.
        """
        if df is None or df.empty:
            return None
        
        df = df.copy()
        
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving Averages
        df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # RSI (Relative Strength Index)
        df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
        df['RSI_7'] = ta.momentum.rsi(df['Close'], window=7)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
        df['BB_Position'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Average True Range (Volatility)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Money Flow Index
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # ADX (Trend Strength)
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        
        # Rate of Change
        df['ROC'] = ta.momentum.roc(df['Close'], window=10)
        
        # Price momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # Volatility
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # Price position relative to period high/low
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        df['Price_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
        
        # Candlestick patterns (simplified)
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Open']
        df['Upper_Shadow'] = (df['High'] - df[['Close', 'Open']].max(axis=1)) / df['Open']
        df['Lower_Shadow'] = (df[['Close', 'Open']].min(axis=1) - df['Low']) / df['Open']
        
        # Create target variable: Will it be a top performer next day?
        # Top performer = gains more than threshold% in next day
        df['Next_Day_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Target'] = (df['Next_Day_Return'] > self.prediction_threshold / 100).astype(int)
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare feature matrix for machine learning.
        """
        if df is None or df.empty:
            return None, None
        
        # Select feature columns (exclude price, volume, and target)
        feature_cols = [col for col in df.columns if col not in [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
            'Next_Day_Return', 'Target', 'High_20', 'Low_20', 'Volume_SMA_20',
            'BB_High', 'BB_Low', 'BB_Mid', 'OBV'  # Exclude some absolute values
        ]]
        
        # Remove rows with NaN values
        df_clean = df[feature_cols + ['Target']].dropna()
        
        if len(df_clean) < 30:  # Need minimum samples
            return None, None
        
        X = df_clean[feature_cols]
        y = df_clean['Target']
        
        return X, y
    
    def train_model(self, stocks_to_analyze=None):
        """
        Train the ML model on historical data from multiple stocks.
        """
        print("=" * 70)
        print("TRAINING MACHINE LEARNING MODEL")
        print("=" * 70)
        
        if stocks_to_analyze is None:
            stocks_to_analyze = self.get_canadian_stock_list()
        
        all_X = []
        all_y = []
        
        print(f"\nAnalyzing {len(stocks_to_analyze)} Canadian stocks...")
        print("Downloading historical data and calculating indicators...\n")
        
        for i, symbol in enumerate(stocks_to_analyze):
            try:
                # Download and process data
                df = self.download_stock_data(symbol)
                if df is None:
                    continue
                
                df = self.calculate_technical_indicators(df)
                if df is None:
                    continue
                
                X, y = self.prepare_features(df)
                if X is None or len(X) == 0:
                    continue
                
                all_X.append(X)
                all_y.append(y)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(stocks_to_analyze)} stocks...")
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        if not all_X:
            print("\nERROR: No valid data collected for training!")
            return False
        
        # Combine all data
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        
        print(f"\n✓ Successfully collected {len(X_combined)} training samples")
        print(f"✓ Positive samples (top performers): {y_combined.sum()} ({y_combined.mean()*100:.1f}%)")
        
        # Store feature names
        self.feature_names = X_combined.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model (using Gradient Boosting for better performance)
        print("\n" + "-" * 70)
        print("Training Gradient Boosting Classifier...")
        print("-" * 70)
        
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ Model trained successfully!")
        print(f"✓ Test Accuracy: {accuracy*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Regular', 'Top Performer']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return True
    
    def predict_next_day_performers(self, stocks_to_predict=None, top_n=10):
        """
        Predict which stocks are likely to be top performers next day.
        """
        print("\n" + "=" * 70)
        print("PREDICTING NEXT DAY TOP PERFORMERS")
        print("=" * 70)
        
        if self.model is None:
            print("ERROR: Model not trained yet! Call train_model() first.")
            return None
        
        if stocks_to_predict is None:
            stocks_to_predict = self.get_canadian_stock_list()
        
        predictions = []
        
        print(f"\nAnalyzing {len(stocks_to_predict)} stocks for tomorrow's trading...")
        
        for symbol in stocks_to_predict:
            try:
                # Get recent data
                df = self.download_stock_data(symbol, period='6mo')
                if df is None or len(df) < 50:
                    continue
                
                # Calculate indicators
                df = self.calculate_technical_indicators(df)
                if df is None:
                    continue
                
                # Get latest data point
                X, _ = self.prepare_features(df)
                if X is None or len(X) == 0:
                    continue
                
                latest_features = X.iloc[-1:][self.feature_names]
                
                # Make prediction
                X_scaled = self.scaler.transform(latest_features)
                probability = self.model.predict_proba(X_scaled)[0][1]  # Probability of being top performer
                
                # Get current price and technical info
                current_price = df['Close'].iloc[-1]
                rsi = df['RSI_14'].iloc[-1]
                volume_ratio = df['Volume_Ratio'].iloc[-1]
                bb_position = df['BB_Position'].iloc[-1]
                
                predictions.append({
                    'Symbol': symbol,
                    'Probability': probability * 100,
                    'Current_Price': current_price,
                    'RSI_14': rsi,
                    'Volume_Ratio': volume_ratio,
                    'BB_Position': bb_position,
                    'Score': probability * 100
                })
                
            except Exception as e:
                print(f"Error predicting {symbol}: {e}")
                continue
        
        if not predictions:
            print("\nNo predictions could be made!")
            return None
        
        # Create results dataframe
        results_df = pd.DataFrame(predictions)
        results_df = results_df.sort_values('Score', ascending=False).reset_index(drop=True)
        
        # Display top predictions
        print("\n" + "=" * 70)
        print(f"TOP {top_n} PREDICTED PERFORMERS FOR NEXT TRADING DAY")
        print("=" * 70)
        print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Prediction Threshold: {self.prediction_threshold}% gain\n")
        
        top_stocks = results_df.head(top_n)
        
        for idx, row in top_stocks.iterrows():
            print(f"\n{idx + 1}. {row['Symbol']}")
            print(f"   {'─' * 65}")
            print(f"   Confidence Score: {row['Probability']:.2f}%")
            print(f"   Current Price: ${row['Current_Price']:.4f}")
            print(f"   RSI (14): {row['RSI_14']:.2f} ", end="")
            
            # RSI interpretation
            if row['RSI_14'] < 30:
                print("(Oversold ✓)")
            elif row['RSI_14'] > 70:
                print("(Overbought ⚠)")
            else:
                print("(Neutral)")
            
            print(f"   Volume Ratio: {row['Volume_Ratio']:.2f}x average")
            print(f"   Bollinger Band Position: {row['BB_Position']:.2f}", end="")
            
            if row['BB_Position'] < 0.2:
                print(" (Near lower band - potential bounce)")
            elif row['BB_Position'] > 0.8:
                print(" (Near upper band - potential resistance)")
            else:
                print(" (Mid-range)")
        
        print("\n" + "=" * 70)
        print("TRADING RECOMMENDATIONS")
        print("=" * 70)
        print("""
⚠️  RISK DISCLAIMER:
   - This is a predictive model based on historical patterns
   - Past performance does NOT guarantee future results
   - Always use proper risk management and stop-losses
   - Never invest more than you can afford to lose
   - Consider this as ONE tool in your trading strategy
        
💡 SUGGESTED STRATEGY:
   - Focus on stocks with confidence > 60%
   - Look for RSI between 30-70 (avoid extreme overbought/oversold)
   - Higher volume ratio (>1.5x) indicates strong momentum
   - Set stop-loss at 2-3% below entry
   - Take profits at 5-10% gains or use trailing stops
   - Monitor pre-market activity before entering positions
   - Consider starting with paper trading to validate signals
        """)
        
        return results_df
    
    def generate_report(self, results_df, filename='stock_predictions.csv'):
        """Save predictions to CSV file."""
        if results_df is not None:
            results_df.to_csv(filename, index=False)
            print(f"\n✓ Full predictions saved to: {filename}")


def main():
    """Main execution function."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        CANADIAN STOCK DAY TRADING PREDICTOR v1.0                ║
    ║        ML-Powered Next-Day Performance Prediction                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize predictor
    # prediction_threshold: % gain to be considered a "top performer"
    # Lower threshold (e.g., 3%) = more predictions but less aggressive
    # Higher threshold (e.g., 10%) = fewer predictions but more aggressive targets
    predictor = CanadianStockPredictor(
        lookback_days=90,
        prediction_threshold=5.0  # Looking for 5%+ gains
    )
    
    # Train the model
    success = predictor.train_model()
    
    if not success:
        print("\n❌ Model training failed! Please check your internet connection and try again.")
        return
    
    # Make predictions for next day
    results = predictor.predict_next_day_performers(top_n=15)
    
    # Save results
    if results is not None:
        predictor.generate_report(results, 'next_day_predictions.csv')
    
    print("\n" + "=" * 70)
    print("Analysis complete! Happy trading! 📈")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
