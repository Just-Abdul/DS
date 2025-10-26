# Canadian Stock Day Trading Predictor

A machine learning algorithm designed to predict top-performing Canadian stocks for day trading. The system analyzes price history and technical indicators to suggest stocks likely to perform excellently the following day.

## Features

- **Data Collection**: Fetches data from major Canadian stocks
- **Technical Analysis**: Calculates RSI, MACD, Moving Averages, Bollinger Bands, and more
- **Machine Learning**: Uses Random Forest and Gradient Boosting classifiers
- **Prediction System**: Ranks stocks by probability of excellent performance
- **Two Versions**: Full version with TA-Lib and simplified version without

## Quick Start

### Option 1: Simple Version (Recommended)
```bash
python simple_stock_predictor.py
```

### Option 2: Full Version (Requires TA-Lib)
```bash
python canadian_stock_predictor.py
```

## Installation

1. **Install Python dependencies:**
   ```bash
   python setup.py
   ```

2. **Or install manually:**
   ```bash
   pip install pandas numpy requests beautifulsoup4 yfinance scikit-learn matplotlib seaborn
   ```

3. **For full version (optional):**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libta-lib-dev
   pip install TA-Lib
   
   # macOS
   brew install ta-lib
   pip install TA-Lib
   
   # Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
   ```

## How It Works

1. **Data Collection**: Fetches historical data for major Canadian stocks
2. **Technical Analysis**: Calculates various technical indicators
3. **Feature Engineering**: Creates features from price history and indicators
4. **Model Training**: Trains ML models to predict stock performance
5. **Prediction**: Ranks stocks by probability of excellent performance

## Technical Indicators Used

- **Moving Averages**: SMA (20, 50, 200), EMA (12, 26)
- **Momentum**: RSI, MACD, MACD Signal, MACD Histogram
- **Volatility**: Bollinger Bands, ATR (Average True Range)
- **Volume**: Volume ratios, OBV (On-Balance Volume)
- **Price Patterns**: Doji, Hammer, Engulfing patterns
- **Custom Features**: Price changes, volume changes, crossovers

## Output

The system provides:
- Ranked list of top predicted stocks
- Probability scores for each prediction
- Current prices and recent performance
- Risk disclaimers and educational warnings

## Important Disclaimers

⚠️ **This software is for educational purposes only.**

- **Not Financial Advice**: This is not professional financial advice
- **Risk Warning**: Trading involves substantial risk of loss
- **No Guarantees**: Past performance does not guarantee future results
- **Do Your Research**: Always conduct your own research before investing
- **Professional Advice**: Consult with financial professionals before trading

## Files

- `simple_stock_predictor.py` - Simplified version (no TA-Lib required)
- `canadian_stock_predictor.py` - Full version with advanced indicators
- `setup.py` - Installation script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Usage Example

```python
from simple_stock_predictor import SimpleStockPredictor

# Create predictor
predictor = SimpleStockPredictor()

# Run prediction pipeline
top_stocks = predictor.run_prediction_pipeline(top_n=10)

# Display results
for i, stock in enumerate(top_stocks, 1):
    print(f"{i}. {stock['ticker']}: {stock['probability']:.4f} probability")
```

## Contributing

This is an educational project. Feel free to:
- Add more technical indicators
- Improve the ML models
- Add more data sources
- Enhance the prediction logic

## License

This project is for educational purposes. Use at your own risk.

---

**Remember**: Always do your own research and never invest more than you can afford to lose!