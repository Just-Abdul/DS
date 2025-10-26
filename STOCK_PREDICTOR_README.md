# Canadian Stock Day Trading Predictor 📈

A machine learning-powered tool for predicting next-day top-performing Canadian stocks for day trading.

## 🎯 What Does This Do?

This algorithm:
1. **Analyzes** historical data from 100+ Canadian stocks (TSX, TSXV, CSE)
2. **Calculates** 30+ technical indicators (RSI, MACD, Bollinger Bands, Volume, etc.)
3. **Trains** a machine learning model to recognize patterns of stocks that become top performers
4. **Predicts** which stocks are likely to gain 5%+ the next trading day
5. **Provides** ranked recommendations with confidence scores

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### Run the Predictor

```bash
python canadian_stock_predictor.py
```

The script will:
- Download 6 months of historical data for Canadian stocks
- Train the ML model (takes 2-5 minutes)
- Generate predictions for the next trading day
- Save results to `next_day_predictions.csv`

## 📊 Technical Indicators Used

### Momentum Indicators
- **RSI (7, 14)**: Identifies overbought/oversold conditions
- **Stochastic Oscillator**: Momentum indicator comparing closing price to price range
- **ROC (Rate of Change)**: Measures momentum

### Trend Indicators
- **Moving Averages (SMA 5, 10, 20, 50)**: Trend direction
- **EMA (12, 26)**: Exponential moving averages
- **MACD**: Trend and momentum
- **ADX**: Trend strength

### Volatility Indicators
- **Bollinger Bands**: Price volatility and potential breakouts
- **ATR (Average True Range)**: Volatility measurement
- **Historical Volatility**: Standard deviation of returns

### Volume Indicators
- **Volume Ratio**: Current volume vs average
- **OBV (On-Balance Volume)**: Volume-price momentum
- **MFI (Money Flow Index)**: Volume-weighted RSI

### Price Action
- **Support/Resistance Levels**: 20-day highs and lows
- **Candlestick Patterns**: Body size, shadows
- **Price Position**: Relative position in recent range

## 🎓 How the ML Model Works

### Training Phase
1. **Data Collection**: Gathers historical data from 100+ Canadian stocks
2. **Feature Engineering**: Calculates 30+ technical indicators for each day
3. **Label Creation**: Identifies which days were followed by 5%+ gains
4. **Model Training**: Gradient Boosting Classifier learns patterns
5. **Validation**: Tests accuracy on unseen data

### Prediction Phase
1. **Current Data**: Fetches latest data for all stocks
2. **Feature Calculation**: Computes all technical indicators
3. **Probability Scoring**: ML model assigns probability (0-100%)
4. **Ranking**: Sorts stocks by confidence score
5. **Filtering**: Provides top candidates with detailed metrics

## 📈 Understanding the Output

### Example Output:
```
1. GQC.VN
   ─────────────────────────────────────────────────────────────────
   Confidence Score: 78.45%
   Current Price: $1.5500
   RSI (14): 45.23 (Neutral)
   Volume Ratio: 2.3x average
   Bollinger Band Position: 0.35 (Mid-range)
```

### Key Metrics Explained:

- **Confidence Score**: ML model's confidence (higher = stronger signal)
  - 70-100%: Very strong signal
  - 60-70%: Strong signal
  - 50-60%: Moderate signal
  - <50%: Weak signal

- **RSI (14)**: 
  - <30: Oversold (potential bounce)
  - 30-70: Neutral
  - >70: Overbought (potential pullback)

- **Volume Ratio**:
  - >2.0x: Very high interest/momentum
  - 1.5-2.0x: Elevated interest
  - <1.0x: Below average interest

- **Bollinger Band Position**:
  - <0.2: Near lower band (potential bounce)
  - 0.4-0.6: Mid-range
  - >0.8: Near upper band (potential resistance)

## 💡 Trading Strategy Recommendations

### Entry Strategy
1. **Filter by Confidence**: Focus on stocks with >60% confidence
2. **Confirm with RSI**: Avoid extreme RSI (<20 or >80)
3. **Volume Check**: Prefer stocks with >1.5x volume ratio
4. **Pre-Market**: Monitor pre-market activity before entering

### Risk Management
- **Position Size**: Never risk more than 1-2% of portfolio per trade
- **Stop Loss**: Set at 2-3% below entry price
- **Take Profit**: Target 5-10% gains or use trailing stops
- **Max Positions**: Limit to 3-5 positions at once

### Exit Strategy
- **Profit Target**: Take profits at 5%+ gains
- **Trailing Stop**: Use 2-3% trailing stop after 5% gain
- **Time Stop**: Exit before market close (day trading)
- **Cut Losses**: Exit immediately if stop loss hit

## ⚠️ Important Disclaimers

### Risk Warning
- **Past performance does NOT guarantee future results**
- **This is a predictive tool, not financial advice**
- **Trading stocks involves substantial risk of loss**
- **Never invest more than you can afford to lose**
- **Always do your own research**

### Limitations
- Model accuracy varies with market conditions
- Predictions are based on technical analysis only (no fundamentals)
- Unexpected news/events can invalidate predictions
- Low-volume stocks may have limited liquidity
- Penny stocks carry higher risk

### Best Practices
- ✅ Use as ONE tool in your trading strategy
- ✅ Combine with your own research and analysis
- ✅ Start with paper trading to validate signals
- ✅ Keep position sizes small initially
- ✅ Use proper risk management always
- ❌ Don't blindly follow predictions
- ❌ Don't over-leverage
- ❌ Don't ignore risk management

## 🔧 Customization

### Adjust Prediction Threshold
In `canadian_stock_predictor.py`, modify:
```python
predictor = CanadianStockPredictor(
    lookback_days=90,
    prediction_threshold=5.0  # Change this (e.g., 3.0 for 3% gain target)
)
```

### Add More Stocks
Edit the `get_canadian_stock_list()` method to add your preferred stocks:
```python
canadian_stocks = [
    'YOUR.TO',  # Add your stocks here
    # ... existing stocks
]
```

### Modify Model Parameters
In the `train_model()` method:
```python
self.model = GradientBoostingClassifier(
    n_estimators=200,      # Increase for more complexity
    learning_rate=0.1,     # Decrease for more conservative learning
    max_depth=5,           # Increase for more complex patterns
    # ... other parameters
)
```

## 📊 Output Files

- **next_day_predictions.csv**: Full list of predictions with scores
  - Symbol, Probability, Current_Price, RSI_14, Volume_Ratio, BB_Position, Score

## 🔄 Daily Usage Workflow

1. **Morning (Before Market Open)**:
   ```bash
   python canadian_stock_predictor.py
   ```

2. **Review Top Predictions**: Check top 10-15 stocks

3. **Additional Research**: 
   - Check news for selected stocks
   - Review charts on TradingView
   - Verify volume and liquidity

4. **Execute Trades**:
   - Enter positions after market open (9:35-10:00 AM EST)
   - Set stop losses immediately
   - Monitor positions throughout the day

5. **End of Day**:
   - Close all positions before 3:55 PM EST
   - Record results for performance tracking

## 📈 Performance Tracking

Keep a trading journal with:
- Date and time of trade
- Stock symbol
- Entry/exit prices
- Confidence score from predictor
- Actual outcome (% gain/loss)
- Notes and lessons learned

This helps you:
- Identify optimal confidence thresholds
- Recognize market conditions where model performs best
- Improve your trading strategy over time

## 🆘 Troubleshooting

### "No module named 'ta'"
```bash
pip install ta
```

### "No data collected for training"
- Check your internet connection
- Some stocks may be delisted or have insufficient data
- Try running again during market hours

### Model accuracy seems low
- Retrain during different market conditions
- Adjust prediction_threshold to match your risk tolerance
- Consider adding more stocks to training set
- Market conditions may have changed significantly

### Predictions not updating
- Delete old data cache
- Ensure yfinance is up to date: `pip install --upgrade yfinance`

## 📚 Further Learning

### Recommended Resources
- **Technical Analysis**: "Technical Analysis of the Financial Markets" by John Murphy
- **Machine Learning in Trading**: "Advances in Financial Machine Learning" by Marcos López de Prado
- **Risk Management**: "Trading in the Zone" by Mark Douglas

### Practice Platforms
- **Paper Trading**: TradingView, Questrade Practice, Interactive Brokers Paper Trading
- **Backtesting**: QuantConnect, Backtrader, Zipline

## 🤝 Contributing

Feel free to:
- Add more technical indicators
- Experiment with different ML algorithms
- Improve feature engineering
- Add fundamental analysis features
- Create visualization dashboards

## 📞 Support

For issues or questions:
1. Review this README thoroughly
2. Check troubleshooting section
3. Verify all dependencies are installed
4. Test with paper trading first

---

**Remember**: This tool is designed to assist your trading decisions, not replace your judgment. Always trade responsibly and within your risk tolerance.

**Good luck and happy trading! 📈🚀**
