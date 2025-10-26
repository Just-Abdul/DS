#!/usr/bin/env python3
"""Quick test of the Canadian Stock Predictor"""

import sys
sys.path.insert(0, '/workspace')

from canadian_stock_predictor import CanadianStockPredictor

def quick_test():
    """Run a quick test with minimal stocks"""
    print("Running quick validation test...\n")
    
    # Create predictor
    predictor = CanadianStockPredictor(
        lookback_days=90,
        prediction_threshold=5.0
    )
    
    # Test with just a few stocks for speed
    test_stocks = ['SHOP.TO', 'RY.TO', 'TD.TO', 'ENB.TO', 'CNQ.TO', 
                   'ABX.TO', 'GQC.VN', 'MKA.VN']
    
    print(f"Testing with {len(test_stocks)} sample stocks...")
    
    # Train model
    success = predictor.train_model(stocks_to_analyze=test_stocks)
    
    if success:
        print("\n✓ Model training successful!")
        
        # Make predictions
        print("\n" + "="*70)
        print("Testing prediction function...")
        print("="*70)
        results = predictor.predict_next_day_performers(
            stocks_to_predict=test_stocks,
            top_n=5
        )
        
        if results is not None and len(results) > 0:
            print(f"\n✓ Successfully generated {len(results)} predictions!")
            print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
            print("\nThe predictor is ready to use!")
            print("Run: python canadian_stock_predictor.py")
            return True
        else:
            print("\n⚠ Warning: No predictions generated")
            return False
    else:
        print("\n❌ Model training failed")
        return False

if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
