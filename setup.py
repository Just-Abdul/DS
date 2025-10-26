#!/usr/bin/env python3
"""
Setup script for Canadian Stock Predictor
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    # Basic requirements that should work on most systems
    basic_requirements = [
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'requests>=2.28.0',
        'beautifulsoup4>=4.11.0',
        'yfinance>=0.2.0',
        'scikit-learn>=1.1.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0'
    ]
    
    for package in basic_requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            print("Please install manually: pip install " + package)
    
    # Try to install TA-Lib (optional)
    try:
        print("Attempting to install TA-Lib...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'TA-Lib'])
        print("TA-Lib installed successfully!")
    except subprocess.CalledProcessError:
        print("TA-Lib installation failed. This is optional - the simple version will work without it.")
        print("To install TA-Lib manually:")
        print("  - On Ubuntu/Debian: sudo apt-get install libta-lib-dev && pip install TA-Lib")
        print("  - On macOS: brew install ta-lib && pip install TA-Lib")
        print("  - On Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")

def main():
    """Main setup function."""
    print("=" * 60)
    print("Canadian Stock Predictor Setup")
    print("=" * 60)
    
    install_requirements()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("You can now run:")
    print("  python simple_stock_predictor.py  (recommended - no TA-Lib required)")
    print("  python canadian_stock_predictor.py  (full version with TA-Lib)")
    print("=" * 60)

if __name__ == "__main__":
    main()