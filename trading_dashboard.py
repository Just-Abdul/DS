"""
Advanced Trading Dashboard for Canadian Stock Predictions
========================================================

Interactive web dashboard using Dash for real-time stock predictions
and portfolio monitoring.
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

from stock_predictor import CanadianStockPredictor
import threading
import time


class TradingDashboard:
    """
    Interactive web dashboard for Canadian stock trading predictions.
    """
    
    def __init__(self):
        self.predictor = CanadianStockPredictor()
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.predictions = []
        self.watchlist = []
        
        # Load or train model
        if not self.predictor.load_model():
            print("No existing model found. Please run the main predictor first to train the model.")
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """
        Setup the dashboard layout.
        """
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🇨🇦 Canadian Stock Day Trading Predictor", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Control Panel"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Number of Predictions:"),
                                    dbc.Input(
                                        id="n-predictions",
                                        type="number",
                                        value=10,
                                        min=5,
                                        max=50
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Minimum Return Threshold:"),
                                    dbc.Input(
                                        id="return-threshold",
                                        type="number",
                                        value=0.02,
                                        min=0.001,
                                        max=0.1,
                                        step=0.001
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Action:"),
                                    html.Br(),
                                    dbc.Button(
                                        "Generate Predictions",
                                        id="predict-button",
                                        color="primary",
                                        className="me-2"
                                    ),
                                    dbc.Button(
                                        "Refresh Data",
                                        id="refresh-button",
                                        color="secondary"
                                    )
                                ], width=4)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Status and Loading
            dbc.Row([
                dbc.Col([
                    dbc.Alert(
                        "Ready to generate predictions. Click 'Generate Predictions' to start.",
                        id="status-alert",
                        color="info",
                        dismissable=True
                    )
                ])
            ]),
            
            # Loading spinner
            dcc.Loading(
                id="loading",
                type="default",
                children=[
                    # Main Results
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("📈 Top Stock Predictions"),
                                dbc.CardBody([
                                    dash_table.DataTable(
                                        id="predictions-table",
                                        columns=[
                                            {"name": "Rank", "id": "rank"},
                                            {"name": "Symbol", "id": "symbol"},
                                            {"name": "Current Price", "id": "current_price", "type": "numeric", "format": {"specifier": ".2f"}},
                                            {"name": "Predicted Price", "id": "predicted_price", "type": "numeric", "format": {"specifier": ".2f"}},
                                            {"name": "Expected Return", "id": "predicted_return", "type": "numeric", "format": {"specifier": ".2%"}},
                                            {"name": "Confidence", "id": "confidence_score", "type": "numeric", "format": {"specifier": ".2f"}},
                                        ],
                                        style_cell={'textAlign': 'center'},
                                        style_data_conditional=[
                                            {
                                                'if': {'filter_query': '{predicted_return} > 0.05'},
                                                'backgroundColor': '#d4edda',
                                                'color': 'black',
                                            },
                                            {
                                                'if': {'filter_query': '{predicted_return} > 0.03'},
                                                'backgroundColor': '#fff3cd',
                                                'color': 'black',
                                            }
                                        ],
                                        sort_action="native",
                                        page_size=15
                                    )
                                ])
                            ])
                        ], width=12)
                    ], className="mb-4"),
                    
                    # Charts
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("📊 Prediction Analysis"),
                                dbc.CardBody([
                                    dcc.Graph(id="prediction-charts")
                                ])
                            ])
                        ], width=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("🎯 Risk Analysis"),
                                dbc.CardBody([
                                    dcc.Graph(id="risk-analysis")
                                ])
                            ])
                        ], width=4)
                    ], className="mb-4"),
                    
                    # Individual Stock Analysis
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("🔍 Individual Stock Analysis"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Select Stock:"),
                                            dcc.Dropdown(
                                                id="stock-dropdown",
                                                placeholder="Select a stock to analyze"
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            dbc.Label("Time Period:"),
                                            dcc.Dropdown(
                                                id="period-dropdown",
                                                options=[
                                                    {"label": "1 Month", "value": "1mo"},
                                                    {"label": "3 Months", "value": "3mo"},
                                                    {"label": "6 Months", "value": "6mo"},
                                                    {"label": "1 Year", "value": "1y"}
                                                ],
                                                value="3mo"
                                            )
                                        ], width=6)
                                    ]),
                                    html.Hr(),
                                    dcc.Graph(id="individual-stock-chart")
                                ])
                            ])
                        ], width=12)
                    ])
                ]
            )
        ], fluid=True)
    
    def setup_callbacks(self):
        """
        Setup dashboard callbacks for interactivity.
        """
        
        @self.app.callback(
            [Output("predictions-table", "data"),
             Output("prediction-charts", "figure"),
             Output("risk-analysis", "figure"),
             Output("stock-dropdown", "options"),
             Output("status-alert", "children"),
             Output("status-alert", "color")],
            [Input("predict-button", "n_clicks"),
             Input("refresh-button", "n_clicks")],
            [State("n-predictions", "value"),
             State("return-threshold", "value")]
        )
        def update_predictions(predict_clicks, refresh_clicks, n_pred, threshold):
            ctx = dash.callback_context
            
            if not ctx.triggered:
                return [], {}, {}, [], "Ready to generate predictions.", "info"
            
            try:
                # Generate predictions
                self.predictions = self.predictor.get_top_predictions(
                    n_top=n_pred or 10,
                    min_return_threshold=threshold or 0.02
                )
                
                if not self.predictions:
                    return [], {}, {}, [], "No predictions generated. Try lowering the return threshold.", "warning"
                
                # Prepare table data
                table_data = []
                for i, pred in enumerate(self.predictions, 1):
                    table_data.append({
                        "rank": i,
                        "symbol": pred["symbol"],
                        "current_price": pred["current_price"],
                        "predicted_price": pred["predicted_price"],
                        "predicted_return": pred["predicted_return"],
                        "confidence_score": pred["confidence_score"]
                    })
                
                # Create prediction charts
                prediction_fig = self.create_prediction_charts()
                risk_fig = self.create_risk_analysis()
                
                # Update dropdown options
                dropdown_options = [{"label": pred["symbol"], "value": pred["symbol"]} 
                                  for pred in self.predictions]
                
                status_msg = f"Generated {len(self.predictions)} predictions successfully!"
                
                return table_data, prediction_fig, risk_fig, dropdown_options, status_msg, "success"
                
            except Exception as e:
                error_msg = f"Error generating predictions: {str(e)}"
                return [], {}, {}, [], error_msg, "danger"
        
        @self.app.callback(
            Output("individual-stock-chart", "figure"),
            [Input("stock-dropdown", "value"),
             Input("period-dropdown", "value")]
        )
        def update_individual_chart(selected_stock, period):
            if not selected_stock:
                return {}
            
            try:
                # Fetch stock data
                stock_data = yf.Ticker(selected_stock).history(period=period or "3mo")
                
                if stock_data.empty:
                    return {}
                
                # Calculate technical indicators
                tech_data = self.predictor.calculate_technical_indicators(stock_data)
                
                # Create candlestick chart with technical indicators
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(f'{selected_stock} Price & Technical Indicators', 'RSI', 'MACD'),
                    row_width=[0.2, 0.1, 0.1]
                )
                
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=tech_data.index,
                        open=tech_data['Open'],
                        high=tech_data['High'],
                        low=tech_data['Low'],
                        close=tech_data['Close'],
                        name="Price"
                    ),
                    row=1, col=1
                )
                
                # Add moving averages
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['SMA_20'],
                              name='SMA 20', line=dict(color='orange')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['SMA_50'],
                              name='SMA 50', line=dict(color='red')),
                    row=1, col=1
                )
                
                # RSI
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['RSI'],
                              name='RSI', line=dict(color='purple')),
                    row=2, col=1
                )
                
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['MACD'],
                              name='MACD', line=dict(color='blue')),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['MACD_signal'],
                              name='Signal', line=dict(color='red')),
                    row=3, col=1
                )
                
                fig.update_layout(
                    title=f"{selected_stock} Technical Analysis",
                    xaxis_rangeslider_visible=False,
                    height=800
                )
                
                return fig
                
            except Exception as e:
                return {}
    
    def create_prediction_charts(self):
        """
        Create comprehensive prediction visualization.
        """
        if not self.predictions:
            return {}
        
        df = pd.DataFrame(self.predictions)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Expected Returns', 'Confidence vs Return', 
                          'Price Comparison', 'Model Predictions Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Expected Returns
        fig.add_trace(
            go.Bar(
                x=df['symbol'],
                y=df['predicted_return'],
                name='Expected Return',
                marker_color='green',
                text=[f"{x:.2%}" for x in df['predicted_return']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Plot 2: Confidence vs Return scatter
        fig.add_trace(
            go.Scatter(
                x=df['confidence_score'],
                y=df['predicted_return'],
                mode='markers+text',
                text=df['symbol'],
                textposition='top center',
                name='Confidence vs Return',
                marker=dict(
                    size=10,
                    color=df['predicted_return'],
                    colorscale='Viridis',
                    showscale=True
                )
            ),
            row=1, col=2
        )
        
        # Plot 3: Current vs Predicted Price
        fig.add_trace(
            go.Bar(
                x=df['symbol'],
                y=df['current_price'],
                name='Current Price',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=df['symbol'],
                y=df['predicted_price'],
                name='Predicted Price',
                marker_color='darkblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Plot 4: Distribution of predictions
        all_model_preds = []
        for pred in self.predictions:
            all_model_preds.extend(list(pred['predictions'].values()))
        
        fig.add_trace(
            go.Histogram(
                x=all_model_preds,
                name='Model Predictions',
                nbinsx=20,
                marker_color='orange',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Prediction Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_risk_analysis(self):
        """
        Create risk analysis visualization.
        """
        if not self.predictions:
            return {}
        
        df = pd.DataFrame(self.predictions)
        
        # Calculate risk metrics
        returns = df['predicted_return']
        confidence = df['confidence_score']
        
        # Risk-Return scatter
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=confidence,
            y=returns,
            mode='markers+text',
            text=df['symbol'],
            textposition='top center',
            marker=dict(
                size=15,
                color=returns,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Expected Return")
            ),
            name='Stocks'
        ))
        
        # Add quadrant lines
        mean_confidence = confidence.mean()
        mean_return = returns.mean()
        
        fig.add_vline(x=mean_confidence, line_dash="dash", line_color="gray")
        fig.add_hline(y=mean_return, line_dash="dash", line_color="gray")
        
        # Add quadrant labels
        fig.add_annotation(
            x=confidence.max() * 0.9,
            y=returns.max() * 0.9,
            text="High Confidence<br>High Return",
            showarrow=False,
            bgcolor="lightgreen",
            opacity=0.7
        )
        
        fig.add_annotation(
            x=confidence.min() * 1.1,
            y=returns.max() * 0.9,
            text="Low Confidence<br>High Return",
            showarrow=False,
            bgcolor="yellow",
            opacity=0.7
        )
        
        fig.update_layout(
            title="Risk-Return Analysis",
            xaxis_title="Confidence Score",
            yaxis_title="Expected Return",
            height=400
        )
        
        return fig
    
    def run(self, debug=False, port=8050):
        """
        Run the dashboard server.
        """
        print(f"Starting Trading Dashboard on http://localhost:{port}")
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')


def main():
    """
    Main function to run the trading dashboard.
    """
    dashboard = TradingDashboard()
    dashboard.run(debug=True)


if __name__ == "__main__":
    main()