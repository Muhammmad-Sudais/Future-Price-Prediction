# Future-Price-Prediction
A Streamlit app for predicting next day's stock closing prices using Yahoo Finance data and Random Forest model. Features OHLCV inputs, MAE evaluation, and interactive charts for stocks like AAPL and TSLA. Run with streamlit run app.py.
# Task Objective
The project aims to predict the next trading day's closing stock price for selected assets (e.g., AAPL, TSLA) using historical market data. It provides an interactive dashboard for users to select a stock ticker, specify a training start date, train a machine learning model, and visualize predictions alongside actual market trends. The goal is to assist users in making informed investment decisions by forecasting short-term price movements based on features like Open, High, Low, Close, and Volume (OHLCV).
# Dataset Used
Source: Historical stock data fetched from Yahoo Finance via the yfinance library.
Tickers: Predefined options include AAPL (Apple), TSLA (Tesla), MSFT (Microsoft), GOOGL (Alphabet), AMZN (Amazon), NVDA (NVIDIA), META (Meta Platforms), and NFLX (Netflix). Users can select from these in the app.
Time Period: Data starts from a user-specified date (default: January 1, 2020) up to the most recent available trading day.
Features: OHLCV data (Open, High, Low, Close prices, and Volume) for each trading day.
Target: The next day's closing price (shifted by one day).
Preprocessing: Data is cleaned by dropping rows with missing values. Features are used directly without additional scaling or normalization in the code.
# Models Applied
Primary Model: Random Forest Regressor from scikit-learn (RandomForestRegressor with 100 estimators and random_state=42).
Training Process:
Data is split into training (80%) and testing (20%) sets without shuffling to maintain temporal order.
Features (OHLCV) are used to predict the target (next day's close).
The model is trained on historical data and evaluated on the test set.
Prediction: For the next trading day, the model uses the most recent OHLCV data to generate a single-point forecast.
Evaluation Metric: Mean Absolute Error (MAE) is calculated on the test set to measure prediction accuracy.
# Key Results and Findings
Performance: The app displays the MAE (e.g., ±$X.XX) as a measure of model error on the test set. Actual results depend on the stock and date range but typically show moderate accuracy for short-term predictions (e.g., MAE around $1-5 for major stocks like AAPL, based on typical stock volatility).
Visualization: Interactive Plotly charts compare actual historical prices (area chart) with model predictions (dashed line) on the test set. This highlights how well the model fits recent trends.
Findings:
Random Forest performs well for capturing non-linear patterns in stock data but may struggle with extreme volatility or external events (e.g., earnings reports).
Longer training histories improve accuracy but increase computation time.
The model is not designed for long-term forecasts and should be used cautiously, as stock prediction is inherently uncertain.
No overfitting mitigation (e.g., hyperparameter tuning) is implemented beyond default settings, so results may vary.
