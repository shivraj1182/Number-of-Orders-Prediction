import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class OrdersPredictionModel:
    def __init__(self, data_path=None):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.predictions = {}
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, path):
        self.data = pd.read_csv(path)
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date' if 'date' in self.data.columns else self.data.columns[0])
        return self.data
    
    def split_data(self, train_ratio=0.8):
        split_idx = int(len(self.data) * train_ratio)
        self.train_data = self.data[:split_idx]
        self.test_data = self.data[split_idx:]
        return self.train_data, self.test_data
    
    def arima_forecast(self, order=(5,1,2)):
        try:
            orders = self.train_data.iloc[:, -1].values
            model = ARIMA(orders, order=order)
            fit = model.fit()
            self.predictions['ARIMA'] = fit.forecast(steps=len(self.test_data))
            return self.predictions['ARIMA']
        except Exception as e:
            print(f"ARIMA error: {e}")
            return None
    
    def exponential_smoothing_forecast(self):
        try:
            orders = self.train_data.iloc[:, -1].values
            if len(orders) > 13:
                model = ExponentialSmoothing(orders, seasonal='add', seasonal_periods=12)
                fit = model.fit()
                self.predictions['ExpSmoothing'] = fit.forecast(steps=len(self.test_data))
            return self.predictions.get('ExpSmoothing')
        except Exception as e:
            print(f"Exponential Smoothing error: {e}")
            return None
    
    def ml_regression_forecast(self):
        X_train = self._create_features(self.train_data)
        X_test = self._create_features(self.test_data)
        y_train = self.train_data.iloc[:, -1].values
        
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                self.predictions[name] = model.predict(X_test)
            except Exception as e:
                print(f"{name} error: {e}")
        
        return self.predictions
    
    def _create_features(self, df):
        data = df.iloc[:, -1].values
        features = []
        for i in range(len(data) - 4):
            features.append(data[i:i+5])
        return np.array(features)
    
    def evaluate_models(self):
        y_actual = self.test_data.iloc[:, -1].values
        results = {}
        
        for model_name, predictions in self.predictions.items():
            if predictions is not None and len(predictions) == len(y_actual):
                mae = mean_absolute_error(y_actual, predictions)
                rmse = np.sqrt(mean_squared_error(y_actual, predictions))
                r2 = r2_score(y_actual, predictions)
                results[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
        return results
    
    def plot_predictions(self):
        plt.figure(figsize=(15, 6))
        y_actual = self.test_data.iloc[:, -1].values
        plt.plot(y_actual, label='Actual', marker='o', linewidth=2)
        
        for model_name, predictions in self.predictions.items():
            if predictions is not None and len(predictions) == len(y_actual):
                plt.plot(predictions, label=model_name, marker='s', alpha=0.7)
        
        plt.xlabel('Time Period')
        plt.ylabel('Number of Orders')
        plt.title('Order Prediction Models Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt

if __name__ == '__main__':
    model = OrdersPredictionModel()
    print('Orders Prediction Model initialized')
