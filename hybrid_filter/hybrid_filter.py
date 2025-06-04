import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DeepKalmanNetwork(nn.Module):
    """
    Neural network component that learns to predict state transitions
    and observation mappings for the Kalman filter
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(DeepKalmanNetwork, self).__init__()
        
        # Transition model network (predicts next state)
        self.transition_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Observation model network (maps state to observations)
        self.observation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Process noise covariance network - diagonal elements only
        self.process_noise_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Softplus()
        )
        
        # Observation noise network
        self.obs_noise_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        if torch.isnan(state).any() or torch.isinf(state).any():
            state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        next_state = self.transition_net(state)
        observation = self.observation_net(state)
        process_noise_diag = self.process_noise_net(state) + 1e-4
        obs_noise = self.obs_noise_net(state) + 1e-4
        
        return next_state, observation, process_noise_diag, obs_noise

class HybridKalmanFilter:
    """
    Hybrid Kalman Filter combining deep learning with traditional Kalman filtering
    """
    def __init__(self, state_dim: int, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = DeepKalmanNetwork(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.state = None
        self.covariance = None
    
    def initialize_state(self, initial_state: np.ndarray, initial_covariance: np.ndarray):
        if initial_state.shape[0] != self.state_dim:
            raise ValueError(f"Initial state dimension {initial_state.shape[0]} does not match state_dim {self.state_dim}")
        initial_state = np.nan_to_num(initial_state, nan=0.0, posinf=1.0, neginf=-1.0)
        initial_covariance = np.nan_to_num(initial_covariance, nan=0.1, posinf=1.0, neginf=0.1)
        self.state = torch.tensor(initial_state, dtype=torch.float32).to(self.device)
        self.covariance = torch.tensor(initial_covariance, dtype=torch.float32).to(self.device)
    
    def predict_step(self):
        self.network.eval()
        with torch.no_grad():
            predicted_state, _, process_noise_diag, _ = self.network(self.state.unsqueeze(0))
            process_noise = torch.diag(process_noise_diag.squeeze(0))
            self.state = predicted_state.squeeze(0)
            self.covariance = self.covariance + process_noise
            eigenvals, eigenvecs = torch.linalg.eigh(self.covariance)
            eigenvals = torch.clamp(eigenvals, min=1e-6)
            self.covariance = torch.mm(torch.mm(eigenvecs, torch.diag(eigenvals)), eigenvecs.t())
    
    def update_step(self, observation: float):
        if np.isnan(observation) or np.isinf(observation):
            return
        self.network.eval()
        with torch.no_grad():
            _, predicted_obs, _, obs_noise = self.network(self.state.unsqueeze(0))
            H = torch.zeros(1, self.state_dim).to(self.device)
            H[0, 0] = 1.0
            innovation = observation - predicted_obs.item()
            S = torch.mm(torch.mm(H, self.covariance), H.t()) + obs_noise.item()
            S = torch.clamp(S, min=1e-6)
            K = torch.mm(torch.mm(self.covariance, H.t()), torch.inverse(S))
            self.state = self.state + K.squeeze() * innovation
            I_KH = torch.eye(self.state_dim).to(self.device) - torch.mm(K, H)
            self.covariance = torch.mm(I_KH, self.covariance)
            self.state = torch.clamp(self.state, min=-100, max=100)

class StockMarketPredictor:
    """
    Main class for stock market prediction using Hybrid Kalman Filter
    """
    def __init__(self, state_dim: int = 5):
        self.state_dim = state_dim
        self.scaler = StandardScaler()
        self.hybrid_kf = HybridKalmanFilter(state_dim)
        self.close_mean = None  # To store mean of 'Close' feature
        self.close_std = None   # To store std of 'Close' feature
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['volatility'] = df['returns'].rolling(window=10).std()
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['seasonal_sin'] = np.sin(2 * np.pi * np.arange(len(df)) / 252)
        df['momentum'] = df['sma_5'] - df['sma_20']
        state_cols = ['Close', 'momentum', 'volatility', 'rsi', 'seasonal_sin']
        state_features = df[state_cols].fillna(method='bfill').fillna(method='ffill').fillna(0)
        prices = df['Close'].values
        state_features_array = np.nan_to_num(state_features.values, nan=0.0, posinf=1.0, neginf=-1.0)
        prices = np.nan_to_num(prices, nan=100.0, posinf=1000.0, neginf=10.0)
        return state_features_array, prices
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def fit(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32):
        features, prices = self.prepare_features(df)
        features_scaled = self.scaler.fit_transform(features)
        # Extract scaling parameters for 'Close' (first feature)
        self.close_mean = self.scaler.mean_[0]
        self.close_std = self.scaler.scale_[0]
        states = features_scaled
        observations = states[:, 0]  # Scaled Close prices
        
        n = len(states) - 1
        X = states[:n]  # state_t
        y_states = states[1:]  # state_{t+1}
        y_obs = observations[:n]  # observation_t
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.hybrid_kf.device)
        y_states_tensor = torch.tensor(y_states, dtype=torch.float32).to(self.hybrid_kf.device)
        y_obs_tensor = torch.tensor(y_obs, dtype=torch.float32).to(self.hybrid_kf.device)
        
        losses = []
        for epoch in range(epochs):
            epoch_losses = []
            indices = torch.randperm(len(X_tensor))
            X_shuffled = X_tensor[indices]
            y_states_shuffled = y_states_tensor[indices]
            y_obs_shuffled = y_obs_tensor[indices]
            
            for i in range(0, len(X_shuffled), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y_states = y_states_shuffled[i:i+batch_size]
                batch_y_obs = y_obs_shuffled[i:i+batch_size]
                
                predicted_states, predicted_obs, process_noise_diag, obs_noise = self.hybrid_kf.network(batch_X)
                
                state_loss = nn.MSELoss()(predicted_states, batch_y_states)
                obs_loss = nn.MSELoss()(predicted_obs.squeeze(), batch_y_obs)
                noise_reg = torch.mean(1.0 / (obs_noise + 1e-4)) + torch.mean(1.0 / (process_noise_diag + 1e-4))
                loss = state_loss + obs_loss + 0.001 * noise_reg
                
                self.hybrid_kf.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.hybrid_kf.network.parameters(), max_norm=0.5)
                self.hybrid_kf.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                losses.append(avg_loss)
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def predict(self, df: pd.DataFrame, steps_ahead: int = 1, test_prices: Optional[np.ndarray] = None):
        features, prices = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        initial_state = features_scaled[-1, :]
        initial_covariance = np.eye(self.state_dim) * 0.1
        self.hybrid_kf.initialize_state(initial_state, initial_covariance)
        
        predictions = []
        states = []
        
        for step in range(steps_ahead):
            self.hybrid_kf.predict_step()
            current_state = self.hybrid_kf.state.cpu().numpy()
            states.append(current_state.copy())
            price_pred_scaled = current_state[0]  # Predicted 'Close' in scaled space
            # Inverse transform to original scale
            price_pred_original = price_pred_scaled * self.close_std + self.close_mean
            predictions.append(price_pred_original)
            if test_prices is not None and step < len(test_prices):
                actual_price = test_prices[step]  # Original scale
                # Scale actual price to match training scale
                actual_price_scaled = (actual_price - self.close_mean) / self.close_std
                self.hybrid_kf.update_step(actual_price_scaled)
        
        return np.array(predictions), states

def demonstrate_hybrid_kalman_filter():
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    trend = np.linspace(100, 200, 1000)
    noise = np.random.normal(0, 5, 1000)
    seasonal = 10 * np.sin(np.arange(1000) * 2 * np.pi / 252)
    prices = trend + seasonal + noise
    prices = np.maximum(prices, 50)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.01, 1000)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.02, 1000))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, 1000))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 1000)
    })
    
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    print("Hybrid Kalman Filter for Stock Market Prediction")
    print("=" * 50)
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    predictor = StockMarketPredictor(state_dim=5)
    print("\nTraining model...")
    losses = predictor.fit(train_df, epochs=100, batch_size=32)
    
    print("\nMaking predictions...")
    predictions, states = predictor.predict(train_df, steps_ahead=len(test_df), test_prices=test_df['Close'].values)
    
    actual_prices = test_df['Close'].values
    mse = mean_squared_error(actual_prices[:len(predictions)], predictions)
    mae = mean_absolute_error(actual_prices[:len(predictions)], predictions)
    
    print(f"\nResults:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    if losses:
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(train_df['Close'].values[-100:], label='Train (Last 100)', alpha=0.7)
    plt.plot(range(len(train_df), len(train_df) + len(predictions)), 
             predictions, label='Predictions', color='red', linewidth=2)
    plt.plot(range(len(train_df), len(train_df) + len(actual_prices)), 
             actual_prices, label='Actual', color='green', alpha=0.7)
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    errors = actual_prices[:len(predictions)] - predictions
    plt.plot(errors)
    plt.title('Prediction Errors')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.subplot(2, 2, 4)
    if states:
        states_array = np.array(states)
        state_labels = ['Price', 'Momentum', 'Volatility', 'RSI', 'Seasonal Sin']
        for i in range(min(3, states_array.shape[1])):
            plt.plot(states_array[:, i], label=state_labels[i], alpha=0.7)
        plt.title('Hidden State Evolution')
        plt.xlabel('Time')
        plt.ylabel('State Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return predictor, df, predictions, actual_prices

if __name__ == "__main__":
    predictor, data, preds, actual = demonstrate_hybrid_kalman_filter()
    print("\nModel trained successfully!")
    print("You can now use the predictor with real Kaggle stock market data.")
    print("\nTo use with your own data:")
    print("1. Load your CSV: df = pd.read_csv('your_stock_data.csv')")
    print("2. Ensure columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']")
    print("3. Train: predictor.fit(df)")
    print("4. Predict: predictions, states = predictor.predict(df, steps_ahead=30, test_prices=your_test_prices)")