#!/usr/bin/env python3
"""
LSTM-based Process Behavior Prediction for Advanced EAS
Predicts future CPU and memory usage patterns
"""

# Line 1-15: LSTM-based behavior prediction
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
from collections import deque
import pickle
import os
from typing import Dict, List, Optional
from ml_process_classifier import ProcessFeatures

class ProcessBehaviorPredictor:
    def __init__(self, sequence_length=30, prediction_horizon=30):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = None
        self.model_path = "behavior_predictor_model.h5"
        self.build_model()
        
    def build_model(self):
        # Line 16-35: LSTM model architecture
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, 4)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.prediction_horizon * 2)  # Predict CPU and memory
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            try:
                self.model.load_weights(self.model_path)
                print(f"Loaded existing LSTM model from {self.model_path}")
            except:
                print("Failed to load existing model, using new model")
        
    def prepare_sequence(self, cpu_history: List[float], memory_history: List[float], 
                        io_history: List[float], network_history: List[float]) -> np.ndarray:
        # Line 36-50: Prepare input sequence for LSTM
        if len(cpu_history) < self.sequence_length:
            # Pad with zeros if insufficient history
            cpu_history = [0] * (self.sequence_length - len(cpu_history)) + cpu_history
            memory_history = [0] * (self.sequence_length - len(memory_history)) + memory_history
            io_history = [0] * (self.sequence_length - len(io_history)) + io_history
            network_history = [0] * (self.sequence_length - len(network_history)) + network_history
            
        # Take last sequence_length points
        cpu_seq = cpu_history[-self.sequence_length:]
        memory_seq = memory_history[-self.sequence_length:]
        io_seq = io_history[-self.sequence_length:]
        network_seq = network_history[-self.sequence_length:]
        
        # Stack features
        sequence = np.column_stack([cpu_seq, memory_seq, io_seq, network_seq])
        return sequence.reshape(1, self.sequence_length, 4)
        
    def predict_behavior(self, process_features: ProcessFeatures) -> Dict[str, List[float]]:
        # Line 51-70: Generate predictions
        if self.model is None:
            return {'cpu_prediction': [0] * self.prediction_horizon, 
                   'memory_prediction': [0] * self.prediction_horizon}
                   
        # Prepare input sequence
        io_history = [process_features.io_read_rate] * self.sequence_length  # Simplified
        network_history = [process_features.network_bytes_sent] * self.sequence_length  # Simplified
        
        sequence = self.prepare_sequence(
            process_features.cpu_usage_history,
            process_features.memory_usage_history,
            io_history,
            network_history
        )
        
        # Make prediction
        prediction = self.model.predict(sequence, verbose=0)[0]
        
        # Split prediction into CPU and memory
        cpu_prediction = prediction[:self.prediction_horizon].tolist()
        memory_prediction = prediction[self.prediction_horizon:].tolist()
        
        return {
            'cpu_prediction': cpu_prediction,
            'memory_prediction': memory_prediction,
            'confidence': self._calculate_prediction_confidence(prediction)
        }
        
    def _calculate_prediction_confidence(self, prediction: np.ndarray) -> float:
        # Line 71-80: Calculate confidence based on prediction stability
        variance = np.var(prediction)
        # Lower variance = higher confidence
        confidence = max(0.1, min(0.9, 1.0 / (1.0 + variance)))
        return confidence
        
    def train_model(self, training_sequences: List[np.ndarray], training_targets: List[np.ndarray]):
        """Train the LSTM model with historical data"""
        if not training_sequences or not training_targets:
            print("No training data provided")
            return
            
        X = np.array(training_sequences)
        y = np.array(training_targets)
        
        print(f"Training LSTM model with {len(X)} sequences...")
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Save model
        self.model.save_weights(self.model_path)
        print(f"Model trained and saved to {self.model_path}")
        
        return history
        
    def generate_synthetic_training_data(self, num_sequences: int = 1000) -> tuple:
        """Generate synthetic training data for initial model training"""
        print(f"Generating {num_sequences} synthetic training sequences...")
        
        sequences = []
        targets = []
        
        for _ in range(num_sequences):
            # Generate synthetic process behavior patterns
            pattern_type = np.random.choice(['steady', 'bursty', 'declining', 'growing'])
            
            if pattern_type == 'steady':
                base_cpu = np.random.uniform(10, 50)
                base_memory = np.random.uniform(50, 200)
                cpu_seq = base_cpu + np.random.normal(0, 5, self.sequence_length)
                memory_seq = base_memory + np.random.normal(0, 10, self.sequence_length)
                
            elif pattern_type == 'bursty':
                base_cpu = np.random.uniform(5, 20)
                base_memory = np.random.uniform(30, 100)
                cpu_seq = base_cpu + np.random.exponential(20, self.sequence_length)
                memory_seq = base_memory + np.random.exponential(50, self.sequence_length)
                
            elif pattern_type == 'declining':
                start_cpu = np.random.uniform(50, 90)
                start_memory = np.random.uniform(100, 300)
                cpu_seq = np.linspace(start_cpu, start_cpu * 0.3, self.sequence_length)
                memory_seq = np.linspace(start_memory, start_memory * 0.5, self.sequence_length)
                
            else:  # growing
                start_cpu = np.random.uniform(5, 30)
                start_memory = np.random.uniform(20, 80)
                cpu_seq = np.linspace(start_cpu, start_cpu * 2, self.sequence_length)
                memory_seq = np.linspace(start_memory, start_memory * 1.5, self.sequence_length)
            
            # Add noise
            cpu_seq += np.random.normal(0, 2, self.sequence_length)
            memory_seq += np.random.normal(0, 5, self.sequence_length)
            
            # Clamp values
            cpu_seq = np.clip(cpu_seq, 0, 100)
            memory_seq = np.clip(memory_seq, 0, 1000)
            
            # Generate I/O and network patterns
            io_seq = np.random.exponential(1000, self.sequence_length)
            network_seq = np.random.poisson(5, self.sequence_length)
            
            # Create sequence
            sequence = np.column_stack([cpu_seq, memory_seq, io_seq, network_seq])
            sequences.append(sequence)
            
            # Generate target (future values)
            if pattern_type == 'steady':
                future_cpu = cpu_seq[-1] + np.random.normal(0, 3, self.prediction_horizon)
                future_memory = memory_seq[-1] + np.random.normal(0, 8, self.prediction_horizon)
            elif pattern_type == 'bursty':
                future_cpu = np.maximum(cpu_seq[-1] * 0.8, 
                                      cpu_seq[-1] + np.random.exponential(10, self.prediction_horizon))
                future_memory = memory_seq[-1] + np.random.exponential(30, self.prediction_horizon)
            elif pattern_type == 'declining':
                future_cpu = cpu_seq[-1] * np.linspace(1.0, 0.7, self.prediction_horizon)
                future_memory = memory_seq[-1] * np.linspace(1.0, 0.8, self.prediction_horizon)
            else:  # growing
                future_cpu = cpu_seq[-1] * np.linspace(1.0, 1.3, self.prediction_horizon)
                future_memory = memory_seq[-1] * np.linspace(1.0, 1.2, self.prediction_horizon)
            
            # Clamp future values
            future_cpu = np.clip(future_cpu, 0, 100)
            future_memory = np.clip(future_memory, 0, 1000)
            
            # Combine CPU and memory predictions
            target = np.concatenate([future_cpu, future_memory])
            targets.append(target)
        
        return sequences, targets

# Test function
def test_behavior_predictor():
    """Test the behavior predictor"""
    print("🔮 Testing LSTM Behavior Predictor")
    print("=" * 50)
    
    predictor = ProcessBehaviorPredictor()
    
    # Generate and train on synthetic data
    sequences, targets = predictor.generate_synthetic_training_data(100)
    predictor.train_model(sequences, targets)
    
    # Test prediction with dummy process features
    dummy_features = ProcessFeatures(
        cpu_usage_history=[20, 25, 30, 28, 32, 35, 33, 30, 28, 25] * 3,
        memory_usage_history=[100, 105, 110, 108, 112, 115, 113, 110, 108, 105] * 3,
        io_read_rate=5000,
        io_write_rate=2000,
        network_bytes_sent=1000,
        network_bytes_recv=5,
        thread_count=8,
        file_descriptors=20,
        context_switches=1000,
        voluntary_switches=800,
        involuntary_switches=200,
        page_faults=50,
        cpu_affinity=[0, 1, 2, 3],
        nice_value=0,
        process_age=3600
    )
    
    # Make prediction
    prediction = predictor.predict_behavior(dummy_features)
    
    print(f"📊 Prediction Results:")
    print(f"  CPU Prediction (next 30 points): {prediction['cpu_prediction'][:10]}...")
    print(f"  Memory Prediction (next 30 points): {prediction['memory_prediction'][:10]}...")
    print(f"  Confidence: {prediction['confidence']:.3f}")

if __name__ == "__main__":
    test_behavior_predictor()