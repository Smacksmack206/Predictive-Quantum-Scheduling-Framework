#!/usr/bin/env python3
"""
ML-Based Process Classification for Advanced EAS
Implements intelligent process classification using machine learning
"""

# Line 1-10: Imports and dependencies
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import psutil
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Line 11-25: Data structures
@dataclass
class ProcessFeatures:
    cpu_usage_history: List[float]
    memory_usage_history: List[float]
    io_read_rate: float
    io_write_rate: float
    network_bytes_sent: float
    network_bytes_recv: float
    thread_count: int
    file_descriptors: int
    context_switches: int
    voluntary_switches: int
    involuntary_switches: int
    page_faults: int
    cpu_affinity: List[int]
    nice_value: int
    process_age: float

# Line 26-50: Feature extraction class
class ProcessFeatureExtractor:
    def __init__(self, history_window=30):
        self.history_window = history_window
        self.process_history = {}
        
    def extract_features(self, pid: int) -> Optional[ProcessFeatures]:
        try:
            proc = psutil.Process(pid)
            
            # Initialize history if new process
            if pid not in self.process_history:
                try:
                    last_io = proc.io_counters()
                except (psutil.AccessDenied, AttributeError):
                    last_io = None
                    
                try:
                    last_net = proc.connections()
                except (psutil.AccessDenied, AttributeError):
                    last_net = []
                    
                self.process_history[pid] = {
                    'cpu_history': deque(maxlen=self.history_window),
                    'memory_history': deque(maxlen=self.history_window),
                    'last_io': last_io,
                    'last_net': last_net,
                    'start_time': time.time()
                }
            
            history = self.process_history[pid]
            
            # Collect current metrics
            cpu_percent = proc.cpu_percent()
            memory_info = proc.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Update history
            history['cpu_history'].append(cpu_percent)
            history['memory_history'].append(memory_mb)   
         
            # Line 51-80: Calculate I/O and network rates
            try:
                current_io = proc.io_counters()
                if history['last_io']:
                    io_read_rate = current_io.read_bytes - history['last_io'].read_bytes
                    io_write_rate = current_io.write_bytes - history['last_io'].write_bytes
                else:
                    io_read_rate = 0
                    io_write_rate = 0
                history['last_io'] = current_io
            except (psutil.AccessDenied, AttributeError):
                io_read_rate = 0
                io_write_rate = 0
            
            # Network statistics
            try:
                connections = proc.connections()
                network_bytes_sent = sum(conn.laddr.port for conn in connections if conn.laddr)
                network_bytes_recv = len(connections)
            except (psutil.AccessDenied, AttributeError):
                network_bytes_sent = 0
                network_bytes_recv = 0
            
            # Thread and system info
            try:
                thread_count = proc.num_threads()
            except (psutil.AccessDenied, AttributeError):
                thread_count = 1
                
            try:
                file_descriptors = proc.num_fds()
            except (psutil.AccessDenied, AttributeError):
                file_descriptors = 0
                
            # Context switches
            try:
                ctx_switches = proc.num_ctx_switches()
                voluntary_switches = ctx_switches.voluntary
                involuntary_switches = ctx_switches.involuntary
            except (psutil.AccessDenied, AttributeError):
                voluntary_switches = 0
                involuntary_switches = 0
            
            # Memory info
            try:
                memory_full = proc.memory_full_info()
                page_faults = memory_full.pfaults
            except (psutil.AccessDenied, AttributeError):
                page_faults = 0
            
            # CPU affinity and priority
            try:
                cpu_affinity = proc.cpu_affinity()
            except:
                cpu_affinity = list(range(psutil.cpu_count()))
                
            nice_value = proc.nice()
            process_age = time.time() - history['start_time']
            
            return ProcessFeatures(
                cpu_usage_history=list(history['cpu_history']),
                memory_usage_history=list(history['memory_history']),
                io_read_rate=io_read_rate,
                io_write_rate=io_write_rate,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                thread_count=thread_count,
                file_descriptors=file_descriptors,
                context_switches=voluntary_switches + involuntary_switches,
                voluntary_switches=voluntary_switches,
                involuntary_switches=involuntary_switches,
                page_faults=page_faults,
                cpu_affinity=cpu_affinity,
                nice_value=nice_value,
                process_age=process_age
            )
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

# Line 81-120: ML Classifier implementation
class MLProcessClassifier:
    def __init__(self, model_path: str = "process_classifier_model.joblib"):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.feature_extractor = ProcessFeatureExtractor()
        self.model = None
        self.feature_names = [
            'cpu_mean', 'cpu_std', 'cpu_trend',
            'memory_mean', 'memory_std', 'memory_trend',
            'io_read_rate', 'io_write_rate',
            'network_bytes_sent', 'network_bytes_recv',
            'thread_count', 'file_descriptors',
            'context_switches', 'voluntary_switches', 'involuntary_switches',
            'page_faults', 'cpu_affinity_count', 'nice_value', 'process_age'
        ]
        self.load_or_create_model()
        
    def features_to_vector(self, features: ProcessFeatures) -> np.ndarray:
        # Convert ProcessFeatures to numerical vector
        cpu_history = np.array(features.cpu_usage_history) if features.cpu_usage_history else np.array([0])
        memory_history = np.array(features.memory_usage_history) if features.memory_usage_history else np.array([0])
        
        # Statistical features from history
        cpu_mean = np.mean(cpu_history)
        cpu_std = np.std(cpu_history)
        cpu_trend = np.polyfit(range(len(cpu_history)), cpu_history, 1)[0] if len(cpu_history) > 1 else 0
        
        memory_mean = np.mean(memory_history)
        memory_std = np.std(memory_history)
        memory_trend = np.polyfit(range(len(memory_history)), memory_history, 1)[0] if len(memory_history) > 1 else 0
        
        return np.array([
            cpu_mean, cpu_std, cpu_trend,
            memory_mean, memory_std, memory_trend,
            features.io_read_rate, features.io_write_rate,
            features.network_bytes_sent, features.network_bytes_recv,
            features.thread_count, features.file_descriptors,
            features.context_switches, features.voluntary_switches, features.involuntary_switches,
            features.page_faults, len(features.cpu_affinity), features.nice_value, features.process_age
        ])
        
    def load_or_create_model(self):
        try:
            # Try to load existing model
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            print(f"Loaded existing model from {self.model_path}")
        except FileNotFoundError:
            # Create new model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            print("Created new RandomForest model")   
         
    def classify_process(self, pid: int, process_name: str) -> Tuple[str, float]:
        # Extract features
        features = self.feature_extractor.extract_features(pid)
        if features is None:
            return "unknown", 0.1
            
        # Convert to vector and scale
        feature_vector = self.features_to_vector(features).reshape(1, -1)
        
        if hasattr(self.model, 'predict_proba') and hasattr(self.model, 'classes_'):
            try:
                # Use ML model if trained
                scaled_features = self.scaler.transform(feature_vector)
                probabilities = self.model.predict_proba(scaled_features)[0]
                predicted_class = self.model.classes_[np.argmax(probabilities)]
                confidence = np.max(probabilities)
                return predicted_class, confidence
            except:
                # Fallback to rule-based if ML fails
                return self._rule_based_classification(process_name, features)
        else:
            # Fallback to rule-based classification
            return self._rule_based_classification(process_name, features)
            
    def _rule_based_classification(self, name: str, features: ProcessFeatures) -> Tuple[str, float]:
        name_lower = name.lower()
        
        # High CPU + High I/O = Compute intensive
        if np.mean(features.cpu_usage_history) > 50 and features.io_read_rate > 1000000:
            return "compute_intensive", 0.8
            
        # High network activity = Network application
        if features.network_bytes_sent > 1000 or features.network_bytes_recv > 10:
            return "network_application", 0.7
            
        # Many threads + GUI indicators = Interactive application
        if features.thread_count > 10 and any(keyword in name_lower for keyword in ['app', 'ui', 'gui']):
            return "interactive_application", 0.8
            
        # Low resource usage + daemon naming = Background service
        if name_lower.endswith('d') and np.mean(features.cpu_usage_history) < 5:
            return "background_service", 0.9
            
        return "unknown", 0.3
    
    def train_model(self, training_data: List[Tuple[ProcessFeatures, str]]):
        """Train the ML model with labeled data"""
        if not training_data:
            print("No training data provided")
            return
            
        # Prepare training data
        X = []
        y = []
        
        for features, label in training_data:
            feature_vector = self.features_to_vector(features)
            X.append(feature_vector)
            y.append(label)
            
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, self.model_path)
        print(f"Model trained and saved to {self.model_path}")
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = {}
            for i, importance in enumerate(self.model.feature_importances_):
                if i < len(self.feature_names):
                    importance_dict[self.feature_names[i]] = importance
            return importance_dict
        return {}

# Test function
def test_ml_classifier():
    """Test the ML classifier"""
    print("🧠 Testing ML Process Classifier")
    print("=" * 50)
    
    classifier = MLProcessClassifier()
    
    # Test on current processes
    process_count = 0
    for proc in psutil.process_iter(['pid', 'name']):
        if process_count >= 10:  # Test first 10 processes
            break
            
        try:
            pid = proc.info['pid']
            name = proc.info['name']
            
            if pid > 10 and name:
                classification, confidence = classifier.classify_process(pid, name)
                print(f"  {name:20} → {classification:20} (confidence: {confidence:.3f})")
                process_count += 1
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
            
    # Show feature importance if available
    importance = classifier.get_feature_importance()
    if importance:
        print(f"\n📊 Top Feature Importances:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_features[:5]:
            print(f"  {feature:20}: {imp:.3f}")

if __name__ == "__main__":
    test_ml_classifier()