#!/usr/bin/env python3
"""
Reinforcement Learning Scheduler for Advanced EAS
Deep Q-Network based intelligent process scheduling
"""

# Line 1-30: RL Scheduler imports and setup
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
from typing import Dict, List, Tuple, Optional
import pickle
import os

class RLSchedulerEnvironment:
    """Environment for reinforcement learning scheduler"""
    
    def __init__(self, num_cores: int = 8):
        self.num_cores = num_cores
        self.p_cores = 4  # Performance cores
        self.e_cores = 4  # Efficiency cores
        
        # State space: [core_loads, process_priorities, thermal_state, battery_level]
        self.state_size = num_cores + 10  # Core loads + additional features
        
        # Action space: Core assignment for each process (0-7 for 8 cores)
        self.action_size = num_cores
        
        self.reset()
        
    def reset(self) -> np.ndarray:
        # Line 31-45: Reset environment state
        self.core_loads = np.zeros(self.num_cores)
        self.thermal_pressure = 0.0
        self.battery_level = 1.0
        self.power_consumption = 0.0
        self.performance_score = 0.0
        self.step_count = 0
        
        return self._get_state()
        
    def _get_state(self) -> np.ndarray:
        # Line 46-55: Get current environment state
        state = np.concatenate([
            self.core_loads,
            [self.thermal_pressure],
            [self.battery_level],
            [self.power_consumption],
            [self.performance_score],
            [self.step_count / 1000.0],  # Normalized step count
            np.zeros(5)  # Reserved for future features
        ])
        return state
        
    def step(self, action: int, process_load: float, process_priority: float) -> Tuple[np.ndarray, float, bool]:
        # Line 56-90: Execute action and return new state, reward, done
        self.step_count += 1
        
        # Validate action
        core_id = action % self.num_cores
        
        # Update core load
        self.core_loads[core_id] += process_load
        
        # Calculate reward
        reward = self._calculate_reward(core_id, process_load, process_priority)
        
        # Update system metrics
        self._update_system_metrics()
        
        # Check if episode is done (simplified)
        done = self.step_count >= 1000 or self.thermal_pressure > 0.9
        
        return self._get_state(), reward, done
        
    def _calculate_reward(self, core_id: int, process_load: float, process_priority: float) -> float:
        # Line 91-120: Calculate reward for the action
        reward = 0.0
        
        # Performance reward: High priority processes on P-cores
        if core_id < self.p_cores and process_priority > 0.7:
            reward += 10.0  # Good assignment
        elif core_id >= self.p_cores and process_priority < 0.3:
            reward += 5.0   # Good assignment to E-core
        else:
            reward -= 2.0   # Suboptimal assignment
            
        # Load balancing reward
        core_load_variance = np.var(self.core_loads)
        reward -= core_load_variance * 5.0  # Penalize uneven load distribution
        
        # Thermal penalty
        if self.thermal_pressure > 0.7:
            reward -= 15.0
        elif self.thermal_pressure > 0.5:
            reward -= 5.0
            
        # Power efficiency reward
        if self.power_consumption < 0.5:
            reward += 3.0
        elif self.power_consumption > 0.8:
            reward -= 8.0
            
        # Core type efficiency
        if core_id < self.p_cores:  # P-core
            # P-cores are less efficient but more powerful
            reward -= process_load * 2.0  # Power penalty
            reward += process_priority * 5.0  # Performance bonus
        else:  # E-core
            # E-cores are more efficient
            reward += (1.0 - process_load) * 3.0  # Efficiency bonus
            
        return reward
        
    def _update_system_metrics(self):
        # Line 121-135: Update system-wide metrics
        # Thermal pressure based on core loads
        max_load = np.max(self.core_loads)
        avg_load = np.mean(self.core_loads)
        self.thermal_pressure = min(1.0, (max_load + avg_load) / 2.0)
        
        # Power consumption estimation
        p_core_power = np.sum(self.core_loads[:self.p_cores]) * 1.5  # P-cores use more power
        e_core_power = np.sum(self.core_loads[self.p_cores:]) * 0.8  # E-cores are efficient
        self.power_consumption = min(1.0, (p_core_power + e_core_power) / 10.0)
        
        # Performance score
        self.performance_score = np.mean(self.core_loads[:self.p_cores]) * 0.7 + np.mean(self.core_loads[self.p_cores:]) * 0.3
        
        # Battery drain (simplified)
        self.battery_level = max(0.0, self.battery_level - self.power_consumption * 0.001)

class DQNScheduler:
    """Deep Q-Network based scheduler"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        # Line 136-155: Initialize DQN
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
    def _build_model(self) -> tf.keras.Model:
        # Line 156-175: Build neural network model
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model 
       
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        # Line 176-180: Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state: np.ndarray) -> int:
        # Line 181-190: Choose action using epsilon-greedy policy
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
        
    def replay(self):
        # Line 191-220: Train the model on a batch of experiences
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q values
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
                
        # Train the model
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        # Line 221-225: Update target network weights
        self.target_network.set_weights(self.q_network.get_weights())
        
    def save_model(self, filepath: str):
        # Line 226-230: Save trained model
        self.q_network.save(filepath)
        
    def load_model(self, filepath: str):
        # Line 231-235: Load trained model
        if os.path.exists(filepath):
            self.q_network = tf.keras.models.load_model(filepath)
            self.update_target_network()

class RLSchedulerTrainer:
    """Trainer for the RL scheduler"""
    
    def __init__(self):
        self.env = RLSchedulerEnvironment()
        self.agent = DQNScheduler(self.env.state_size, self.env.action_size)
        
    def train(self, episodes: int = 1000):
        """Train the RL scheduler"""
        print(f"🤖 Training RL Scheduler for {episodes} episodes...")
        
        scores = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            # Simulate process scheduling for one episode
            for step in range(100):  # Max steps per episode
                # Generate random process characteristics
                process_load = np.random.uniform(0.1, 1.0)
                process_priority = np.random.uniform(0.0, 1.0)
                
                # Choose action
                action = self.agent.act(state)
                
                # Take step in environment
                next_state, reward, done = self.env.step(action, process_load, process_priority)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            scores.append(total_reward)
            
            # Train the agent
            if len(self.agent.memory) > self.agent.batch_size:
                self.agent.replay()
                
            # Update target network periodically
            if episode % 100 == 0:
                self.agent.update_target_network()
                
            # Print progress
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.agent.epsilon:.3f}")
                
        print("🎯 Training completed!")
        return scores
        
    def test_scheduler(self, num_tests: int = 10):
        """Test the trained scheduler"""
        print(f"🧪 Testing RL Scheduler with {num_tests} test cases...")
        
        test_scores = []
        
        for test in range(num_tests):
            state = self.env.reset()
            total_reward = 0
            
            # Disable exploration for testing
            original_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.0
            
            for step in range(50):
                process_load = np.random.uniform(0.1, 1.0)
                process_priority = np.random.uniform(0.0, 1.0)
                
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action, process_load, process_priority)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            test_scores.append(total_reward)
            self.agent.epsilon = original_epsilon
            
        avg_test_score = np.mean(test_scores)
        print(f"📊 Average Test Score: {avg_test_score:.2f}")
        print(f"📈 Test Score Range: {min(test_scores):.2f} to {max(test_scores):.2f}")
        
        return test_scores

# Test function
def test_rl_scheduler():
    """Test the RL scheduler"""
    print("🤖 Testing RL Scheduler")
    print("=" * 50)
    
    # Create and train scheduler
    trainer = RLSchedulerTrainer()
    
    # Quick training run
    scores = trainer.train(episodes=200)
    
    # Test the trained scheduler
    test_scores = trainer.test_scheduler(num_tests=5)
    
    # Save the trained model
    trainer.agent.save_model("rl_scheduler_model.h5")
    print("💾 Model saved to rl_scheduler_model.h5")
    
    # Show training progress
    if len(scores) >= 10:
        print(f"\n📈 Training Progress:")
        print(f"  Initial 10 episodes avg: {np.mean(scores[:10]):.2f}")
        print(f"  Final 10 episodes avg: {np.mean(scores[-10:]):.2f}")
        print(f"  Improvement: {np.mean(scores[-10:]) - np.mean(scores[:10]):.2f}")

if __name__ == "__main__":
    test_rl_scheduler()