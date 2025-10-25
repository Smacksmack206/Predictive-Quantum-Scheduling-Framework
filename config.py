"""
PQS Framework Configuration
Centralized settings with validation and YAML support
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import json

@dataclass
class QuantumConfig:
    """Quantum engine configuration"""
    engine: str = 'qiskit'  # or 'cirq'
    max_qubits: int = 48
    optimization_interval: int = 10  # seconds
    use_quantum_max: bool = True
    fallback_to_classical: bool = True
    
@dataclass
class IdleConfig:
    """Idle management configuration"""
    suspend_delay: int = 30  # seconds
    sleep_delay: int = 120
    cpu_idle_threshold: float = 5.0  # percent
    enable_ml_prediction: bool = True
    aggressive_mode: bool = True
    
@dataclass
class BatteryConfig:
    """Battery management configuration"""
    critical_threshold: int = 20  # percent
    low_threshold: int = 40
    aggressive_mode: bool = True
    enable_guardian: bool = True
    auto_protection: bool = True

@dataclass
class UIConfig:
    """UI configuration"""
    theme: str = 'dark'
    enable_modern_ui: bool = True
    refresh_interval: int = 5  # seconds
    show_notifications: bool = True

@dataclass
class PQSConfig:
    """Main PQS Framework configuration"""
    quantum: QuantumConfig
    idle: IdleConfig
    battery: BatteryConfig
    ui: UIConfig
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'PQSConfig':
        """Load config from JSON or use defaults"""
        if config_path and config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        return cls.default()
    
    @classmethod
    def default(cls) -> 'PQSConfig':
        """Default configuration"""
        return cls(
            quantum=QuantumConfig(),
            idle=IdleConfig(),
            battery=BatteryConfig(),
            ui=UIConfig()
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PQSConfig':
        """Create config from dictionary"""
        return cls(
            quantum=QuantumConfig(**data.get('quantum', {})),
            idle=IdleConfig(**data.get('idle', {})),
            battery=BatteryConfig(**data.get('battery', {})),
            ui=UIConfig(**data.get('ui', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'quantum': asdict(self.quantum),
            'idle': asdict(self.idle),
            'battery': asdict(self.battery),
            'ui': asdict(self.ui)
        }
    
    def save(self, config_path: Path):
        """Save config to JSON"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

# Global config instance
config = PQSConfig.default()

def load_config(config_path: Optional[Path] = None) -> PQSConfig:
    """Load configuration from file or use defaults"""
    global config
    config = PQSConfig.load(config_path)
    return config
