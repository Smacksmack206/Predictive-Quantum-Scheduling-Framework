#!/usr/bin/env python3
"""
Enhanced Process Analyzer with ML Integration
Combines ML classification, behavior prediction, and context awareness
"""

# Line 1-25: Enhanced analyzer with ML integration
from ml_process_classifier import MLProcessClassifier, ProcessFeatures
from behavior_predictor import ProcessBehaviorPredictor
from context_analyzer import ContextAnalyzer, SystemContext
import psutil
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class EnhancedProcessIntelligence:
    # Basic info
    pid: int
    name: str
    cpu_usage: float = 0.0
    memory_mb: float = 0.0
    
    # ML-based classification
    ml_classification: str = "unknown"
    ml_confidence: float = 0.0
    
    # Behavior prediction
    predicted_cpu_usage: List[float] = None
    predicted_memory_usage: List[float] = None
    prediction_confidence: float = 0.0
    
    # Context awareness
    context_priority_boost: float = 0.0
    user_interaction_score: float = 0.0
    
    # Advanced metrics
    energy_efficiency_score: float = 0.0
    thermal_impact_score: float = 0.0
    
    # Timestamps
    last_updated: Optional[datetime] = None
    
class EnhancedProcessAnalyzer:
    def __init__(self, db_path: str):
        self.db_path = os.path.expanduser(db_path)
        self.ml_classifier = MLProcessClassifier()
        self.behavior_predictor = ProcessBehaviorPredictor()
        self.context_analyzer = ContextAnalyzer()
        self.process_cache = {}
        
        # Performance tracking
        self.analysis_times = []
        
    def analyze_process_enhanced(self, pid: int, name: str) -> EnhancedProcessIntelligence:
        # Line 26-50: Main analysis method
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{name}_{pid}"
        if cache_key in self.process_cache:
            cached = self.process_cache[cache_key]
            if cached.last_updated and (datetime.now() - cached.last_updated).seconds < 30:
                return cached
                
        # Create intelligence object
        intel = EnhancedProcessIntelligence(
            pid=pid,
            name=name,
            last_updated=datetime.now()
        )
        
        try:
            # Get basic process info
            proc = psutil.Process(pid)
            intel.cpu_usage = proc.cpu_percent(interval=0)
            intel.memory_mb = proc.memory_info().rss / (1024 * 1024)
            
            # ML-based classification
            intel.ml_classification, intel.ml_confidence = self.ml_classifier.classify_process(pid, name)
            
            # Behavior prediction
            features = self.ml_classifier.feature_extractor.extract_features(pid)
            if features:
                predictions = self.behavior_predictor.predict_behavior(features)
                intel.predicted_cpu_usage = predictions['cpu_prediction']
                intel.predicted_memory_usage = predictions['memory_prediction']
                intel.prediction_confidence = predictions['confidence']
                
            # Context-aware priority adjustment
            system_context = self.context_analyzer.get_system_context()
            intel.context_priority_boost = self._calculate_context_boost(intel, system_context)
            
            # Advanced scoring
            intel.energy_efficiency_score = self._calculate_energy_efficiency(intel)
            intel.thermal_impact_score = self._calculate_thermal_impact(intel)
            intel.user_interaction_score = self._calculate_user_interaction_score(intel, system_context)
            
            # Cache result
            self.process_cache[cache_key] = intel
            
            # Track performance
            analysis_time = time.time() - start_time
            self.analysis_times.append(analysis_time)
            
            return intel
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Handle system processes
            intel.ml_classification = "system_process"
            intel.ml_confidence = 0.8
            intel.energy_efficiency_score = 0.9  # System processes are usually efficient
            self.process_cache[cache_key] = intel
            return intel
            
    def _calculate_context_boost(self, intel: EnhancedProcessIntelligence, 
                                context: SystemContext) -> float:
        # Line 51-80: Context-aware priority boost calculation
        boost = 0.0
        
        # Meeting boost
        if context.meeting_in_progress:
            meeting_apps = ['zoom', 'teams', 'webex', 'skype', 'facetime']
            if any(app in intel.name.lower() for app in meeting_apps):
                boost += 0.5  # High boost for meeting apps during meetings
                
        # Focus level boost
        if context.user_focus_level > 0.7:
            # Boost interactive applications during high focus
            if intel.ml_classification in ['interactive_application', 'development_tool']:
                boost += 0.3
                
        # Workflow phase boost
        if context.workflow_phase == 'development':
            dev_tools = ['xcode', 'vscode', 'pycharm', 'terminal', 'git']
            if any(tool in intel.name.lower() for tool in dev_tools):
                boost += 0.4
        elif context.workflow_phase == 'creative_work':
            creative_tools = ['photoshop', 'illustrator', 'blender', 'final cut']
            if any(tool in intel.name.lower() for tool in creative_tools):
                boost += 0.4
                
        # Battery level consideration
        if context.battery_level < 20:
            # Reduce boost when battery is low
            boost *= 0.5
        elif context.battery_level < 50:
            boost *= 0.8
            
        # Thermal state consideration
        if context.thermal_state == "hot":
            boost *= 0.6  # Reduce boost when system is hot
        elif context.thermal_state == "warm":
            boost *= 0.8
            
        return min(1.0, boost)
        
    def _calculate_energy_efficiency(self, intel: EnhancedProcessIntelligence) -> float:
        # Line 81-100: Energy efficiency scoring
        try:
            # Base efficiency on CPU usage vs utility
            if intel.cpu_usage == 0:
                return 1.0  # Idle processes are very efficient
                
            # Calculate efficiency based on classification and usage
            efficiency_map = {
                'system_process': 0.9,
                'background_service': 0.8,
                'interactive_application': 0.6,
                'development_tool': 0.5,
                'compute_intensive': 0.3,
                'unknown': 0.5
            }
            
            base_efficiency = efficiency_map.get(intel.ml_classification, 0.5)
            
            # Adjust based on CPU usage
            if intel.cpu_usage > 80:
                efficiency_penalty = 0.3
            elif intel.cpu_usage > 50:
                efficiency_penalty = 0.1
            else:
                efficiency_penalty = 0.0
                
            return max(0.1, base_efficiency - efficiency_penalty)
            
        except:
            return 0.5
            
    def _calculate_thermal_impact(self, intel: EnhancedProcessIntelligence) -> float:
        # Line 101-120: Thermal impact calculation
        try:
            # Thermal impact based on CPU usage and process type
            base_impact = intel.cpu_usage / 100.0
            
            # Multiply by process-specific thermal factors
            thermal_factors = {
                'compute_intensive': 1.5,  # High thermal impact
                'development_tool': 1.2,
                'interactive_application': 1.0,
                'background_service': 0.8,
                'system_process': 0.6,
                'unknown': 1.0
            }
            
            factor = thermal_factors.get(intel.ml_classification, 1.0)
            thermal_impact = base_impact * factor
            
            # Consider memory usage (high memory can also generate heat)
            if intel.memory_mb > 1000:  # > 1GB
                thermal_impact += 0.1
            elif intel.memory_mb > 500:  # > 500MB
                thermal_impact += 0.05
                
            return min(1.0, thermal_impact)
            
        except:
            return 0.5
            
    def _calculate_user_interaction_score(self, intel: EnhancedProcessIntelligence, 
                                        context: SystemContext) -> float:
        # Line 121-150: User interaction scoring
        try:
            base_score = 0.0
            
            # Classification-based scoring
            interaction_scores = {
                'interactive_application': 0.9,
                'development_tool': 0.8,
                'communication_app': 0.9,
                'browser': 0.7,
                'background_service': 0.1,
                'system_process': 0.1,
                'unknown': 0.3
            }
            
            base_score = interaction_scores.get(intel.ml_classification, 0.3)
            
            # Context adjustments
            if context.meeting_in_progress:
                meeting_apps = ['zoom', 'teams', 'webex', 'skype', 'facetime']
                if any(app in intel.name.lower() for app in meeting_apps):
                    base_score = 0.95  # Maximum priority during meetings
                    
            # Focus level adjustment
            if context.user_focus_level > 0.8:
                if intel.ml_classification in ['interactive_application', 'development_tool']:
                    base_score += 0.1
                    
            # Time of day adjustment
            if context.time_of_day in ['morning', 'afternoon']:
                base_score += 0.05  # Slight boost during work hours
                
            return min(1.0, base_score)
            
        except:
            return 0.5
            
    def get_performance_stats(self) -> Dict:
        # Line 151-165: Performance monitoring
        if not self.analysis_times:
            return {'avg_analysis_time': 0, 'total_analyses': 0}
            
        return {
            'avg_analysis_time': sum(self.analysis_times) / len(self.analysis_times),
            'max_analysis_time': max(self.analysis_times),
            'min_analysis_time': min(self.analysis_times),
            'total_analyses': len(self.analysis_times),
            'cache_hit_rate': len(self.process_cache) / len(self.analysis_times) if self.analysis_times else 0
        }

# Test function
def test_enhanced_analyzer():
    """Test the enhanced process analyzer"""
    print("🧠 Testing Enhanced Process Analyzer")
    print("=" * 50)
    
    analyzer = EnhancedProcessAnalyzer("~/.advanced_eas_test.db")
    
    # Test on current processes
    process_count = 0
    for proc in psutil.process_iter(['pid', 'name']):
        if process_count >= 10:  # Test first 10 processes
            break
            
        try:
            pid = proc.info['pid']
            name = proc.info['name']
            
            if pid > 10 and name:
                intel = analyzer.analyze_process_enhanced(pid, name)
                
                print(f"📊 Process: {name}")
                print(f"  ML Classification: {intel.ml_classification} (confidence: {intel.ml_confidence:.3f})")
                print(f"  CPU Usage: {intel.cpu_usage:.1f}%")
                print(f"  Memory: {intel.memory_mb:.1f} MB")
                print(f"  Energy Efficiency: {intel.energy_efficiency_score:.3f}")
                print(f"  Thermal Impact: {intel.thermal_impact_score:.3f}")
                print(f"  User Interaction: {intel.user_interaction_score:.3f}")
                print(f"  Context Boost: {intel.context_priority_boost:.3f}")
                
                if intel.predicted_cpu_usage:
                    avg_predicted_cpu = sum(intel.predicted_cpu_usage) / len(intel.predicted_cpu_usage)
                    print(f"  Predicted CPU (avg): {avg_predicted_cpu:.1f}%")
                
                print()
                process_count += 1
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Show performance stats
    stats = analyzer.get_performance_stats()
    print(f"⚡ Performance Stats:")
    print(f"  Average Analysis Time: {stats['avg_analysis_time']*1000:.2f}ms")
    print(f"  Total Analyses: {stats['total_analyses']}")
    print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.2f}")

if __name__ == "__main__":
    test_enhanced_analyzer()