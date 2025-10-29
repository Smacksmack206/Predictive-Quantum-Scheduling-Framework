#!/usr/bin/env python3
"""
Automatic Battery Protection Service
=====================================

Runs in background and automatically protects apps from excessive battery drain
Specifically optimized for Kiro and other Electron-based apps
"""

import time
import logging
import threading
from quantum_battery_guardian import get_guardian
from quantum_ml_persistence import get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoBatteryProtectionService:
    """
    Background service that continuously monitors and protects battery
    Uses quantum-ML to dynamically learn which apps need protection
    """
    
    def __init__(self, check_interval: int = 30):
        """
        Args:
            check_interval: Seconds between protection checks
        """
        self.guardian = get_guardian()
        self.db = get_database()  # Always get database
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        
        # DYNAMIC priority apps - learned from behavior, not hardcoded
        self.priority_apps = []  # Will be populated by ML
        self.app_impact_scores = {}  # Battery impact score per app
        self.app_usage_patterns = {}  # Usage patterns per app
        
        # PHASE 2: Enhanced pattern tracking
        self.pattern_observations = {}  # Count observations per pattern
        self.pattern_history = {}  # Time-stamped observation history
        self.pattern_metrics = {}  # Variance and consistency metrics
        self.validation_scores = {}  # Cross-validation accuracy
        
        # Statistics - MUST be initialized BEFORE loading from database
        self.stats = {
            'total_protections': 0,
            'total_savings': 0.0,
            'apps_protected': set(),
            'start_time': time.time(),
            'apps_learned': 0,
            'dynamic_priorities': 0
        }
        
        # Load learned priority apps from database (after stats is initialized)
        self._load_priority_apps_from_db()
    
    def _load_priority_apps_from_db(self):
        """Load dynamically learned priority apps from database"""
        if not self.db:
            logger.error("❌ Database not available - using empty priority list")
            return
        
        try:
            logger.info(f"🔍 Loading priority apps from database for architecture: {self.guardian.architecture}")
            
            # Query apps with high battery impact from database
            import sqlite3
            cursor = self.db.conn.cursor()
            
            # Get apps that have been optimized frequently (high impact)
            cursor.execute('''
                SELECT process_name, 
                       AVG(avg_energy_saved) as avg_impact,
                       SUM(times_applied) as frequency,
                       AVG(success_rate) as success
                FROM process_optimizations
                WHERE architecture = ?
                GROUP BY process_name
                HAVING frequency > 3 AND success > 0.5
                ORDER BY avg_impact DESC
                LIMIT 20
            ''', (self.guardian.architecture,))
            
            rows = cursor.fetchall()
            logger.info(f"📊 Query returned {len(rows)} rows from database")
            
            for row in rows:
                app_name = row[0]
                impact = row[1]
                frequency = row[2]
                success = row[3]
                
                # Calculate priority score
                priority_score = impact * frequency * success
                
                self.priority_apps.append(app_name)
                self.app_impact_scores[app_name] = {
                    'impact': impact,
                    'frequency': frequency,
                    'success_rate': success,
                    'priority_score': priority_score
                }
                logger.debug(f"  Loaded: {app_name} (impact: {impact:.1f}, freq: {frequency})")
            
            self.stats['apps_learned'] = len(self.priority_apps)
            
            if self.priority_apps:
                logger.info(f"✅ Loaded {len(self.priority_apps)} priority apps from ML database")
                logger.info(f"   Top 5: {', '.join(self.priority_apps[:5])}")
            else:
                logger.warning("⚠️ No learned priority apps found in database (query returned 0 results)")
                
        except Exception as e:
            logger.error(f"❌ Error loading priority apps from database: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.priority_apps = []
    
    def start(self):
        """Start the protection service"""
        if self.running:
            logger.warning("Service already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._protection_loop, daemon=True)
        self.thread.start()
        logger.info(f"🛡️ Auto Battery Protection Service started (check every {self.check_interval}s)")
    
    def stop(self):
        """Stop the protection service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("🛡️ Auto Battery Protection Service stopped")
    
    def _protection_loop(self):
        """Main protection loop with dynamic learning"""
        while self.running:
            try:
                # PHASE 1: Discover high-impact apps dynamically
                if self.stats['total_protections'] % 5 == 0:
                    self._discover_high_impact_apps()
                
                # PHASE 2: Apply protection (None = all apps, let guardian decide)
                # If we have learned priorities, use them; otherwise protect all
                target_apps = self.priority_apps if self.priority_apps else None
                
                result = self.guardian.apply_guardian_protection(
                    target_apps=target_apps
                )
                
                # PHASE 3: Learn from results
                self._learn_from_protection_results(result)
                
                # Update statistics
                self.stats['total_protections'] += 1
                self.stats['total_savings'] += result.get('estimated_savings', 0)
                
                strategies = result.get('strategies', [])
                logger.debug(f"Received {len(strategies)} strategies from guardian")
                
                for strategy in strategies:
                    app_name = strategy['app']
                    self.stats['apps_protected'].add(app_name)
                    
                    # Update app usage patterns
                    self._update_app_pattern(app_name, strategy)
                    
                    # Save to database for persistence
                    if self.db:
                        try:
                            pattern = strategy.get('pattern', 'unknown')
                            savings = strategy.get('savings', 0)
                            
                            self.db.save_process_optimization(
                                app_name.lower(),
                                pattern,
                                5,  # Default nice adjustment
                                savings,
                                self.guardian.architecture,
                                success=True
                            )
                            logger.info(f"💾 Saved optimization: {app_name} ({pattern}, {savings:.1f}% savings)")
                        except Exception as e:
                            logger.error(f"Could not save optimization for {app_name}: {e}")
                
                # Log if significant protection applied
                if result.get('apps_protected', 0) > 0:
                    logger.info(f"🛡️ Protected {result['apps_protected']} apps, "
                              f"saved ~{result['estimated_savings']:.1f}% energy")
                
                # PHASE 4: Adaptive learning and threshold adjustment
                if self.stats['total_protections'] % 10 == 0:
                    self.guardian.learn_and_adapt()
                    self._update_priority_apps()
                
                # PHASE 5: Proactive optimization
                if self.stats['total_protections'] % 3 == 0:
                    self._proactive_optimization()
                
                # PHASE 6: App standby monitoring (every 2 cycles)
                if self.stats['total_protections'] % 2 == 0:
                    self._monitor_and_standby()
                
                # Wait for next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                import traceback
                logger.error(f"Protection loop error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(self.check_interval)
    
    def _discover_high_impact_apps(self):
        """
        Dynamically discover apps with high battery impact
        Uses quantum-ML to analyze all running apps
        """
        try:
            import psutil
            
            # Scan all processes for battery impact
            high_impact_apps = []
            
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    # Safely get process info
                    if not proc.info or 'name' not in proc.info:
                        continue
                    
                    name = proc.info['name']
                    cpu = proc.cpu_percent(interval=0.1)
                    memory = proc.memory_percent()
                    
                    # Calculate impact score
                    impact_score = (cpu * 2.0) + (memory * 0.5)
                    
                    if impact_score > 5.0:  # Significant impact
                        high_impact_apps.append({
                            'name': name,
                            'impact': impact_score,
                            'cpu': cpu,
                            'memory': memory
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Update app impact scores
            for app in high_impact_apps:
                app_name = app['name'].lower()
                
                if app_name not in self.app_impact_scores:
                    self.app_impact_scores[app_name] = {
                        'impact': app['impact'],
                        'frequency': 1,
                        'success_rate': 1.0,
                        'priority_score': app['impact']
                    }
                else:
                    # Update running average
                    current = self.app_impact_scores[app_name]
                    current['frequency'] += 1
                    current['impact'] = (current['impact'] * 0.8) + (app['impact'] * 0.2)
                    current['priority_score'] = current['impact'] * current['frequency']
            
            logger.debug(f"🔍 Discovered {len(high_impact_apps)} high-impact apps")
            
        except Exception as e:
            logger.error(f"Error discovering high-impact apps: {e}")
    
    def _learn_from_protection_results(self, result: dict):
        """
        Learn from protection results to improve future optimizations
        Uses quantum-ML to identify patterns
        """
        if not result.get('strategies'):
            return
        
        for strategy in result['strategies']:
            app_name = strategy['app'].lower()
            pattern = strategy.get('pattern', 'unknown')
            savings = strategy.get('savings', 0)
            
            # Update app usage patterns
            if app_name not in self.app_usage_patterns:
                self.app_usage_patterns[app_name] = {
                    'patterns': [],
                    'avg_savings': 0.0,
                    'optimization_count': 0
                }
            
            app_pattern = self.app_usage_patterns[app_name]
            app_pattern['patterns'].append(pattern)
            app_pattern['optimization_count'] += 1
            
            # Update running average of savings
            count = app_pattern['optimization_count']
            app_pattern['avg_savings'] = (
                (app_pattern['avg_savings'] * (count - 1)) + savings
            ) / count
    
    def _update_app_pattern(self, app_name: str, strategy: dict):
        """
        Update app behavior pattern with PHASE 2 improvements
        
        PHASE 1:
        - Bayesian confidence scoring
        - Sample size requirements
        
        PHASE 2:
        - Time-weighted observations
        - Variance-based confidence
        - Cross-validation
        """
        pattern = strategy.get('pattern', 'unknown')
        app_lower = app_name.lower()
        current_time = time.time()
        
        # Initialize tracking structures
        if app_lower not in self.pattern_observations:
            self.pattern_observations[app_lower] = {}
        if app_lower not in self.pattern_history:
            self.pattern_history[app_lower] = []
        if app_lower not in self.pattern_metrics:
            self.pattern_metrics[app_lower] = {'recent_patterns': []}
        
        # Record observation with timestamp (PHASE 2)
        self.pattern_observations[app_lower][pattern] = \
            self.pattern_observations[app_lower].get(pattern, 0) + 1
        
        self.pattern_history[app_lower].append({
            'pattern': pattern,
            'timestamp': current_time,
            'weight': 1.0
        })
        
        # Keep recent patterns for variance calculation
        self.pattern_metrics[app_lower]['recent_patterns'].append(pattern)
        if len(self.pattern_metrics[app_lower]['recent_patterns']) > 50:
            self.pattern_metrics[app_lower]['recent_patterns'].pop(0)
        
        # Calculate comprehensive confidence
        confidence_data = self._calculate_comprehensive_confidence(app_lower, pattern)
        
        # Determine dominant pattern
        dominant_pattern = max(self.pattern_observations[app_lower], 
                              key=self.pattern_observations[app_lower].get)
        
        # Update pattern data
        if app_lower not in self.app_usage_patterns:
            self.app_usage_patterns[app_lower] = {}
        
        self.app_usage_patterns[app_lower].update({
            'dominant_pattern': dominant_pattern,
            'pattern_confidence': confidence_data['final_confidence'],
            'confidence_breakdown': confidence_data['breakdown'],
            'last_seen': current_time,
            'total_observations': sum(self.pattern_observations[app_lower].values()),
            'pattern_counts': dict(self.pattern_observations[app_lower]),
            'consistency_score': confidence_data['factors']['consistency'],
            'recency_factor': confidence_data['factors']['recency']
        })
    
    def _calculate_comprehensive_confidence(self, app_name: str, pattern: str) -> dict:
        """
        PHASE 2: Calculate confidence from multiple factors
        
        Combines:
        1. Bayesian probability
        2. Time-weighted observations
        3. Variance-based consistency
        4. Sample size requirements
        5. Cross-validation (if enough data)
        """
        # Factor 1: Bayesian probability
        bayesian_conf = self._calculate_bayesian_confidence(app_name, pattern)
        
        # Factor 2: Time-weighted observations
        time_weighted_conf = self._calculate_time_weighted_confidence(app_name, pattern)
        
        # Factor 3: Variance-adjusted (consistency)
        variance_conf = self._calculate_variance_confidence(app_name, pattern)
        
        # Factor 4: Sample size adjusted
        sample_conf = self._get_sample_size_factor(app_name)
        
        # Factor 5: Cross-validation (if enough data)
        validation_conf = self._calculate_validation_confidence(app_name, pattern)
        
        # Weighted combination
        weights = {
            'bayesian': 0.25,
            'time_weighted': 0.20,
            'variance': 0.20,
            'sample_size': 0.20,
            'validation': 0.15
        }
        
        final_confidence = (
            bayesian_conf * weights['bayesian'] +
            time_weighted_conf * weights['time_weighted'] +
            variance_conf * weights['variance'] +
            sample_conf * weights['sample_size'] +
            validation_conf * weights['validation']
        )
        
        # Calculate additional factors
        total_obs = sum(self.pattern_observations[app_name].values())
        consistency = self._calculate_consistency_score(app_name, pattern)
        recency = time_weighted_conf / bayesian_conf if bayesian_conf > 0 else 1.0
        
        return {
            'final_confidence': final_confidence,
            'breakdown': {
                'bayesian': bayesian_conf,
                'time_weighted': time_weighted_conf,
                'variance': variance_conf,
                'sample_size': sample_conf,
                'validation': validation_conf
            },
            'factors': {
                'observations': total_obs,
                'consistency': consistency,
                'recency': recency
            }
        }
    
    def _calculate_bayesian_confidence(self, app_name: str, pattern: str) -> float:
        """Calculate Bayesian posterior probability"""
        if app_name not in self.pattern_observations:
            return 0.3
        
        pattern_count = self.pattern_observations[app_name].get(pattern, 0)
        total_count = sum(self.pattern_observations[app_name].values())
        
        if total_count == 0:
            return 0.3
        
        # Bayesian prior
        prior = 2.0
        num_patterns = len(self.pattern_observations[app_name])
        
        # Posterior probability
        confidence = (pattern_count + prior) / (total_count + prior * num_patterns)
        
        return confidence
    
    def _calculate_time_weighted_confidence(self, app_name: str, pattern: str) -> float:
        """
        PHASE 2: Calculate confidence with time decay
        
        Recent observations weighted higher than old ones
        Decay rate: 10% per day
        """
        if app_name not in self.pattern_history:
            return 0.3
        
        current_time = time.time()
        decay_rate = 0.1  # 10% decay per day
        
        # Apply time decay to all observations
        pattern_weight = 0.0
        total_weight = 0.0
        
        for obs in self.pattern_history[app_name]:
            age_days = (current_time - obs['timestamp']) / 86400
            weight = obs['weight'] * (1.0 - decay_rate) ** age_days
            
            total_weight += weight
            if obs['pattern'] == pattern:
                pattern_weight += weight
        
        if total_weight == 0:
            return 0.3
        
        confidence = pattern_weight / total_weight
        return confidence
    
    def _calculate_variance_confidence(self, app_name: str, pattern: str) -> float:
        """
        PHASE 2: Adjust confidence based on variance
        
        High variance = inconsistent = lower confidence
        Low variance = consistent = higher confidence
        """
        if app_name not in self.pattern_metrics:
            return 0.3
        
        recent_patterns = self.pattern_metrics[app_name].get('recent_patterns', [])
        
        if len(recent_patterns) < 5:
            return 0.3
        
        # Calculate proportion matching pattern
        matches = sum(1 for p in recent_patterns if p == pattern)
        proportion = matches / len(recent_patterns)
        
        # Calculate variance
        variance = proportion * (1 - proportion)
        
        # Convert variance to consistency factor
        # variance ranges from 0 (perfect) to 0.25 (50/50 split)
        consistency_factor = 1.0 - (variance * 4)
        
        # Combine proportion with consistency
        adjusted_confidence = proportion * (0.5 + 0.5 * consistency_factor)
        
        return adjusted_confidence
    
    def _get_sample_size_factor(self, app_name: str) -> float:
        """Get confidence factor based on sample size"""
        if app_name not in self.pattern_observations:
            return 0.3
        
        total_obs = sum(self.pattern_observations[app_name].values())
        
        if total_obs < 5:
            return 0.3
        elif total_obs < 10:
            return 0.5
        elif total_obs < 20:
            return 0.7
        else:
            return 1.0
    
    def _calculate_validation_confidence(self, app_name: str, pattern: str) -> float:
        """
        PHASE 2: Cross-validation confidence
        
        Test predictions on held-out data
        """
        if app_name not in self.pattern_history:
            return 0.5  # Neutral
        
        history = self.pattern_history[app_name]
        
        if len(history) < 10:
            return 0.5  # Need at least 10 observations
        
        # Use last 20% as test set
        split_point = int(len(history) * 0.8)
        train_data = history[:split_point]
        test_data = history[split_point:]
        
        if len(test_data) == 0:
            return 0.5
        
        # Count patterns in training data
        train_counts = {}
        for obs in train_data:
            p = obs['pattern']
            train_counts[p] = train_counts.get(p, 0) + 1
        
        # Predict dominant pattern from training
        if not train_counts:
            return 0.5
        
        predicted_pattern = max(train_counts, key=train_counts.get)
        
        # Test on held-out data
        correct = sum(1 for obs in test_data if obs['pattern'] == predicted_pattern)
        accuracy = correct / len(test_data)
        
        # Store validation score
        if app_name not in self.validation_scores:
            self.validation_scores[app_name] = []
        
        self.validation_scores[app_name].append(accuracy)
        
        # Keep last 10 validation scores
        if len(self.validation_scores[app_name]) > 10:
            self.validation_scores[app_name].pop(0)
        
        # Return average validation accuracy
        avg_accuracy = sum(self.validation_scores[app_name]) / len(self.validation_scores[app_name])
        
        return avg_accuracy
    
    def _calculate_consistency_score(self, app_name: str, pattern: str) -> float:
        """Calculate how consistent the pattern is"""
        if app_name not in self.pattern_metrics:
            return 0.0
        
        recent = self.pattern_metrics[app_name].get('recent_patterns', [])
        
        if len(recent) < 5:
            return 0.0
        
        matches = sum(1 for p in recent if p == pattern)
        consistency = matches / len(recent)
        
        return consistency
    
    def _update_priority_apps(self):
        """
        Dynamically update priority apps list based on learned behavior
        Uses quantum-ML scoring to rank apps
        """
        if not self.app_impact_scores:
            return
        
        # Sort apps by priority score
        sorted_apps = sorted(
            self.app_impact_scores.items(),
            key=lambda x: x[1]['priority_score'],
            reverse=True
        )
        
        # Update priority list (top 15 apps)
        old_count = len(self.priority_apps)
        self.priority_apps = [app[0] for app in sorted_apps[:15]]
        new_count = len(self.priority_apps)
        
        if new_count != old_count:
            self.stats['dynamic_priorities'] += 1
            logger.info(f"🔄 Updated priority apps: {old_count} → {new_count}")
            logger.info(f"   Top 5: {', '.join(self.priority_apps[:5])}")
    
    def _proactive_optimization(self):
        """
        Proactively optimize apps based on predicted behavior
        Uses quantum-ML to predict which apps will become problematic
        """
        try:
            import psutil
            
            # Get current battery state
            try:
                battery = psutil.sensors_battery()
                if battery:
                    battery_level = battery.percent
                    on_battery = not battery.power_plugged
                else:
                    return  # Can't optimize without battery info
            except:
                return
            
            # Only proactive optimize on battery
            if not on_battery:
                return
            
            # Predict which apps will cause problems
            for app_name, pattern_data in self.app_usage_patterns.items():
                pattern = pattern_data.get('dominant_pattern', 'unknown')
                confidence = pattern_data.get('pattern_confidence', 0)
                
                # High confidence predictions
                if confidence > 0.7:
                    # Proactively throttle apps with problematic patterns
                    if pattern in ['chaotic', 'burst'] and battery_level < 50:
                        # Find and throttle this app
                        for proc in psutil.process_iter(['pid', 'name']):
                            try:
                                if proc.info['name'].lower() == app_name:
                                    # Proactive throttling
                                    current_nice = proc.nice()
                                    if current_nice < 10:
                                        proc.nice(min(19, current_nice + 5))
                                        logger.info(f"⚡ Proactively throttled {app_name} "
                                                  f"(pattern: {pattern}, confidence: {confidence:.1%})")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue
                                
        except Exception as e:
            logger.error(f"Proactive optimization error: {e}")
    
    def _monitor_and_standby(self):
        """
        Monitor apps and put idle ones into standby
        """
        try:
            result = self.guardian.monitor_and_standby_apps()
            
            if result.get('success') and result.get('apps_suspended', 0) > 0:
                logger.info(f"🛌 Put {result['apps_suspended']} idle apps into standby")
                
                # Track suspended apps
                for app in result.get('suspended_apps', []):
                    logger.debug(f"   Suspended: {app['name']} (method: {app['method']})")
        
        except Exception as e:
            logger.error(f"Standby monitoring error: {e}")
    
    def get_statistics(self) -> dict:
        """Get service statistics including dynamic learning metrics"""
        runtime = time.time() - self.stats['start_time']
        
        # Get top priority apps with scores
        top_priority_apps = []
        for app_name in self.priority_apps[:10]:
            if app_name in self.app_impact_scores:
                score_data = self.app_impact_scores[app_name]
                top_priority_apps.append({
                    'name': app_name,
                    'impact': score_data['impact'],
                    'frequency': score_data['frequency'],
                    'priority_score': score_data['priority_score']
                })
        
        # Get learned patterns (lowered threshold from 0.6 to 0.4 for better visibility)
        learned_patterns = {}
        for app_name, pattern_data in self.app_usage_patterns.items():
            confidence = pattern_data.get('pattern_confidence', 0)
            if confidence > 0.4:  # Show patterns with 40%+ confidence
                learned_patterns[app_name] = {
                    'pattern': pattern_data.get('dominant_pattern', 'unknown'),
                    'confidence': confidence,
                    'status': 'confident' if confidence >= 0.6 else 'learning',
                    'observations': pattern_data.get('total_observations', 0),
                    'consistency': pattern_data.get('consistency_score', 0),
                    'breakdown': pattern_data.get('confidence_breakdown', {})
                }
        
        return {
            'running': self.running,
            'runtime_minutes': runtime / 60,
            'total_protections': self.stats['total_protections'],
            'total_savings': self.stats['total_savings'],
            'apps_protected': list(self.stats['apps_protected']),
            'avg_savings_per_check': (
                self.stats['total_savings'] / self.stats['total_protections']
                if self.stats['total_protections'] > 0 else 0
            ),
            # Dynamic learning metrics
            'apps_learned': self.stats['apps_learned'],
            'dynamic_priorities': self.stats['dynamic_priorities'],
            'current_priority_apps': len(self.priority_apps),
            'top_priority_apps': top_priority_apps,
            'learned_patterns': learned_patterns,
            'total_apps_analyzed': len(self.app_impact_scores)
        }
    
    def get_app_insights(self, app_name: str) -> dict:
        """
        Get detailed insights about a specific app's behavior and optimization
        """
        app_lower = app_name.lower()
        
        insights = {
            'app_name': app_name,
            'is_priority': app_lower in self.priority_apps,
            'has_impact_data': app_lower in self.app_impact_scores,
            'has_pattern_data': app_lower in self.app_usage_patterns
        }
        
        if app_lower in self.app_impact_scores:
            insights['impact_data'] = self.app_impact_scores[app_lower]
        
        if app_lower in self.app_usage_patterns:
            insights['pattern_data'] = self.app_usage_patterns[app_lower]
        
        # Get recommendations
        recommendations = []
        
        if insights['has_pattern_data']:
            pattern = self.app_usage_patterns[app_lower].get('dominant_pattern')
            confidence = self.app_usage_patterns[app_lower].get('pattern_confidence', 0)
            
            if pattern == 'idle' and confidence > 0.7:
                recommendations.append("App is frequently idle - consider closing when not in use")
            elif pattern == 'chaotic' and confidence > 0.7:
                recommendations.append("App has erratic behavior - may benefit from restart")
            elif pattern == 'burst' and confidence > 0.7:
                recommendations.append("App has burst usage - normal for interactive apps")
        
        if insights['has_impact_data']:
            impact = self.app_impact_scores[app_lower]['impact']
            if impact > 15.0:
                recommendations.append(f"High battery impact ({impact:.1f}) - actively optimizing")
            elif impact > 8.0:
                recommendations.append(f"Moderate battery impact ({impact:.1f}) - monitoring")
        
        insights['recommendations'] = recommendations
        
        return insights


# Global service instance
_service_instance = None

def get_service() -> AutoBatteryProtectionService:
    """Get or create global service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = AutoBatteryProtectionService()
    return _service_instance


if __name__ == "__main__":
    # Test the service with dynamic learning
    print("🛡️ Starting Auto Battery Protection Service (Dynamic Learning)")
    print("=" * 70)
    
    service = get_service()
    
    print(f"\n📚 Initial State:")
    print(f"   Learned priority apps: {service.stats['apps_learned']}")
    if service.priority_apps:
        print(f"   Top 5 priorities: {', '.join(service.priority_apps[:5])}")
    else:
        print(f"   No learned priorities yet - will discover dynamically")
    
    service.start()
    
    try:
        # Run for 2 minutes as a test
        print("\n⏱️  Running for 2 minutes (press Ctrl+C to stop)...")
        print("   Watching for dynamic learning and proactive optimization...\n")
        
        for i in range(4):  # 4 checks over 2 minutes
            time.sleep(30)
            stats = service.get_statistics()
            
            print(f"\n{'='*70}")
            print(f"📊 Statistics after {stats['runtime_minutes']:.1f} minutes:")
            print(f"{'='*70}")
            print(f"   Total protections: {stats['total_protections']}")
            print(f"   Total savings: {stats['total_savings']:.1f}%")
            print(f"   Apps protected: {len(stats['apps_protected'])}")
            
            # Dynamic learning metrics
            print(f"\n🧠 Dynamic Learning:")
            print(f"   Apps learned: {stats['apps_learned']}")
            print(f"   Current priorities: {stats['current_priority_apps']}")
            print(f"   Priority updates: {stats['dynamic_priorities']}")
            print(f"   Total apps analyzed: {stats['total_apps_analyzed']}")
            
            # Show top priority apps
            if stats['top_priority_apps']:
                print(f"\n🎯 Top Priority Apps (Dynamic):")
                for app in stats['top_priority_apps'][:5]:
                    print(f"      {app['name'][:30]:30} "
                          f"Impact: {app['impact']:>6.1f} "
                          f"Freq: {app['frequency']:>3} "
                          f"Score: {app['priority_score']:>7.1f}")
            
            # Show learned patterns
            if stats['learned_patterns']:
                print(f"\n🔍 Learned Patterns:")
                for app_name, pattern_info in list(stats['learned_patterns'].items())[:5]:
                    print(f"      {app_name[:30]:30} "
                          f"{pattern_info['pattern']:10} "
                          f"({pattern_info['confidence']:.0%} confidence)")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping service...")
    
    finally:
        service.stop()
        final_stats = service.get_statistics()
        
        print(f"\n{'='*70}")
        print(f"✅ Final Statistics")
        print(f"{'='*70}")
        print(f"   Runtime: {final_stats['runtime_minutes']:.1f} minutes")
        print(f"   Total protections: {final_stats['total_protections']}")
        print(f"   Total savings: {final_stats['total_savings']:.1f}%")
        print(f"   Average savings: {final_stats['avg_savings_per_check']:.1f}% per check")
        print(f"\n🧠 Learning Summary:")
        print(f"   Apps learned: {final_stats['apps_learned']}")
        print(f"   Dynamic priorities: {final_stats['current_priority_apps']}")
        print(f"   Priority updates: {final_stats['dynamic_priorities']}")
        print(f"   Apps analyzed: {final_stats['total_apps_analyzed']}")
        
        # Test app insights for Kiro
        if 'kiro' in [app.lower() for app in final_stats['apps_protected']]:
            print(f"\n🔍 Kiro Insights:")
            kiro_insights = service.get_app_insights('Kiro')
            print(f"   Is priority: {kiro_insights['is_priority']}")
            if kiro_insights.get('impact_data'):
                impact = kiro_insights['impact_data']
                print(f"   Battery impact: {impact['impact']:.1f}")
                print(f"   Optimization frequency: {impact['frequency']}")
            if kiro_insights.get('recommendations'):
                print(f"   Recommendations:")
                for rec in kiro_insights['recommendations']:
                    print(f"      • {rec}")
