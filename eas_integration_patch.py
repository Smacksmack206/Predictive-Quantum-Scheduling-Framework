#!/usr/bin/env python3
"""
EAS Integration Patch - Updates existing enhanced_app.py to use the new intelligent classifier
This patches the existing EnergyAwareScheduler class with enhanced capabilities
"""

import os
import sys
import time
import psutil
from enhanced_eas_classifier import DynamicProcessClassifier, EnhancedEASScheduler

class EASIntegrationPatch:
    """Patches existing EAS with enhanced classification"""
    
    def __init__(self, existing_eas_instance):
        self.original_eas = existing_eas_instance
        
        # Initialize enhanced classifier
        db_path = os.path.expanduser("~/.battery_optimizer_enhanced_eas.db")
        self.enhanced_scheduler = EnhancedEASScheduler(db_path)
        
        # Patch methods
        self._patch_classification_methods()
    
    def _patch_classification_methods(self):
        """Patch the original EAS methods with enhanced versions"""
        
        # Store original methods
        self.original_eas._original_classify_workload = self.original_eas.classify_workload
        self.original_eas._original_calculate_optimal_assignment = self.original_eas.calculate_optimal_assignment
        
        # Replace with enhanced methods
        self.original_eas.classify_workload = self._enhanced_classify_workload
        self.original_eas.calculate_optimal_assignment = self._enhanced_calculate_optimal_assignment
        
        # Add new methods
        self.original_eas.get_classification_insights = self._get_classification_insights
        self.original_eas.get_learning_stats = self._get_learning_stats
        self.original_eas.force_reclassify_all = self._force_reclassify_all
    
    def _enhanced_classify_workload(self, pid, name):
        """Enhanced workload classification using ML and behavioral analysis"""
        try:
            # Use the enhanced classifier
            classification, confidence = self.enhanced_scheduler.classifier.classify_process_intelligent(pid, name)
            
            # Map enhanced classifications to original EAS categories
            classification_mapping = {
                'system_critical': 'background',
                'user_facing': 'interactive',
                'interactive_heavy': 'interactive',
                'interactive_light': 'interactive_light',
                'compute_intensive': 'compute',
                'background_compute': 'compute',
                'background': 'background',
                'system_service': 'background',
                'io_intensive': 'compute',
                'network_intensive': 'background',
                'memory_intensive': 'compute',
                'high_priority': 'interactive',
                'medium_priority': 'interactive_light',
                'low_priority': 'background',
                'unknown_application': 'background'
            }
            
            mapped_classification = classification_mapping.get(classification, 'background')
            
            # Store enhanced data for later use
            if not hasattr(self.original_eas, '_enhanced_classifications'):
                self.original_eas._enhanced_classifications = {}
            
            self.original_eas._enhanced_classifications[pid] = {
                'original_classification': classification,
                'mapped_classification': mapped_classification,
                'confidence': confidence,
                'timestamp': time.time()
            }
            
            return mapped_classification
            
        except Exception as e:
            print(f"Enhanced classification error for {name}: {e}")
            # Fallback to original method
            return self.original_eas._original_classify_workload(pid, name)
    
    def _enhanced_calculate_optimal_assignment(self, pid, name):
        """Enhanced optimal assignment calculation"""
        try:
            # Get enhanced assignment
            enhanced_assignment = self.enhanced_scheduler.classify_and_assign(pid, name)
            
            # Convert to original format but with enhanced data
            original_assignment = {
                'pid': pid,
                'name': name,
                'workload_type': enhanced_assignment['classification'],
                'optimal_core': enhanced_assignment['target_core'],
                'cpu_usage': enhanced_assignment.get('cpu_usage', 0),
                'energy_p': 0,  # Will be calculated by original method
                'energy_e': 0,  # Will be calculated by original method
                
                # Enhanced fields
                'enhanced_classification': enhanced_assignment['classification'],
                'assignment_strategy': enhanced_assignment['strategy'],
                'confidence': enhanced_assignment['confidence'],
                'priority_adjustment': enhanced_assignment['priority_adjustment'],
                'system_load': enhanced_assignment.get('system_load', 0),
                'p_core_load': enhanced_assignment.get('p_core_load', 0),
                'e_core_load': enhanced_assignment.get('e_core_load', 0)
            }
            
            # Calculate energy costs using original method logic
            try:
                proc = psutil.Process(pid)
                cpu_usage = enhanced_assignment.get('cpu_usage', proc.cpu_percent(interval=0.1))
            except:
                cpu_usage = 5
            
            # Energy efficiency calculation (from original)
            p_energy = self.original_eas.energy_models['p_core']['base_power'] + \
                      (cpu_usage/100) * self.original_eas.energy_models['p_core']['power_per_mhz'] * 1000
            e_energy = self.original_eas.energy_models['e_core']['base_power'] + \
                      (cpu_usage/100) * self.original_eas.energy_models['e_core']['power_per_mhz'] * 1000
            
            original_assignment['energy_p'] = p_energy
            original_assignment['energy_e'] = e_energy
            
            return original_assignment
            
        except Exception as e:
            print(f"Enhanced assignment error for {name}: {e}")
            # Fallback to original method
            return self.original_eas._original_calculate_optimal_assignment(pid, name)
    
    def _get_classification_insights(self):
        """Get insights about current classifications"""
        stats = self.enhanced_scheduler.classifier.get_classification_stats()
        
        # Add current process insights
        current_classifications = getattr(self.original_eas, '_enhanced_classifications', {})
        
        insights = {
            'total_processes_classified': len(current_classifications),
            'classification_stats': stats,
            'confidence_distribution': self._calculate_confidence_distribution(current_classifications),
            'classification_breakdown': self._calculate_classification_breakdown(current_classifications),
            'learning_effectiveness': self._calculate_learning_effectiveness(stats)
        }
        
        return insights
    
    def _calculate_confidence_distribution(self, classifications):
        """Calculate confidence score distribution"""
        if not classifications:
            return {}
        
        confidences = [data['confidence'] for data in classifications.values()]
        
        return {
            'high_confidence': len([c for c in confidences if c > 0.8]),
            'medium_confidence': len([c for c in confidences if 0.5 <= c <= 0.8]),
            'low_confidence': len([c for c in confidences if c < 0.5]),
            'average_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences)
        }
    
    def _calculate_classification_breakdown(self, classifications):
        """Calculate breakdown of classification types"""
        if not classifications:
            return {}
        
        breakdown = {}
        for data in classifications.values():
            original_class = data['original_classification']
            breakdown[original_class] = breakdown.get(original_class, 0) + 1
        
        return breakdown
    
    def _calculate_learning_effectiveness(self, stats):
        """Calculate how effective the learning system is"""
        total_classifications = stats.get('total_classifications', 0)
        avg_confidence = stats.get('average_confidence', 0)
        
        if total_classifications == 0:
            return 0.0
        
        # Effectiveness based on volume and confidence
        volume_score = min(total_classifications / 1000, 1.0)  # Normalize to 1000 classifications
        confidence_score = avg_confidence
        
        return (volume_score * 0.4 + confidence_score * 0.6)
    
    def _get_learning_stats(self):
        """Get detailed learning statistics"""
        return self.enhanced_scheduler.classifier.get_classification_stats()
    
    def _force_reclassify_all(self):
        """Force reclassification of all current processes"""
        reclassified = 0
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                pid = proc.info['pid']
                name = proc.info['name']
                
                if pid > 50:  # Skip system processes
                    self._enhanced_classify_workload(pid, name)
                    reclassified += 1
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return reclassified

def patch_existing_eas(eas_instance):
    """Patch an existing EAS instance with enhanced capabilities"""
    return EASIntegrationPatch(eas_instance)

# Integration code for enhanced_app.py
def integrate_enhanced_eas():
    """
    Integration function to be called from enhanced_app.py
    
    Add this to your enhanced_app.py file:
    
    # At the top, add import
    from eas_integration_patch import integrate_enhanced_eas, patch_existing_eas
    
    # In the EnergyAwareScheduler.__init__ method, add at the end:
    self.enhanced_patch = None
    
    # Add this method to EnergyAwareScheduler class:
    def enable_enhanced_classification(self):
        if self.enhanced_patch is None:
            from eas_integration_patch import patch_existing_eas
            self.enhanced_patch = patch_existing_eas(self)
            print("🧠 Enhanced EAS classification enabled!")
        return self.enhanced_patch
    
    # Add these API endpoints to your Flask app:
    
    @app.route('/api/eas-insights')
    def eas_insights():
        if hasattr(eas, 'enhanced_patch') and eas.enhanced_patch:
            insights = eas.enhanced_patch._get_classification_insights()
            return jsonify(insights)
        return jsonify({'error': 'Enhanced EAS not enabled'})
    
    @app.route('/api/eas-learning-stats')
    def eas_learning_stats():
        if hasattr(eas, 'enhanced_patch') and eas.enhanced_patch:
            stats = eas.enhanced_patch._get_learning_stats()
            return jsonify(stats)
        return jsonify({'error': 'Enhanced EAS not enabled'})
    
    @app.route('/api/eas-reclassify', methods=['POST'])
    def eas_reclassify():
        if hasattr(eas, 'enhanced_patch') and eas.enhanced_patch:
            count = eas.enhanced_patch._force_reclassify_all()
            return jsonify({'reclassified_processes': count})
        return jsonify({'error': 'Enhanced EAS not enabled'})
    """
    
    integration_code = '''
# Enhanced EAS Integration Code
# Add this to your enhanced_app.py file

# 1. Add import at the top
from eas_integration_patch import patch_existing_eas

# 2. In EnergyAwareScheduler.__init__, add:
self.enhanced_patch = None

# 3. Add this method to EnergyAwareScheduler class:
def enable_enhanced_classification(self):
    """Enable enhanced ML-based classification"""
    if self.enhanced_patch is None:
        try:
            self.enhanced_patch = patch_existing_eas(self)
            print("🧠 Enhanced EAS classification enabled!")
            return True
        except Exception as e:
            print(f"Failed to enable enhanced EAS: {e}")
            return False
    return True

# 4. Add these Flask API endpoints:

@app.route('/api/eas-insights')
def eas_insights():
    """Get EAS classification insights"""
    if hasattr(eas, 'enhanced_patch') and eas.enhanced_patch:
        try:
            insights = eas.enhanced_patch._get_classification_insights()
            return jsonify(insights)
        except Exception as e:
            return jsonify({'error': str(e)})
    return jsonify({'error': 'Enhanced EAS not enabled'})

@app.route('/api/eas-learning-stats')
def eas_learning_stats():
    """Get EAS learning statistics"""
    if hasattr(eas, 'enhanced_patch') and eas.enhanced_patch:
        try:
            stats = eas.enhanced_patch._get_learning_stats()
            return jsonify(stats)
        except Exception as e:
            return jsonify({'error': str(e)})
    return jsonify({'error': 'Enhanced EAS not enabled'})

@app.route('/api/eas-reclassify', methods=['POST'])
def eas_reclassify():
    """Force reclassification of all processes"""
    if hasattr(eas, 'enhanced_patch') and eas.enhanced_patch:
        try:
            count = eas.enhanced_patch._force_reclassify_all()
            return jsonify({'reclassified_processes': count, 'success': True})
        except Exception as e:
            return jsonify({'error': str(e), 'success': False})
    return jsonify({'error': 'Enhanced EAS not enabled', 'success': False})

@app.route('/api/eas-enable-enhanced', methods=['POST'])
def eas_enable_enhanced():
    """Enable enhanced EAS classification"""
    try:
        success = eas.enable_enhanced_classification()
        return jsonify({'success': success, 'enabled': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# 5. In your main optimization loop, enable enhanced EAS:
# Add this after creating the EAS instance:
if config.get('enhanced_eas_enabled', True):
    eas.enable_enhanced_classification()
'''
    
    return integration_code

if __name__ == "__main__":
    print("EAS Integration Patch")
    print("=" * 30)
    print(integrate_enhanced_eas())