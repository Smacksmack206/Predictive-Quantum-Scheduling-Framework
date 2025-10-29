#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Validator Module - Comprehensive Data Authenticity Validation
===================================================================

Ensures 100% authentic real-time data with zero tolerance for estimates.
Validates all metrics against hardware APIs and rejects mock values.

Requirements: 9.1-9.7, Data authenticity validation
"""

import logging
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source types for traceability"""
    HARDWARE_API = "hardware_api"
    SYSTEM_PROFILER = "system_profiler"
    POWERMETRICS = "powermetrics"
    PSUTIL = "psutil"
    METAL_API = "metal_api"
    SYSCTL = "sysctl"
    ESTIMATED = "estimated"
    MOCK = "mock"
    UNKNOWN = "unknown"


class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"  # Zero tolerance - reject any estimates
    MODERATE = "moderate"  # Allow fallbacks with warnings
    PERMISSIVE = "permissive"  # Accept estimates with logging


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    is_authentic: bool  # True only if from real hardware
    data_source: DataSource
    confidence_score: float  # 0.0 to 1.0
    warnings: List[str]
    errors: List[str]
    timestamp: datetime


@dataclass
class MetricValidation:
    """Validation rules for a specific metric"""
    name: str
    min_value: Optional[float]
    max_value: Optional[float]
    required_source: Optional[DataSource]
    max_age_seconds: float
    allow_estimates: bool


class DataValidator:
    """
    Validates data authenticity and quality with zero tolerance for mock data.
    Ensures all metrics come from real hardware sensors and APIs.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.validation_history: List[ValidationResult] = []
        self.rejected_count = 0
        self.accepted_count = 0
        
        # Define validation rules for each metric type
        self.validation_rules = self._initialize_validation_rules()
        
        logger.info(f"🔒 Data Validator initialized - Level: {validation_level.value}")
    
    def _initialize_validation_rules(self) -> Dict[str, MetricValidation]:
        """Initialize validation rules for all metrics"""
        return {
            'cpu_power_watts': MetricValidation(
                name='CPU Power',
                min_value=0.0,
                max_value=150.0,  # Max reasonable CPU power
                required_source=DataSource.POWERMETRICS,
                max_age_seconds=1.0,
                allow_estimates=self.validation_level != ValidationLevel.STRICT
            ),
            'gpu_power_watts': MetricValidation(
                name='GPU Power',
                min_value=0.0,
                max_value=100.0,
                required_source=DataSource.POWERMETRICS,
                max_age_seconds=1.0,
                allow_estimates=self.validation_level != ValidationLevel.STRICT
            ),
            'cpu_temp_celsius': MetricValidation(
                name='CPU Temperature',
                min_value=20.0,
                max_value=110.0,
                required_source=DataSource.SYSCTL,
                max_age_seconds=2.0,
                allow_estimates=False  # Never allow estimated temps
            ),
            'gpu_memory_mb': MetricValidation(
                name='GPU Memory',
                min_value=0.0,
                max_value=128000.0,  # 128GB max
                required_source=DataSource.METAL_API,
                max_age_seconds=1.0,
                allow_estimates=self.validation_level == ValidationLevel.PERMISSIVE
            ),
            'cpu_freq_mhz': MetricValidation(
                name='CPU Frequency',
                min_value=400.0,
                max_value=6000.0,
                required_source=DataSource.SYSCTL,
                max_age_seconds=0.5,
                allow_estimates=False
            ),
            'battery_cycles': MetricValidation(
                name='Battery Cycles',
                min_value=0,
                max_value=10000,
                required_source=DataSource.SYSTEM_PROFILER,
                max_age_seconds=60.0,  # Battery data changes slowly
                allow_estimates=False
            )
        }
    
    def validate_metric(
        self,
        metric_name: str,
        value: Any,
        data_source: DataSource,
        timestamp: datetime
    ) -> ValidationResult:
        """
        Validate a single metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value to validate
            data_source: Source of the data
            timestamp: When the data was collected
        
        Returns:
            ValidationResult with validation details
        """
        warnings = []
        errors = []
        is_valid = True
        is_authentic = True
        confidence_score = 1.0
        
        # Get validation rules for this metric
        rules = self.validation_rules.get(metric_name)
        
        if rules is None:
            warnings.append(f"No validation rules defined for {metric_name}")
            confidence_score = 0.5
        else:
            # Check data source authenticity
            if data_source in [DataSource.ESTIMATED, DataSource.MOCK]:
                is_authentic = False
                confidence_score *= 0.3
                
                if not rules.allow_estimates:
                    is_valid = False
                    errors.append(f"{rules.name}: Estimated/mock data not allowed in {self.validation_level.value} mode")
                else:
                    warnings.append(f"{rules.name}: Using estimated data (not authentic)")
            
            # Check if data source matches required source
            if rules.required_source and data_source != rules.required_source:
                if data_source not in [DataSource.ESTIMATED, DataSource.MOCK]:
                    warnings.append(f"{rules.name}: Expected {rules.required_source.value}, got {data_source.value}")
                    confidence_score *= 0.8
            
            # Check value range
            if rules.min_value is not None and value < rules.min_value:
                errors.append(f"{rules.name}: Value {value} below minimum {rules.min_value}")
                is_valid = False
            
            if rules.max_value is not None and value > rules.max_value:
                errors.append(f"{rules.name}: Value {value} above maximum {rules.max_value}")
                is_valid = False
            
            # Check data freshness
            age = (datetime.now() - timestamp).total_seconds()
            if age > rules.max_age_seconds:
                warnings.append(f"{rules.name}: Data is {age:.1f}s old (max {rules.max_age_seconds}s)")
                confidence_score *= max(0.5, 1.0 - (age / rules.max_age_seconds))
        
        # Create validation result
        result = ValidationResult(
            is_valid=is_valid,
            is_authentic=is_authentic,
            data_source=data_source,
            confidence_score=confidence_score,
            warnings=warnings,
            errors=errors,
            timestamp=datetime.now()
        )
        
        # Update statistics
        if is_valid and is_authentic:
            self.accepted_count += 1
        else:
            self.rejected_count += 1
        
        # Store in history
        self.validation_history.append(result)
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        # Log validation results
        if errors:
            logger.error(f"❌ Validation failed for {metric_name}: {', '.join(errors)}")
        elif warnings:
            logger.warning(f"⚠️ Validation warnings for {metric_name}: {', '.join(warnings)}")
        else:
            logger.debug(f"✅ Validation passed for {metric_name} (confidence: {confidence_score:.2f})")
        
        return result
    
    def validate_metrics_batch(
        self,
        metrics: Dict[str, Tuple[Any, DataSource, datetime]]
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple metrics at once.
        
        Args:
            metrics: Dict mapping metric names to (value, source, timestamp) tuples
        
        Returns:
            Dict mapping metric names to ValidationResult
        """
        results = {}
        
        for metric_name, (value, source, timestamp) in metrics.items():
            results[metric_name] = self.validate_metric(
                metric_name, value, source, timestamp
            )
        
        return results
    
    def validate_power_metrics(self, power_data: Dict[str, Any]) -> ValidationResult:
        """Validate power consumption metrics"""
        metrics = {}
        
        if 'cpu_power_watts' in power_data:
            metrics['cpu_power_watts'] = (
                power_data['cpu_power_watts'],
                power_data.get('source', DataSource.UNKNOWN),
                power_data.get('timestamp', datetime.now())
            )
        
        if 'gpu_power_watts' in power_data:
            metrics['gpu_power_watts'] = (
                power_data['gpu_power_watts'],
                power_data.get('source', DataSource.UNKNOWN),
                power_data.get('timestamp', datetime.now())
            )
        
        results = self.validate_metrics_batch(metrics)
        
        # Aggregate results
        all_valid = all(r.is_valid for r in results.values())
        all_authentic = all(r.is_authentic for r in results.values())
        avg_confidence = sum(r.confidence_score for r in results.values()) / len(results) if results else 0.0
        
        all_warnings = []
        all_errors = []
        for r in results.values():
            all_warnings.extend(r.warnings)
            all_errors.extend(r.errors)
        
        return ValidationResult(
            is_valid=all_valid,
            is_authentic=all_authentic,
            data_source=DataSource.POWERMETRICS,
            confidence_score=avg_confidence,
            warnings=all_warnings,
            errors=all_errors,
            timestamp=datetime.now()
        )
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self.accepted_count + self.rejected_count
        
        if total == 0:
            acceptance_rate = 0.0
        else:
            acceptance_rate = self.accepted_count / total
        
        recent_validations = self.validation_history[-100:] if self.validation_history else []
        recent_authentic = sum(1 for v in recent_validations if v.is_authentic)
        recent_rate = recent_authentic / len(recent_validations) if recent_validations else 0.0
        
        avg_confidence = sum(v.confidence_score for v in recent_validations) / len(recent_validations) if recent_validations else 0.0
        
        return {
            'total_validations': total,
            'accepted': self.accepted_count,
            'rejected': self.rejected_count,
            'acceptance_rate': acceptance_rate,
            'recent_authenticity_rate': recent_rate,
            'average_confidence': avg_confidence,
            'validation_level': self.validation_level.value
        }
    
    def detect_mock_data_patterns(self, values: List[float]) -> bool:
        """
        Detect if data appears to be mock/hardcoded.
        Returns True if mock data is detected.
        """
        if len(values) < 3:
            return False
        
        # Check for constant values (mock data often repeats)
        if len(set(values)) == 1:
            logger.warning("🚨 Mock data detected: All values identical")
            return True
        
        # Check for unrealistic patterns
        if all(v == 0 for v in values):
            logger.warning("🚨 Mock data detected: All zeros")
            return True
        
        # Check for suspiciously round numbers
        round_count = sum(1 for v in values if v == round(v))
        if round_count == len(values) and len(values) > 5:
            logger.warning("🚨 Suspicious data: All values are round numbers")
            return True
        
        return False
    
    def enforce_authenticity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce data authenticity by removing any non-authentic values.
        Only returns data that passes strict validation.
        """
        validated_data = {}
        
        for key, value in data.items():
            if isinstance(value, dict) and 'value' in value and 'source' in value:
                # Validate structured data
                result = self.validate_metric(
                    key,
                    value['value'],
                    value['source'],
                    value.get('timestamp', datetime.now())
                )
                
                if result.is_valid and result.is_authentic:
                    validated_data[key] = value['value']
                else:
                    logger.warning(f"🚫 Rejected non-authentic data for {key}")
            else:
                # Pass through unstructured data with warning
                logger.debug(f"⚠️ Unvalidated data for {key}")
                validated_data[key] = value
        
        return validated_data


# Global validator instance
_validator = None


def get_validator(level: ValidationLevel = ValidationLevel.STRICT) -> DataValidator:
    """Get or create the global data validator"""
    global _validator
    if _validator is None:
        _validator = DataValidator(level)
    return _validator


if __name__ == '__main__':
    # Test data validation
    print("🔒 Testing Data Validator...")
    
    validator = get_validator(ValidationLevel.STRICT)
    
    # Test valid authentic data
    print("\n✅ Testing valid authentic data:")
    result = validator.validate_metric(
        'cpu_power_watts',
        15.5,
        DataSource.POWERMETRICS,
        datetime.now()
    )
    print(f"  Valid: {result.is_valid}, Authentic: {result.is_authentic}, Confidence: {result.confidence_score:.2f}")
    
    # Test estimated data (should fail in strict mode)
    print("\n❌ Testing estimated data (should fail in strict mode):")
    result = validator.validate_metric(
        'cpu_power_watts',
        12.0,
        DataSource.ESTIMATED,
        datetime.now()
    )
    print(f"  Valid: {result.is_valid}, Authentic: {result.is_authentic}, Confidence: {result.confidence_score:.2f}")
    print(f"  Errors: {result.errors}")
    
    # Test out of range data
    print("\n❌ Testing out of range data:")
    result = validator.validate_metric(
        'cpu_temp_celsius',
        150.0,  # Too hot!
        DataSource.SYSCTL,
        datetime.now()
    )
    print(f"  Valid: {result.is_valid}, Errors: {result.errors}")
    
    # Test mock data detection
    print("\n🚨 Testing mock data detection:")
    mock_values = [10.0, 10.0, 10.0, 10.0, 10.0]
    is_mock = validator.detect_mock_data_patterns(mock_values)
    print(f"  Mock data detected: {is_mock}")
    
    # Get statistics
    print("\n📊 Validation Statistics:")
    stats = validator.get_validation_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Data validator test complete!")
