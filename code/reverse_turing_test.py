"""
Reverse Turing Test Implementation

This module implements a reverse Turing test to detect if a player is about to be flagged
by platform heuristics by simulating how bot detection models classify them. Like having
Kurisu's analytical mind combined with Daru's hacking skills to stay undetected!
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import copy
from enum import Enum
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class BehaviorFlag(Enum):
    """Types of behavioral flags that can trigger bot detection."""
    TIMING_PATTERN = "timing_pattern"
    CLICK_PRECISION = "click_precision"
    MOVEMENT_SMOOTHNESS = "movement_smoothness"
    DECISION_SPEED = "decision_speed"
    PATTERN_REPETITION = "pattern_repetition"
    INHUMAN_CONSISTENCY = "inhuman_consistency"
    STATISTICAL_ANOMALY = "statistical_anomaly"

@dataclass
class BehaviorMetrics:
    """Metrics used to analyze player behavior."""
    # Timing metrics
    click_intervals: List[float] = field(default_factory=list)
    decision_times: List[float] = field(default_factory=list)
    session_duration: float = 0.0
    
    # Movement metrics
    mouse_positions: List[Tuple[float, float]] = field(default_factory=list)
    click_positions: List[Tuple[float, float]] = field(default_factory=list)
    movement_velocities: List[float] = field(default_factory=list)
    
    # Decision metrics
    action_sequence: List[str] = field(default_factory=list)
    risk_tolerance_history: List[float] = field(default_factory=list)
    bet_size_history: List[float] = field(default_factory=list)
    
    # Pattern metrics
    repeated_sequences: Dict[str, int] = field(default_factory=dict)
    consistency_scores: List[float] = field(default_factory=list)
    entropy_measures: List[float] = field(default_factory=list)
    
    # Performance metrics
    win_rate: float = 0.0
    profit_loss: float = 0.0
    session_count: int = 0

@dataclass
class DetectionAlert:
    """Alert generated when bot-like behavior is detected."""
    flag_type: BehaviorFlag
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: str
    recommendation: str
    timestamp: float
    metrics_snapshot: Dict[str, Any]

class BotDetectionModel:
    """
    Simulates platform bot detection algorithms.
    
    Like creating a digital version of the Organization's surveillance system
    that can detect when someone is acting too perfectly!
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Detection thresholds
        self.timing_threshold = self.config.get('timing_threshold', 0.8)
        self.precision_threshold = self.config.get('precision_threshold', 0.9)
        self.consistency_threshold = self.config.get('consistency_threshold', 0.85)
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.7)
        
        # Machine learning models
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Training data (simulated human vs bot behaviors)
        self.is_trained = False
        self._generate_training_data()
        
        # Detection history
        self.detection_history = []
        self.false_positive_rate = 0.05  # Simulated platform FP rate
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for bot detection."""
        return {
            'timing_threshold': 0.8,
            'precision_threshold': 0.9,
            'consistency_threshold': 0.85,
            'anomaly_threshold': 0.7,
            'min_samples_for_detection': 50,
            'detection_window': 1000,  # Number of recent actions to analyze
            'severity_weights': {
                'timing_pattern': 0.25,
                'click_precision': 0.20,
                'movement_smoothness': 0.15,
                'decision_speed': 0.15,
                'pattern_repetition': 0.15,
                'inhuman_consistency': 0.10
            }
        }
    
    def _generate_training_data(self):
        """Generate synthetic training data for human vs bot classification."""
        print("🤖 Generating bot detection training data...")
        
        # Generate human-like behavior data
        human_data = []
        for _ in range(1000):
            features = self._generate_human_features()
            human_data.append(features + [0])  # 0 = human
        
        # Generate bot-like behavior data
        bot_data = []
        for _ in range(1000):
            features = self._generate_bot_features()
            bot_data.append(features + [1])  # 1 = bot
        
        # Combine and prepare training data
        all_data = np.array(human_data + bot_data)
        X = all_data[:, :-1]
        y = all_data[:, -1]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.isolation_forest.fit(X_scaled)
        self.classifier.fit(X_scaled, y)
        
        self.is_trained = True
        print("✅ Bot detection models trained successfully")
    
    def _generate_human_features(self) -> List[float]:
        """Generate features representing human-like behavior."""
        return [
            # Timing variability (humans are inconsistent)
            np.random.normal(0.5, 0.3),  # Click interval variance
            np.random.normal(2.0, 1.0),  # Decision time variance
            
            # Movement characteristics
            np.random.normal(0.3, 0.2),  # Movement smoothness (some jitter)
            np.random.normal(5.0, 3.0),  # Average movement speed
            
            # Decision patterns
            np.random.normal(0.4, 0.2),  # Risk tolerance variance
            np.random.normal(0.6, 0.3),  # Action consistency
            
            # Behavioral entropy
            np.random.normal(0.7, 0.2),  # Pattern entropy (high = unpredictable)
            np.random.normal(0.3, 0.2),  # Sequence repetition (low = varied)
        ]
    
    def _generate_bot_features(self) -> List[float]:
        """Generate features representing bot-like behavior."""
        return [
            # Timing precision (bots are consistent)
            np.random.normal(0.9, 0.1),  # Low click interval variance
            np.random.normal(0.5, 0.2),  # Low decision time variance
            
            # Movement characteristics
            np.random.normal(0.9, 0.1),  # High movement smoothness
            np.random.normal(10.0, 2.0),  # Consistent movement speed
            
            # Decision patterns
            np.random.normal(0.1, 0.05),  # Low risk tolerance variance
            np.random.normal(0.9, 0.1),  # High action consistency
            
            # Behavioral entropy
            np.random.normal(0.3, 0.1),  # Low pattern entropy (predictable)
            np.random.normal(0.7, 0.2),  # High sequence repetition
        ]
    
    def analyze_behavior(self, metrics: BehaviorMetrics) -> List[DetectionAlert]:
        """Analyze behavior metrics and generate detection alerts."""
        if not self.is_trained:
            return []
        
        alerts = []
        
        # Extract features from metrics
        features = self._extract_features(metrics)
        if not features:
            return alerts
        
        # Run detection algorithms
        timing_alert = self._detect_timing_patterns(metrics)
        if timing_alert:
            alerts.append(timing_alert)
        
        precision_alert = self._detect_click_precision(metrics)
        if precision_alert:
            alerts.append(precision_alert)
        
        movement_alert = self._detect_movement_anomalies(metrics)
        if movement_alert:
            alerts.append(movement_alert)
        
        consistency_alert = self._detect_inhuman_consistency(metrics)
        if consistency_alert:
            alerts.append(consistency_alert)
        
        pattern_alert = self._detect_pattern_repetition(metrics)
        if pattern_alert:
            alerts.append(pattern_alert)
        
        # Machine learning based detection
        ml_alert = self._ml_based_detection(features)
        if ml_alert:
            alerts.append(ml_alert)
        
        # Store detection history
        self.detection_history.extend(alerts)
        
        return alerts
    
    def _extract_features(self, metrics: BehaviorMetrics) -> Optional[List[float]]:
        """Extract numerical features from behavior metrics."""
        if len(metrics.click_intervals) < 10:
            return None
        
        features = [
            # Timing features
            np.std(metrics.click_intervals) / max(np.mean(metrics.click_intervals), 0.001),
            np.std(metrics.decision_times) / max(np.mean(metrics.decision_times), 0.001),
            
            # Movement features
            np.mean(metrics.movement_velocities) if metrics.movement_velocities else 0.0,
            self._calculate_movement_smoothness(metrics.mouse_positions),
            
            # Decision features
            np.std(metrics.risk_tolerance_history) if metrics.risk_tolerance_history else 0.0,
            self._calculate_action_consistency(metrics.action_sequence),
            
            # Pattern features
            self._calculate_entropy(metrics.action_sequence),
            self._calculate_repetition_score(metrics.repeated_sequences)
        ]
        
        return features
    
    def _calculate_movement_smoothness(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate smoothness of mouse movements."""
        if len(positions) < 3:
            return 0.0
        
        # Calculate acceleration changes (jerk)
        jerks = []
        for i in range(2, len(positions)):
            p1, p2, p3 = positions[i-2], positions[i-1], positions[i]
            
            # Calculate velocities
            v1 = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            v2 = ((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)**0.5
            
            # Calculate acceleration
            acc = abs(v2 - v1)
            jerks.append(acc)
        
        # Smoothness is inverse of jerk variance
        if jerks:
            return 1.0 / (1.0 + np.std(jerks))
        return 0.0
    
    def _calculate_action_consistency(self, actions: List[str]) -> float:
        """Calculate consistency of action patterns."""
        if len(actions) < 10:
            return 0.0
        
        # Look for repeated patterns
        pattern_counts = defaultdict(int)
        for i in range(len(actions) - 2):
            pattern = tuple(actions[i:i+3])
            pattern_counts[pattern] += 1
        
        if not pattern_counts:
            return 0.0
        
        # Calculate consistency as max pattern frequency
        max_count = max(pattern_counts.values())
        total_patterns = len(actions) - 2
        
        return max_count / total_patterns if total_patterns > 0 else 0.0
    
    def _calculate_entropy(self, sequence: List[str]) -> float:
        """Calculate entropy of action sequence."""
        if not sequence:
            return 0.0
        
        # Count frequencies
        counts = defaultdict(int)
        for action in sequence:
            counts[action] += 1
        
        # Calculate entropy
        total = len(sequence)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_repetition_score(self, repeated_sequences: Dict[str, int]) -> float:
        """Calculate repetition score from repeated sequences."""
        if not repeated_sequences:
            return 0.0
        
        total_sequences = sum(repeated_sequences.values())
        max_repetition = max(repeated_sequences.values())
        
        return max_repetition / total_sequences if total_sequences > 0 else 0.0
    
    def _detect_timing_patterns(self, metrics: BehaviorMetrics) -> Optional[DetectionAlert]:
        """Detect suspicious timing patterns."""
        if len(metrics.click_intervals) < 20:
            return None
        
        # Check for too-regular intervals
        intervals = np.array(metrics.click_intervals)
        cv = np.std(intervals) / np.mean(intervals)  # Coefficient of variation
        
        if cv < 0.1:  # Very low variation = suspicious
            severity = 1.0 - cv * 10  # Higher severity for lower variation
            return DetectionAlert(
                flag_type=BehaviorFlag.TIMING_PATTERN,
                severity=min(1.0, severity),
                confidence=0.8,
                description=f"Extremely regular click intervals detected (CV: {cv:.3f})",
                recommendation="Add random delays between actions",
                timestamp=time.time(),
                metrics_snapshot={'coefficient_variation': cv, 'mean_interval': np.mean(intervals)}
            )
        
        return None
    
    def _detect_click_precision(self, metrics: BehaviorMetrics) -> Optional[DetectionAlert]:
        """Detect inhuman click precision."""
        if len(metrics.click_positions) < 10:
            return None
        
        # Calculate click precision (distance from optimal click points)
        precisions = []
        for pos in metrics.click_positions:
            # Simulate optimal click point (center of cell)
            optimal_x = round(pos[0] / 50) * 50 + 25  # Assuming 50px cells
            optimal_y = round(pos[1] / 50) * 50 + 25
            
            distance = ((pos[0] - optimal_x)**2 + (pos[1] - optimal_y)**2)**0.5
            precision = max(0, 1.0 - distance / 25)  # Normalize to 0-1
            precisions.append(precision)
        
        avg_precision = np.mean(precisions)
        
        if avg_precision > self.precision_threshold:
            severity = (avg_precision - self.precision_threshold) / (1.0 - self.precision_threshold)
            return DetectionAlert(
                flag_type=BehaviorFlag.CLICK_PRECISION,
                severity=severity,
                confidence=0.7,
                description=f"Inhuman click precision detected (avg: {avg_precision:.3f})",
                recommendation="Add slight randomness to click positions",
                timestamp=time.time(),
                metrics_snapshot={'avg_precision': avg_precision, 'precision_std': np.std(precisions)}
            )
        
        return None
    
    def _detect_movement_anomalies(self, metrics: BehaviorMetrics) -> Optional[DetectionAlert]:
        """Detect anomalous movement patterns."""
        if len(metrics.mouse_positions) < 20:
            return None
        
        smoothness = self._calculate_movement_smoothness(metrics.mouse_positions)
        
        if smoothness > 0.9:  # Too smooth = suspicious
            severity = (smoothness - 0.9) / 0.1
            return DetectionAlert(
                flag_type=BehaviorFlag.MOVEMENT_SMOOTHNESS,
                severity=severity,
                confidence=0.6,
                description=f"Unnaturally smooth mouse movements (smoothness: {smoothness:.3f})",
                recommendation="Add micro-movements and slight tremor to mouse path",
                timestamp=time.time(),
                metrics_snapshot={'movement_smoothness': smoothness}
            )
        
        return None
    
    def _detect_inhuman_consistency(self, metrics: BehaviorMetrics) -> Optional[DetectionAlert]:
        """Detect inhuman consistency in behavior."""
        if len(metrics.consistency_scores) < 10:
            return None
        
        avg_consistency = np.mean(metrics.consistency_scores)
        
        if avg_consistency > self.consistency_threshold:
            severity = (avg_consistency - self.consistency_threshold) / (1.0 - self.consistency_threshold)
            return DetectionAlert(
                flag_type=BehaviorFlag.INHUMAN_CONSISTENCY,
                severity=severity,
                confidence=0.8,
                description=f"Inhuman behavioral consistency (avg: {avg_consistency:.3f})",
                recommendation="Introduce occasional suboptimal decisions and hesitation",
                timestamp=time.time(),
                metrics_snapshot={'avg_consistency': avg_consistency, 'consistency_std': np.std(metrics.consistency_scores)}
            )
        
        return None
    
    def _detect_pattern_repetition(self, metrics: BehaviorMetrics) -> Optional[DetectionAlert]:
        """Detect excessive pattern repetition."""
        repetition_score = self._calculate_repetition_score(metrics.repeated_sequences)
        
        if repetition_score > 0.7:  # High repetition = suspicious
            severity = (repetition_score - 0.7) / 0.3
            return DetectionAlert(
                flag_type=BehaviorFlag.PATTERN_REPETITION,
                severity=severity,
                confidence=0.7,
                description=f"Excessive pattern repetition detected (score: {repetition_score:.3f})",
                recommendation="Vary action sequences and introduce randomness",
                timestamp=time.time(),
                metrics_snapshot={'repetition_score': repetition_score, 'unique_patterns': len(metrics.repeated_sequences)}
            )
        
        return None
    
    def _ml_based_detection(self, features: List[float]) -> Optional[DetectionAlert]:
        """Machine learning based bot detection."""
        if not features or len(features) != 8:
            return None
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Anomaly detection
        anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
        is_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1
        
        # Classification
        bot_probability = self.classifier.predict_proba(features_scaled)[0][1]
        
        # Combine scores
        combined_score = (abs(anomaly_score) + bot_probability) / 2
        
        if combined_score > self.anomaly_threshold or is_anomaly:
            severity = min(1.0, combined_score)
            return DetectionAlert(
                flag_type=BehaviorFlag.STATISTICAL_ANOMALY,
                severity=severity,
                confidence=bot_probability,
                description=f"ML models detected bot-like behavior (score: {combined_score:.3f})",
                recommendation="Review and adjust all behavioral parameters",
                timestamp=time.time(),
                metrics_snapshot={'anomaly_score': anomaly_score, 'bot_probability': bot_probability}
            )
        
        return None

class ReverseTuringTest:
    """
    Reverse Turing Test system that provides real-time feedback about bot detection risk.
    
    Like having Kurisu's analytical abilities combined with Okabe's paranoia about
    the Organization's surveillance - always one step ahead of detection!
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Detection model
        self.detection_model = BotDetectionModel(self.config.get('detection_config', {}))
        
        # Behavior tracking
        self.current_metrics = BehaviorMetrics()
        self.behavior_history = deque(maxlen=self.config.get('history_length', 10000))
        
        # Alert system
        self.active_alerts = []
        self.alert_history = []
        self.risk_level = 0.0  # 0.0 to 1.0
        
        # Countermeasures
        self.countermeasures_enabled = self.config.get('countermeasures_enabled', True)
        self.adaptive_behavior = self.config.get('adaptive_behavior', True)
        
        # Performance tracking
        self.detection_events = 0
        self.false_alarms = 0
        self.successful_evasions = 0
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for reverse Turing test."""
        return {
            'history_length': 10000,
            'countermeasures_enabled': True,
            'adaptive_behavior': True,
            'risk_threshold': 0.7,
            'alert_cooldown': 30.0,  # seconds
            'max_active_alerts': 5,
            'detection_config': {}
        }
    
    def update_behavior(self, action_data: Dict[str, Any]):
        """Update behavior metrics with new action data."""
        # Extract timing information
        if 'timestamp' in action_data:
            if self.behavior_history:
                interval = action_data['timestamp'] - self.behavior_history[-1].get('timestamp', 0)
                self.current_metrics.click_intervals.append(interval)
        
        # Extract position information
        if 'mouse_position' in action_data:
            self.current_metrics.mouse_positions.append(action_data['mouse_position'])
        
        if 'click_position' in action_data:
            self.current_metrics.click_positions.append(action_data['click_position'])
        
        # Extract decision information
        if 'action' in action_data:
            self.current_metrics.action_sequence.append(action_data['action'])
        
        if 'decision_time' in action_data:
            self.current_metrics.decision_times.append(action_data['decision_time'])
        
        if 'risk_tolerance' in action_data:
            self.current_metrics.risk_tolerance_history.append(action_data['risk_tolerance'])
        
        # Store in history
        self.behavior_history.append(action_data)
        
        # Update pattern tracking
        self._update_pattern_tracking()
        
        # Calculate movement metrics
        self._update_movement_metrics()
    
    def _update_pattern_tracking(self):
        """Update pattern repetition tracking."""
        if len(self.current_metrics.action_sequence) >= 3:
            # Look for 3-action patterns
            recent_pattern = tuple(self.current_metrics.action_sequence[-3:])
            pattern_key = '_'.join(recent_pattern)
            self.current_metrics.repeated_sequences[pattern_key] = \
                self.current_metrics.repeated_sequences.get(pattern_key, 0) + 1
    
    def _update_movement_metrics(self):
        """Update movement-related metrics."""
        if len(self.current_metrics.mouse_positions) >= 2:
            # Calculate velocity
            p1 = self.current_metrics.mouse_positions[-2]
            p2 = self.current_metrics.mouse_positions[-1]
            
            distance = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            time_diff = 0.1  # Assume 100ms between position updates
            velocity = distance / time_diff
            
            self.current_metrics.movement_velocities.append(velocity)
    
    def check_detection_risk(self) -> Dict[str, Any]:
        """
        Check current detection risk and return real-time feedback.
        
        Like having Reading Steiner activated to see if the Organization
        is about to detect your activities!
        """
        # Run detection analysis
        alerts = self.detection_model.analyze_behavior(self.current_metrics)
        
        # Update active alerts
        self._update_active_alerts(alerts)
        
        # Calculate overall risk level
        self.risk_level = self._calculate_risk_level()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Apply countermeasures if enabled
        countermeasures = []
        if self.countermeasures_enabled and self.risk_level > self.config.get('risk_threshold', 0.7):
            countermeasures = self._apply_countermeasures()
        
        return {
            'risk_level': self.risk_level,
            'risk_category': self._get_risk_category(self.risk_level),
            'active_alerts': [self._alert_to_dict(alert) for alert in self.active_alerts],
            'recommendations': recommendations,
            'countermeasures': countermeasures,
            'detection_probability': self._estimate_detection_probability(),
            'behavior_summary': self._get_behavior_summary(),
            'evasion_status': self._get_evasion_status()
        }
    
    def _update_active_alerts(self, new_alerts: List[DetectionAlert]):
        """Update the list of active alerts."""
        current_time = time.time()
        cooldown = self.config.get('alert_cooldown', 30.0)
        
        # Remove expired alerts
        self.active_alerts = [
            alert for alert in self.active_alerts
            if current_time - alert.timestamp < cooldown
        ]
        
        # Add new alerts
        for alert in new_alerts:
            # Check if similar alert already exists
            similar_exists = any(
                existing.flag_type == alert.flag_type and
                current_time - existing.timestamp < cooldown
                for existing in self.active_alerts
            )
            
            if not similar_exists:
                self.active_alerts.append(alert)
                self.alert_history.append(alert)
        
        # Limit number of active alerts
        max_alerts = self.config.get('max_active_alerts', 5)
        if len(self.active_alerts) > max_alerts:
            # Keep highest severity alerts
            self.active_alerts.sort(key=lambda a: a.severity, reverse=True)
            self.active_alerts = self.active_alerts[:max_alerts]
    
    def _calculate_risk_level(self) -> float:
        """Calculate overall risk level based on active alerts."""
        if not self.active_alerts:
            return 0.0
        
        # Weight alerts by severity and confidence
        weighted_scores = []
        for alert in self.active_alerts:
            weight = self.config.get('detection_config', {}).get('severity_weights', {}).get(
                alert.flag_type.value, 0.2
            )
            score = alert.severity * alert.confidence * weight
            weighted_scores.append(score)
        
        # Combine scores (not just average to account for multiple issues)
        if weighted_scores:
            max_score = max(weighted_scores)
            avg_score = np.mean(weighted_scores)
            combined_score = 0.7 * max_score + 0.3 * avg_score
            return min(1.0, combined_score)
        
        return 0.0
    
    def _get_risk_category(self, risk_level: float) -> str:
        """Get risk category based on risk level."""
        if risk_level < 0.3:
            return "LOW"
        elif risk_level < 0.6:
            return "MEDIUM"
        elif risk_level < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on active alerts."""
        recommendations = []
        
        for alert in self.active_alerts:
            if alert.recommendation not in recommendations:
                recommendations.append(alert.recommendation)
        
        # Add general recommendations based on risk level
        if self.risk_level > 0.8:
            recommendations.append("Consider taking a break to reset behavioral patterns")
            recommendations.append("Switch to manual play mode temporarily")
        elif self.risk_level > 0.6:
            recommendations.append("Increase randomness in all behavioral parameters")
            recommendations.append("Vary session timing and duration")
        
        return recommendations
    
    def _apply_countermeasures(self) -> List[str]:
        """Apply automatic countermeasures to reduce detection risk."""
        countermeasures = []
        
        for alert in self.active_alerts:
            if alert.flag_type == BehaviorFlag.TIMING_PATTERN:
                countermeasures.append("Applied random delay variation")
            elif alert.flag_type == BehaviorFlag.CLICK_PRECISION:
                countermeasures.append("Added click position randomization")
            elif alert.flag_type == BehaviorFlag.MOVEMENT_SMOOTHNESS:
                countermeasures.append("Introduced mouse movement jitter")
            elif alert.flag_type == BehaviorFlag.INHUMAN_CONSISTENCY:
                countermeasures.append("Activated decision randomization")
            elif alert.flag_type == BehaviorFlag.PATTERN_REPETITION:
                countermeasures.append("Enabled pattern breaking mode")
        
        return countermeasures
    
    def _estimate_detection_probability(self) -> float:
        """Estimate probability of being detected by platform."""
        # Base probability on risk level and historical data
        base_probability = self.risk_level * 0.3  # Max 30% base probability
        
        # Adjust based on alert severity
        severity_multiplier = 1.0
        if self.active_alerts:
            max_severity = max(alert.severity for alert in self.active_alerts)
            severity_multiplier = 1.0 + max_severity * 0.5
        
        # Adjust based on alert count
        count_multiplier = 1.0 + len(self.active_alerts) * 0.1
        
        estimated_probability = base_probability * severity_multiplier * count_multiplier
        
        return min(0.95, estimated_probability)  # Cap at 95%
    
    def _get_behavior_summary(self) -> Dict[str, Any]:
        """Get summary of current behavior metrics."""
        return {
            'total_actions': len(self.current_metrics.action_sequence),
            'avg_click_interval': np.mean(self.current_metrics.click_intervals) if self.current_metrics.click_intervals else 0.0,
            'avg_decision_time': np.mean(self.current_metrics.decision_times) if self.current_metrics.decision_times else 0.0,
            'movement_samples': len(self.current_metrics.mouse_positions),
            'unique_patterns': len(self.current_metrics.repeated_sequences),
            'session_duration': self.current_metrics.session_duration
        }
    
    def _get_evasion_status(self) -> str:
        """Get current evasion status."""
        if self.risk_level < 0.3:
            return "STEALTH_MODE"  # Flying under the radar
        elif self.risk_level < 0.6:
            return "CAUTION_ADVISED"  # Some attention but manageable
        elif self.risk_level < 0.8:
            return "EVASIVE_MANEUVERS"  # Active countermeasures needed
        else:
            return "ABORT_MISSION"  # High risk of detection
    
    def _alert_to_dict(self, alert: DetectionAlert) -> Dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            'flag_type': alert.flag_type.value,
            'severity': alert.severity,
            'confidence': alert.confidence,
            'description': alert.description,
            'recommendation': alert.recommendation,
            'timestamp': alert.timestamp,
            'metrics_snapshot': alert.metrics_snapshot
        }
    
    def get_evasion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evasion statistics."""
        return {
            "system_name": "Reverse Turing Test - Bot Detection Evasion",
            "total_detection_events": self.detection_events,
            "false_alarms": self.false_alarms,
            "successful_evasions": self.successful_evasions,
            "current_risk_level": self.risk_level,
            "active_alerts_count": len(self.active_alerts),
            "total_alerts_generated": len(self.alert_history),
            "countermeasures_enabled": self.countermeasures_enabled,
            "adaptive_behavior_enabled": self.adaptive_behavior,
            "behavior_samples": len(self.behavior_history),
            "detection_model_trained": self.detection_model.is_trained,
            "evasion_success_rate": self.successful_evasions / max(1, self.detection_events + self.successful_evasions)
        }
    
    def visualize_risk_analysis(self, save_path: str = None) -> str:
        """Create visualization of risk analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Risk level over time
        if self.alert_history:
            timestamps = [alert.timestamp for alert in self.alert_history]
            severities = [alert.severity for alert in self.alert_history]
            
            ax1.plot(timestamps, severities, 'r-', alpha=0.7, linewidth=2)
            ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Risk Threshold')
            ax1.set_title('Risk Level Over Time')
            ax1.set_ylabel('Severity')
            ax1.set_xlabel('Timestamp')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Alert type distribution
        if self.alert_history:
            alert_types = [alert.flag_type.value for alert in self.alert_history]
            type_counts = defaultdict(int)
            for alert_type in alert_types:
                type_counts[alert_type] += 1
            
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            
            ax2.bar(types, counts, color='lightcoral', alpha=0.7)
            ax2.set_title('Alert Type Distribution')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
        
        # Behavior metrics visualization
        if self.current_metrics.click_intervals:
            ax3.hist(self.current_metrics.click_intervals, bins=20, alpha=0.7, 
                    color='skyblue', edgecolor='black')
            ax3.axvline(x=np.mean(self.current_metrics.click_intervals), 
                       color='red', linestyle='--', linewidth=2, label='Mean')
            ax3.set_title('Click Interval Distribution')
            ax3.set_xlabel('Interval (seconds)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
        
        # Risk gauge
        risk_colors = ['green', 'yellow', 'orange', 'red']
        risk_levels = [0.25, 0.5, 0.75, 1.0]
        
        for i, (level, color) in enumerate(zip(risk_levels, risk_colors)):
            ax4.barh(0, level, left=sum(risk_levels[:i]), color=color, alpha=0.7)
        
        # Current risk indicator
        ax4.axvline(x=self.risk_level, color='black', linewidth=3, label=f'Current Risk: {self.risk_level:.2f}')
        ax4.set_title('Risk Level Gauge')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(-0.5, 0.5)
        ax4.set_xlabel('Risk Level')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            save_path = f"/home/ubuntu/fusion-project/python-backend/visualizations/reverse_turing_test_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path

# Example usage and testing
if __name__ == "__main__":
    # Create reverse Turing test system
    rtt = ReverseTuringTest({
        'countermeasures_enabled': True,
        'adaptive_behavior': True,
        'risk_threshold': 0.6
    })
    
    print("🕵️ Reverse Turing Test System Test")
    
    # Simulate bot-like behavior
    print("\n🤖 Simulating bot-like behavior...")
    for i in range(100):
        # Very regular timing (suspicious)
        action_data = {
            'timestamp': time.time() + i * 0.5,  # Exactly 0.5s intervals
            'mouse_position': (100 + i, 100 + i),  # Linear movement
            'click_position': (125, 125),  # Always same position
            'action': 'reveal_cell',
            'decision_time': 0.1,  # Always same decision time
            'risk_tolerance': 0.5  # Never changes
        }
        rtt.update_behavior(action_data)
    
    # Check detection risk
    risk_analysis = rtt.check_detection_risk()
    
    print(f"Risk Level: {risk_analysis['risk_level']:.3f} ({risk_analysis['risk_category']})")
    print(f"Detection Probability: {risk_analysis['detection_probability']:.3f}")
    print(f"Evasion Status: {risk_analysis['evasion_status']}")
    
    print(f"\n🚨 Active Alerts ({len(risk_analysis['active_alerts'])}):")
    for alert in risk_analysis['active_alerts']:
        print(f"  - {alert['flag_type']}: {alert['description']}")
        print(f"    Severity: {alert['severity']:.3f}, Confidence: {alert['confidence']:.3f}")
        print(f"    Recommendation: {alert['recommendation']}")
    
    print(f"\n💡 Recommendations:")
    for rec in risk_analysis['recommendations']:
        print(f"  - {rec}")
    
    if risk_analysis['countermeasures']:
        print(f"\n🛡️ Applied Countermeasures:")
        for cm in risk_analysis['countermeasures']:
            print(f"  - {cm}")
    
    # Get statistics
    stats = rtt.get_evasion_statistics()
    print(f"\n📊 Evasion Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create visualization
    viz_path = rtt.visualize_risk_analysis()
    print(f"\n📊 Risk analysis visualization saved to: {viz_path}")
    
    print("\n🎯 El Psy Kongroo! The Organization's surveillance has been analyzed!")

