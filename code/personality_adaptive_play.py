"""
Personality-Adaptive Play System

This module implements personality-adaptive play that adjusts strategy based on biometric
or interaction-based feedback. Like having an AI that can read your emotional state and
adapt like Senku analyzing human psychology or Lelouch reading people's intentions!
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
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class EmotionalState(Enum):
    """Player's emotional states that can be detected."""
    CALM = "calm"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    CONFIDENT = "confident"
    ANXIOUS = "anxious"
    BORED = "bored"
    FOCUSED = "focused"
    STRESSED = "stressed"

class PersonalityTrait(Enum):
    """Personality traits that influence strategy adaptation."""
    RISK_SEEKING = "risk_seeking"
    RISK_AVERSE = "risk_averse"
    ANALYTICAL = "analytical"
    IMPULSIVE = "impulsive"
    PATIENT = "patient"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"

class BiometricIndicator(Enum):
    """Types of biometric/behavioral indicators."""
    HEART_RATE = "heart_rate"
    CLICK_PRESSURE = "click_pressure"
    MOUSE_TREMOR = "mouse_tremor"
    TYPING_RHYTHM = "typing_rhythm"
    DECISION_LATENCY = "decision_latency"
    INTERACTION_FREQUENCY = "interaction_frequency"
    ATTENTION_SPAN = "attention_span"
    STRESS_MARKERS = "stress_markers"

@dataclass
class BiometricData:
    """Container for biometric and behavioral data."""
    heart_rate: Optional[float] = None
    click_pressure: List[float] = field(default_factory=list)
    mouse_positions: List[Tuple[float, float]] = field(default_factory=list)
    decision_times: List[float] = field(default_factory=list)
    interaction_intervals: List[float] = field(default_factory=list)
    attention_metrics: List[float] = field(default_factory=list)
    stress_indicators: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class PersonalityProfile:
    """Player's personality profile and preferences."""
    traits: Dict[PersonalityTrait, float] = field(default_factory=dict)
    emotional_baseline: Dict[EmotionalState, float] = field(default_factory=dict)
    stress_tolerance: float = 0.5
    risk_preference: float = 0.5
    learning_speed: float = 0.5
    adaptation_sensitivity: float = 0.5
    preferred_strategies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Initialize default trait values
        for trait in PersonalityTrait:
            if trait not in self.traits:
                self.traits[trait] = 0.5
        
        # Initialize emotional baseline
        for emotion in EmotionalState:
            if emotion not in self.emotional_baseline:
                self.emotional_baseline[emotion] = 0.3

@dataclass
class AdaptationRule:
    """Rule for adapting strategy based on emotional state."""
    trigger_emotion: EmotionalState
    trigger_threshold: float
    target_strategy: str
    adaptation_strength: float
    duration: float  # How long to maintain adaptation
    description: str

class EmotionalStateDetector:
    """
    Detects player's emotional state from biometric and behavioral data.
    
    Like having Senku's scientific analysis combined with Lelouch's ability
    to read people's psychological states!
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Detection parameters
        self.baseline_window = self.config.get('baseline_window', 100)
        self.detection_sensitivity = self.config.get('detection_sensitivity', 0.7)
        self.smoothing_factor = self.config.get('smoothing_factor', 0.3)
        
        # Baseline measurements
        self.baseline_metrics = {}
        self.recent_measurements = deque(maxlen=self.baseline_window)
        
        # State tracking
        self.current_state = EmotionalState.CALM
        self.state_confidence = 0.5
        self.state_history = deque(maxlen=1000)
        
        # Calibration data
        self.is_calibrated = False
        self.calibration_data = defaultdict(list)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for emotional state detection."""
        return {
            'baseline_window': 100,
            'detection_sensitivity': 0.7,
            'smoothing_factor': 0.3,
            'stress_threshold': 0.8,
            'excitement_threshold': 0.7,
            'frustration_threshold': 0.6,
            'confidence_threshold': 0.8,
            'min_samples_for_detection': 10
        }
    
    def update_biometric_data(self, data: BiometricData):
        """Update with new biometric data."""
        self.recent_measurements.append(data)
        
        # Update baseline if we have enough data
        if len(self.recent_measurements) >= self.baseline_window and not self.is_calibrated:
            self._calculate_baseline()
            self.is_calibrated = True
        
        # Detect emotional state
        if self.is_calibrated:
            new_state, confidence = self._detect_emotional_state(data)
            self._update_emotional_state(new_state, confidence)
    
    def _calculate_baseline(self):
        """Calculate baseline metrics from recent measurements."""
        if not self.recent_measurements:
            return
        
        # Heart rate baseline
        heart_rates = [m.heart_rate for m in self.recent_measurements if m.heart_rate is not None]
        if heart_rates:
            self.baseline_metrics['heart_rate'] = {
                'mean': np.mean(heart_rates),
                'std': np.std(heart_rates)
            }
        
        # Click pressure baseline
        all_pressures = []
        for m in self.recent_measurements:
            all_pressures.extend(m.click_pressure)
        if all_pressures:
            self.baseline_metrics['click_pressure'] = {
                'mean': np.mean(all_pressures),
                'std': np.std(all_pressures)
            }
        
        # Decision time baseline
        all_decision_times = []
        for m in self.recent_measurements:
            all_decision_times.extend(m.decision_times)
        if all_decision_times:
            self.baseline_metrics['decision_time'] = {
                'mean': np.mean(all_decision_times),
                'std': np.std(all_decision_times)
            }
        
        # Mouse tremor baseline
        tremor_scores = []
        for m in self.recent_measurements:
            if len(m.mouse_positions) >= 3:
                tremor = self._calculate_mouse_tremor(m.mouse_positions)
                tremor_scores.append(tremor)
        if tremor_scores:
            self.baseline_metrics['mouse_tremor'] = {
                'mean': np.mean(tremor_scores),
                'std': np.std(tremor_scores)
            }
    
    def _detect_emotional_state(self, data: BiometricData) -> Tuple[EmotionalState, float]:
        """Detect emotional state from current biometric data."""
        state_scores = defaultdict(float)
        total_confidence = 0.0
        
        # Heart rate analysis
        if data.heart_rate and 'heart_rate' in self.baseline_metrics:
            hr_z_score = (data.heart_rate - self.baseline_metrics['heart_rate']['mean']) / \
                         max(self.baseline_metrics['heart_rate']['std'], 1.0)
            
            if hr_z_score > 2.0:  # Significantly elevated
                state_scores[EmotionalState.EXCITED] += 0.8
                state_scores[EmotionalState.STRESSED] += 0.6
            elif hr_z_score > 1.0:
                state_scores[EmotionalState.ANXIOUS] += 0.6
            elif hr_z_score < -1.0:
                state_scores[EmotionalState.BORED] += 0.5
                state_scores[EmotionalState.CALM] += 0.7
            
            total_confidence += 0.3
        
        # Click pressure analysis
        if data.click_pressure and 'click_pressure' in self.baseline_metrics:
            avg_pressure = np.mean(data.click_pressure)
            pressure_z_score = (avg_pressure - self.baseline_metrics['click_pressure']['mean']) / \
                              max(self.baseline_metrics['click_pressure']['std'], 0.1)
            
            if pressure_z_score > 1.5:  # High pressure
                state_scores[EmotionalState.FRUSTRATED] += 0.7
                state_scores[EmotionalState.STRESSED] += 0.5
            elif pressure_z_score > 0.5:
                state_scores[EmotionalState.FOCUSED] += 0.6
            
            total_confidence += 0.2
        
        # Decision time analysis
        if data.decision_times and 'decision_time' in self.baseline_metrics:
            avg_decision_time = np.mean(data.decision_times)
            dt_z_score = (avg_decision_time - self.baseline_metrics['decision_time']['mean']) / \
                        max(self.baseline_metrics['decision_time']['std'], 0.1)
            
            if dt_z_score > 1.5:  # Slow decisions
                state_scores[EmotionalState.ANXIOUS] += 0.6
                state_scores[EmotionalState.STRESSED] += 0.4
            elif dt_z_score < -1.0:  # Fast decisions
                state_scores[EmotionalState.CONFIDENT] += 0.7
                state_scores[EmotionalState.EXCITED] += 0.5
            
            total_confidence += 0.25
        
        # Mouse tremor analysis
        if len(data.mouse_positions) >= 3 and 'mouse_tremor' in self.baseline_metrics:
            tremor = self._calculate_mouse_tremor(data.mouse_positions)
            tremor_z_score = (tremor - self.baseline_metrics['mouse_tremor']['mean']) / \
                           max(self.baseline_metrics['mouse_tremor']['std'], 0.01)
            
            if tremor_z_score > 1.5:  # High tremor
                state_scores[EmotionalState.ANXIOUS] += 0.8
                state_scores[EmotionalState.STRESSED] += 0.6
            elif tremor_z_score < -0.5:  # Very steady
                state_scores[EmotionalState.FOCUSED] += 0.7
                state_scores[EmotionalState.CALM] += 0.5
            
            total_confidence += 0.25
        
        # Determine most likely state
        if state_scores:
            best_state = max(state_scores.keys(), key=lambda s: state_scores[s])
            confidence = min(1.0, state_scores[best_state] * total_confidence)
            return best_state, confidence
        
        return EmotionalState.CALM, 0.5
    
    def _calculate_mouse_tremor(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate mouse tremor/jitter from position data."""
        if len(positions) < 3:
            return 0.0
        
        # Calculate second derivatives (acceleration changes)
        accelerations = []
        for i in range(2, len(positions)):
            p1, p2, p3 = positions[i-2], positions[i-1], positions[i]
            
            # Velocity vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Acceleration
            acc = ((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2)**0.5
            accelerations.append(acc)
        
        # Tremor is the variance in accelerations
        return np.std(accelerations) if accelerations else 0.0
    
    def _update_emotional_state(self, new_state: EmotionalState, confidence: float):
        """Update current emotional state with smoothing."""
        # Apply smoothing to prevent rapid state changes
        if confidence > self.detection_sensitivity:
            # Weighted update based on confidence
            weight = confidence * (1 - self.smoothing_factor)
            
            if new_state != self.current_state:
                # State change - require higher confidence
                if confidence > 0.8:
                    self.current_state = new_state
                    self.state_confidence = confidence
            else:
                # Same state - reinforce confidence
                self.state_confidence = min(1.0, self.state_confidence * 0.9 + confidence * 0.1)
        
        # Record state history
        self.state_history.append({
            'state': self.current_state,
            'confidence': self.state_confidence,
            'timestamp': time.time()
        })
    
    def get_current_state(self) -> Tuple[EmotionalState, float]:
        """Get current emotional state and confidence."""
        return self.current_state, self.state_confidence
    
    def get_state_history(self, duration: float = 300.0) -> List[Dict[str, Any]]:
        """Get emotional state history for the last duration seconds."""
        current_time = time.time()
        return [
            entry for entry in self.state_history
            if current_time - entry['timestamp'] <= duration
        ]

class PersonalityAdaptiveStrategy:
    """
    Adaptive strategy system that adjusts based on personality and emotional state.
    
    Like having an AI companion that understands your personality and adapts
    its behavior like Jarvis or Cortana, but for gaming strategies!
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Core components
        self.emotion_detector = EmotionalStateDetector(self.config.get('detector_config', {}))
        self.personality_profile = PersonalityProfile()
        
        # Adaptation system
        self.adaptation_rules = self._create_adaptation_rules()
        self.active_adaptations = []
        self.adaptation_history = []
        
        # Strategy mapping
        self.strategy_mapping = {
            EmotionalState.CALM: "senku",  # Analytical when calm
            EmotionalState.EXCITED: "takeshi",  # Aggressive when excited
            EmotionalState.FRUSTRATED: "kazuya",  # Conservative when frustrated
            EmotionalState.CONFIDENT: "lelouch",  # Strategic when confident
            EmotionalState.ANXIOUS: "kazuya",  # Safe when anxious
            EmotionalState.BORED: "okabe",  # Experimental when bored
            EmotionalState.FOCUSED: "senku",  # Analytical when focused
            EmotionalState.STRESSED: "hybrid"  # Balanced when stressed
        }
        
        # Performance tracking
        self.adaptation_effectiveness = defaultdict(list)
        self.personality_learning = True
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for adaptive strategy."""
        return {
            'adaptation_threshold': 0.7,
            'adaptation_duration': 300.0,  # 5 minutes
            'learning_rate': 0.1,
            'personality_update_frequency': 100,  # actions
            'min_confidence_for_adaptation': 0.6,
            'detector_config': {}
        }
    
    def _create_adaptation_rules(self) -> List[AdaptationRule]:
        """Create adaptation rules for different emotional states."""
        return [
            AdaptationRule(
                trigger_emotion=EmotionalState.FRUSTRATED,
                trigger_threshold=0.7,
                target_strategy="kazuya",
                adaptation_strength=0.8,
                duration=180.0,
                description="Switch to conservative strategy when frustrated"
            ),
            AdaptationRule(
                trigger_emotion=EmotionalState.EXCITED,
                trigger_threshold=0.6,
                target_strategy="takeshi",
                adaptation_strength=0.7,
                duration=120.0,
                description="Switch to aggressive strategy when excited"
            ),
            AdaptationRule(
                trigger_emotion=EmotionalState.ANXIOUS,
                trigger_threshold=0.8,
                target_strategy="kazuya",
                adaptation_strength=0.9,
                duration=240.0,
                description="Switch to safe strategy when anxious"
            ),
            AdaptationRule(
                trigger_emotion=EmotionalState.CONFIDENT,
                trigger_threshold=0.7,
                target_strategy="lelouch",
                adaptation_strength=0.6,
                duration=150.0,
                description="Switch to strategic approach when confident"
            ),
            AdaptationRule(
                trigger_emotion=EmotionalState.BORED,
                trigger_threshold=0.6,
                target_strategy="okabe",
                adaptation_strength=0.5,
                duration=90.0,
                description="Switch to experimental strategy when bored"
            ),
            AdaptationRule(
                trigger_emotion=EmotionalState.STRESSED,
                trigger_threshold=0.8,
                target_strategy="hybrid",
                adaptation_strength=0.7,
                duration=300.0,
                description="Switch to balanced strategy when stressed"
            )
        ]
    
    def update_biometric_data(self, biometric_data: BiometricData):
        """Update with new biometric data and trigger adaptations."""
        # Update emotion detector
        self.emotion_detector.update_biometric_data(biometric_data)
        
        # Check for adaptation triggers
        current_state, confidence = self.emotion_detector.get_current_state()
        
        if confidence >= self.config.get('min_confidence_for_adaptation', 0.6):
            self._check_adaptation_triggers(current_state, confidence)
        
        # Clean up expired adaptations
        self._cleanup_expired_adaptations()
    
    def _check_adaptation_triggers(self, emotional_state: EmotionalState, confidence: float):
        """Check if any adaptation rules should be triggered."""
        current_time = time.time()
        
        for rule in self.adaptation_rules:
            if (rule.trigger_emotion == emotional_state and 
                confidence >= rule.trigger_threshold):
                
                # Check if this adaptation is already active
                already_active = any(
                    adaptation['rule'].trigger_emotion == rule.trigger_emotion
                    for adaptation in self.active_adaptations
                )
                
                if not already_active:
                    # Trigger new adaptation
                    adaptation = {
                        'rule': rule,
                        'start_time': current_time,
                        'confidence': confidence,
                        'effectiveness_score': 0.0
                    }
                    
                    self.active_adaptations.append(adaptation)
                    self.adaptation_history.append(adaptation.copy())
                    
                    print(f"🎭 Triggered adaptation: {rule.description}")
    
    def _cleanup_expired_adaptations(self):
        """Remove expired adaptations."""
        current_time = time.time()
        
        self.active_adaptations = [
            adaptation for adaptation in self.active_adaptations
            if current_time - adaptation['start_time'] <= adaptation['rule'].duration
        ]
    
    def get_recommended_strategy(self) -> Dict[str, Any]:
        """Get recommended strategy based on current state and adaptations."""
        current_state, confidence = self.emotion_detector.get_current_state()
        
        # Base strategy from emotional state
        base_strategy = self.strategy_mapping.get(current_state, "senku")
        
        # Check for active adaptations
        active_strategy = base_strategy
        adaptation_strength = 0.0
        adaptation_reason = "Base emotional state mapping"
        
        if self.active_adaptations:
            # Use the most recent high-confidence adaptation
            best_adaptation = max(
                self.active_adaptations,
                key=lambda a: a['confidence'] * a['rule'].adaptation_strength
            )
            
            active_strategy = best_adaptation['rule'].target_strategy
            adaptation_strength = best_adaptation['rule'].adaptation_strength
            adaptation_reason = best_adaptation['rule'].description
        
        # Apply personality modifiers
        personality_modifiers = self._get_personality_modifiers(active_strategy)
        
        return {
            'strategy': active_strategy,
            'emotional_state': current_state.value,
            'state_confidence': confidence,
            'adaptation_strength': adaptation_strength,
            'adaptation_reason': adaptation_reason,
            'personality_modifiers': personality_modifiers,
            'base_strategy': base_strategy,
            'active_adaptations_count': len(self.active_adaptations)
        }
    
    def _get_personality_modifiers(self, strategy: str) -> Dict[str, float]:
        """Get personality-based modifiers for the strategy."""
        modifiers = {}
        
        # Risk tolerance modifier
        risk_trait = self.personality_profile.traits.get(PersonalityTrait.RISK_SEEKING, 0.5)
        if strategy in ["takeshi", "okabe"]:  # Aggressive strategies
            modifiers['risk_multiplier'] = 0.5 + risk_trait
        elif strategy == "kazuya":  # Conservative strategy
            modifiers['risk_multiplier'] = 1.5 - risk_trait
        else:
            modifiers['risk_multiplier'] = 1.0
        
        # Patience modifier
        patience_trait = self.personality_profile.traits.get(PersonalityTrait.PATIENT, 0.5)
        modifiers['patience_multiplier'] = 0.5 + patience_trait
        
        # Analytical modifier
        analytical_trait = self.personality_profile.traits.get(PersonalityTrait.ANALYTICAL, 0.5)
        if strategy in ["senku", "lelouch"]:  # Analytical strategies
            modifiers['analysis_depth'] = 0.5 + analytical_trait
        else:
            modifiers['analysis_depth'] = analytical_trait
        
        return modifiers
    
    def update_personality_profile(self, performance_data: Dict[str, Any]):
        """Update personality profile based on performance and preferences."""
        if not self.personality_learning:
            return
        
        # Update risk preference based on performance
        if 'risk_taken' in performance_data and 'outcome' in performance_data:
            risk_taken = performance_data['risk_taken']
            outcome = performance_data['outcome']  # 1 for success, 0 for failure
            
            # Adjust risk seeking trait
            current_risk_seeking = self.personality_profile.traits[PersonalityTrait.RISK_SEEKING]
            
            if outcome > 0.5:  # Successful outcome
                if risk_taken > 0.7:  # High risk, good outcome
                    adjustment = 0.02
                else:  # Low risk, good outcome
                    adjustment = -0.01
            else:  # Poor outcome
                if risk_taken > 0.7:  # High risk, bad outcome
                    adjustment = -0.03
                else:  # Low risk, bad outcome
                    adjustment = 0.01
            
            new_value = np.clip(current_risk_seeking + adjustment, 0.0, 1.0)
            self.personality_profile.traits[PersonalityTrait.RISK_SEEKING] = new_value
        
        # Update patience based on decision times
        if 'decision_time' in performance_data:
            decision_time = performance_data['decision_time']
            current_patience = self.personality_profile.traits[PersonalityTrait.PATIENT]
            
            # Longer decision times suggest more patience
            if decision_time > 3.0:  # Slow decision
                adjustment = 0.01
            elif decision_time < 0.5:  # Fast decision
                adjustment = -0.01
            else:
                adjustment = 0.0
            
            new_value = np.clip(current_patience + adjustment, 0.0, 1.0)
            self.personality_profile.traits[PersonalityTrait.PATIENT] = new_value
    
    def get_adaptation_effectiveness(self) -> Dict[str, Any]:
        """Get effectiveness metrics for adaptations."""
        if not self.adaptation_history:
            return {"status": "No adaptation data available"}
        
        # Calculate effectiveness by emotional state
        effectiveness_by_state = defaultdict(list)
        for adaptation in self.adaptation_history:
            state = adaptation['rule'].trigger_emotion
            effectiveness_by_state[state.value].append(adaptation.get('effectiveness_score', 0.0))
        
        # Calculate average effectiveness
        avg_effectiveness = {}
        for state, scores in effectiveness_by_state.items():
            avg_effectiveness[state] = np.mean(scores) if scores else 0.0
        
        return {
            "total_adaptations": len(self.adaptation_history),
            "active_adaptations": len(self.active_adaptations),
            "effectiveness_by_state": avg_effectiveness,
            "most_effective_state": max(avg_effectiveness.keys(), key=lambda k: avg_effectiveness[k]) if avg_effectiveness else None,
            "adaptation_frequency": len(self.adaptation_history) / max(1, time.time() - (self.adaptation_history[0]['start_time'] if self.adaptation_history else time.time()))
        }
    
    def visualize_adaptation_analysis(self, save_path: str = None) -> str:
        """Create visualization of adaptation analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Emotional state history
        state_history = self.emotion_detector.get_state_history(3600)  # Last hour
        if state_history:
            timestamps = [entry['timestamp'] for entry in state_history]
            states = [entry['state'].value for entry in state_history]
            confidences = [entry['confidence'] for entry in state_history]
            
            # Create state timeline
            unique_states = list(set(states))
            state_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_states)))
            
            for i, state in enumerate(unique_states):
                state_times = [t for t, s in zip(timestamps, states) if s == state]
                state_confidences = [c for s, c in zip(states, confidences) if s == state]
                ax1.scatter(state_times, [i] * len(state_times), 
                           c=[state_colors[i]], s=[c*100 for c in state_confidences], alpha=0.7)
            
            ax1.set_yticks(range(len(unique_states)))
            ax1.set_yticklabels(unique_states)
            ax1.set_title('Emotional State Timeline')
            ax1.set_xlabel('Time')
        
        # Adaptation effectiveness
        effectiveness_data = self.get_adaptation_effectiveness()
        if 'effectiveness_by_state' in effectiveness_data:
            states = list(effectiveness_data['effectiveness_by_state'].keys())
            effectiveness = list(effectiveness_data['effectiveness_by_state'].values())
            
            ax2.bar(states, effectiveness, color='lightblue', alpha=0.7)
            ax2.set_title('Adaptation Effectiveness by State')
            ax2.set_ylabel('Effectiveness Score')
            ax2.tick_params(axis='x', rotation=45)
        
        # Personality trait radar chart
        traits = list(self.personality_profile.traits.keys())
        values = list(self.personality_profile.traits.values())
        
        if traits and values:
            angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False)
            values_plot = values + [values[0]]  # Complete the circle
            angles_plot = np.concatenate([angles, [angles[0]]])
            
            ax3.plot(angles_plot, values_plot, 'o-', linewidth=2, color='red', alpha=0.7)
            ax3.fill(angles_plot, values_plot, alpha=0.25, color='red')
            ax3.set_xticks(angles)
            ax3.set_xticklabels([trait.value for trait in traits])
            ax3.set_ylim(0, 1)
            ax3.set_title('Personality Profile')
            ax3.grid(True)
        
        # Strategy recommendation distribution
        if self.adaptation_history:
            strategies = [adaptation['rule'].target_strategy for adaptation in self.adaptation_history]
            strategy_counts = defaultdict(int)
            for strategy in strategies:
                strategy_counts[strategy] += 1
            
            strategy_names = list(strategy_counts.keys())
            counts = list(strategy_counts.values())
            
            ax4.pie(counts, labels=strategy_names, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Strategy Adaptation Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            save_path = f"/home/ubuntu/fusion-project/python-backend/visualizations/personality_adaptive_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        current_state, confidence = self.emotion_detector.get_current_state()
        
        return {
            "system_name": "Personality-Adaptive Play System",
            "current_emotional_state": current_state.value,
            "state_confidence": confidence,
            "detector_calibrated": self.emotion_detector.is_calibrated,
            "personality_learning_enabled": self.personality_learning,
            "active_adaptations": len(self.active_adaptations),
            "total_adaptations": len(self.adaptation_history),
            "personality_traits": {trait.value: value for trait, value in self.personality_profile.traits.items()},
            "adaptation_rules_count": len(self.adaptation_rules),
            "biometric_samples": len(self.emotion_detector.recent_measurements),
            "state_history_length": len(self.emotion_detector.state_history)
        }

# Example usage and testing
if __name__ == "__main__":
    # Create personality-adaptive strategy system
    adaptive_system = PersonalityAdaptiveStrategy({
        'adaptation_threshold': 0.6,
        'min_confidence_for_adaptation': 0.5
    })
    
    print("🎭 Personality-Adaptive Play System Test")
    
    # Simulate biometric data showing frustration
    print("\n😤 Simulating frustrated player behavior...")
    for i in range(50):
        # Frustrated behavior: high click pressure, tremor, slow decisions
        biometric_data = BiometricData(
            heart_rate=80 + random.uniform(10, 20),  # Elevated heart rate
            click_pressure=[random.uniform(0.8, 1.0) for _ in range(3)],  # High pressure
            mouse_positions=[(100 + random.uniform(-5, 5), 100 + random.uniform(-5, 5)) for _ in range(5)],  # Tremor
            decision_times=[random.uniform(2.0, 4.0) for _ in range(2)],  # Slow decisions
            interaction_intervals=[random.uniform(0.5, 1.5)],
            timestamp=time.time() + i * 0.1
        )
        adaptive_system.update_biometric_data(biometric_data)
    
    # Get strategy recommendation
    recommendation = adaptive_system.get_recommended_strategy()
    
    print(f"Emotional State: {recommendation['emotional_state']} (confidence: {recommendation['state_confidence']:.3f})")
    print(f"Recommended Strategy: {recommendation['strategy']}")
    print(f"Adaptation Reason: {recommendation['adaptation_reason']}")
    print(f"Active Adaptations: {recommendation['active_adaptations_count']}")
    
    if recommendation['personality_modifiers']:
        print(f"Personality Modifiers:")
        for modifier, value in recommendation['personality_modifiers'].items():
            print(f"  - {modifier}: {value:.3f}")
    
    # Simulate confident behavior
    print("\n😎 Simulating confident player behavior...")
    for i in range(30):
        # Confident behavior: normal heart rate, precise clicks, fast decisions
        biometric_data = BiometricData(
            heart_rate=70 + random.uniform(-5, 5),  # Normal heart rate
            click_pressure=[random.uniform(0.4, 0.6) for _ in range(3)],  # Normal pressure
            mouse_positions=[(100 + i, 100 + i) for _ in range(3)],  # Smooth movement
            decision_times=[random.uniform(0.3, 0.8) for _ in range(2)],  # Fast decisions
            interaction_intervals=[random.uniform(0.2, 0.5)],
            timestamp=time.time() + 50 * 0.1 + i * 0.1
        )
        adaptive_system.update_biometric_data(biometric_data)
    
    # Get updated recommendation
    recommendation = adaptive_system.get_recommended_strategy()
    
    print(f"Updated Emotional State: {recommendation['emotional_state']} (confidence: {recommendation['state_confidence']:.3f})")
    print(f"Updated Strategy: {recommendation['strategy']}")
    print(f"Adaptation Reason: {recommendation['adaptation_reason']}")
    
    # Get system statistics
    stats = adaptive_system.get_system_stats()
    print(f"\n📊 System Statistics:")
    for key, value in stats.items():
        if key != 'personality_traits':
            print(f"  {key}: {value}")
    
    print(f"\n🧠 Personality Traits:")
    for trait, value in stats['personality_traits'].items():
        print(f"  {trait}: {value:.3f}")
    
    # Get adaptation effectiveness
    effectiveness = adaptive_system.get_adaptation_effectiveness()
    print(f"\n📈 Adaptation Effectiveness:")
    for key, value in effectiveness.items():
        print(f"  {key}: {value}")
    
    # Create visualization
    viz_path = adaptive_system.visualize_adaptation_analysis()
    print(f"\n📊 Adaptation analysis visualization saved to: {viz_path}")
    
    print("\n🎯 Personality adaptation complete! The system now understands your emotional patterns!")

