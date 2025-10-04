"""
Rintaro Okabe Strategy - The Leader of Statistical War

This module implements the sixth strategy inspired by Rintaro Okabe from Steins;Gate,
combining game theory leadership with psychological manipulation and prediction.
Like a mad scientist who can read the convergence of probability lines!
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import time
from scipy.stats import beta, norm
from scipy.optimize import minimize
import math

@dataclass
class OpponentModel:
    """Model of an opponent's behavior and risk tolerance."""
    risk_tolerance: float = 0.5
    aggression_level: float = 0.5
    pattern_history: List[str] = field(default_factory=list)
    bluff_frequency: float = 0.1
    fold_threshold: float = 0.3
    confidence_level: float = 0.5
    last_actions: deque = field(default_factory=lambda: deque(maxlen=20))
    
    def update_model(self, action: str, outcome: bool, risk_level: float):
        """Update opponent model based on observed action and outcome."""
        self.last_actions.append((action, outcome, risk_level))
        self.pattern_history.append(action)
        
        # Update risk tolerance based on observed behavior
        if outcome and risk_level > 0.7:  # Successful high-risk action
            self.risk_tolerance = min(1.0, self.risk_tolerance + 0.05)
        elif not outcome and risk_level > 0.5:  # Failed moderate-risk action
            self.risk_tolerance = max(0.0, self.risk_tolerance - 0.03)
        
        # Update aggression level
        if action in ['aggressive_bet', 'all_in', 'high_risk']:
            self.aggression_level = min(1.0, self.aggression_level + 0.02)
        elif action in ['fold', 'conservative', 'cash_out']:
            self.aggression_level = max(0.0, self.aggression_level - 0.02)

@dataclass
class GameState:
    """Current state of the game for strategic analysis."""
    board_size: Tuple[int, int]
    mines_count: int
    revealed_cells: List[Tuple[int, int]]
    safe_cells: List[Tuple[int, int]]
    current_multiplier: float
    bankroll: float
    session_profit: float
    round_number: int
    time_elapsed: float
    
    def get_risk_level(self) -> float:
        """Calculate current risk level based on game state."""
        total_cells = self.board_size[0] * self.board_size[1]
        revealed_count = len(self.revealed_cells)
        remaining_cells = total_cells - revealed_count
        remaining_mines = self.mines_count - len([cell for cell in self.revealed_cells if cell not in self.safe_cells])
        
        if remaining_cells <= 0:
            return 0.0
        
        mine_probability = remaining_mines / remaining_cells
        return mine_probability

class RintaroOkabeStrategy:
    """
    The Leader of Statistical War - Rintaro Okabe Strategy
    
    Combines game theory leadership with psychological manipulation and prediction.
    Like a mad scientist who has mastered the art of Reading Steiner to predict
    the convergence of probability worldlines!
    
    Key Features:
    - Bluff modeling and psychological warfare
    - Opponent risk tolerance estimation
    - Nash equilibrium rebalancing
    - Multi-dimensional strategic analysis
    - Adversarial environment adaptation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Core strategy parameters
        self.psychological_weight = self.config.get('psychological_weight', 0.4)
        self.game_theory_weight = self.config.get('game_theory_weight', 0.3)
        self.prediction_weight = self.config.get('prediction_weight', 0.3)
        
        # Opponent modeling
        self.opponent_models = defaultdict(OpponentModel)
        self.current_opponent = "default"
        
        # Bluff and deception parameters
        self.bluff_probability = self.config.get('bluff_probability', 0.15)
        self.deception_threshold = self.config.get('deception_threshold', 0.6)
        self.psychological_pressure = 0.0
        
        # Nash equilibrium tracking
        self.strategy_frequencies = defaultdict(float)
        self.opponent_strategy_frequencies = defaultdict(float)
        self.nash_adjustment_rate = 0.05
        
        # Prediction and pattern recognition
        self.pattern_memory = deque(maxlen=1000)
        self.prediction_confidence = 0.5
        self.worldline_convergence = 0.0  # Okabe's special ability!
        
        # Performance tracking
        self.performance_history = []
        self.psychological_wins = 0
        self.total_decisions = 0
        
        # Mad scientist parameters
        self.lab_member_count = 5  # Number of strategies in the lab
        self.time_leap_probability = 0.1  # Probability of "changing" strategy mid-game
        self.reading_steiner_active = False  # Special prediction mode
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for Rintaro Okabe strategy."""
        return {
            'psychological_weight': 0.4,
            'game_theory_weight': 0.3,
            'prediction_weight': 0.3,
            'bluff_probability': 0.15,
            'deception_threshold': 0.6,
            'risk_tolerance': 0.65,
            'aggression_factor': 0.7,
            'adaptation_speed': 0.08,
            'confidence_threshold': 0.75,
            'nash_rebalancing': True,
            'psychological_warfare': True,
            'worldline_prediction': True
        }
    
    def make_decision(self, game_state: GameState, market_data: Dict = None) -> Dict[str, Any]:
        """
        Make a strategic decision using the power of Reading Steiner!
        
        Combines psychological analysis, game theory, and prediction to determine
        the optimal action in adversarial environments.
        """
        self.total_decisions += 1
        
        # Activate Reading Steiner for enhanced prediction
        if random.random() < 0.1:  # 10% chance to activate special ability
            self.reading_steiner_active = True
            self.worldline_convergence = self._calculate_worldline_convergence(game_state)
        
        # Multi-dimensional analysis
        psychological_analysis = self._psychological_analysis(game_state)
        game_theory_analysis = self._game_theory_analysis(game_state)
        prediction_analysis = self._prediction_analysis(game_state)
        
        # Weighted decision fusion
        decision_score = (
            self.psychological_weight * psychological_analysis['score'] +
            self.game_theory_weight * game_theory_analysis['score'] +
            self.prediction_weight * prediction_analysis['score']
        )
        
        # Apply Reading Steiner bonus if active
        if self.reading_steiner_active:
            decision_score *= (1 + self.worldline_convergence * 0.2)
            self.reading_steiner_active = False  # One-time use per activation
        
        # Determine action based on analysis
        action = self._determine_action(decision_score, game_state)
        
        # Update opponent model and strategy frequencies
        self._update_models(action, game_state)
        
        # Apply Nash equilibrium rebalancing if enabled
        if self.config.get('nash_rebalancing', True):
            action = self._nash_rebalance(action, game_state)
        
        # Record decision for pattern analysis
        self.pattern_memory.append({
            'game_state': game_state,
            'psychological_score': psychological_analysis['score'],
            'game_theory_score': game_theory_analysis['score'],
            'prediction_score': prediction_analysis['score'],
            'final_score': decision_score,
            'action': action,
            'timestamp': time.time()
        })
        
        return {
            'action': action,
            'confidence': self._calculate_confidence(decision_score),
            'reasoning': self._generate_reasoning(psychological_analysis, game_theory_analysis, prediction_analysis),
            'bluff_detected': psychological_analysis.get('bluff_detected', False),
            'nash_adjustment': game_theory_analysis.get('nash_adjustment', 0.0),
            'worldline_convergence': self.worldline_convergence,
            'lab_member_consensus': self._get_lab_member_consensus(game_state)
        }
    
    def _psychological_analysis(self, game_state: GameState) -> Dict[str, Any]:
        """
        Analyze the psychological aspects of the current situation.
        
        Like reading the micro-expressions and behavioral patterns of opponents
        to predict their next move!
        """
        opponent = self.opponent_models[self.current_opponent]
        
        # Calculate psychological pressure
        risk_level = game_state.get_risk_level()
        self.psychological_pressure = self._calculate_psychological_pressure(game_state, opponent)
        
        # Detect potential bluffs
        bluff_probability = self._detect_bluff(opponent, game_state)
        bluff_detected = bluff_probability > self.deception_threshold
        
        # Estimate opponent's emotional state
        emotional_state = self._estimate_emotional_state(opponent, game_state)
        
        # Calculate psychological advantage
        psychological_advantage = self._calculate_psychological_advantage(
            opponent, emotional_state, self.psychological_pressure
        )
        
        # Determine if we should apply psychological pressure
        apply_pressure = (
            psychological_advantage > 0.6 and 
            risk_level < 0.7 and 
            random.random() < self.bluff_probability
        )
        
        # Calculate final psychological score
        base_score = 0.5
        if bluff_detected:
            base_score += 0.3  # Exploit detected bluff
        if apply_pressure:
            base_score += 0.2  # Apply psychological pressure
        if psychological_advantage > 0.7:
            base_score += 0.15  # Strong psychological position
        
        # Adjust for opponent's risk tolerance
        if opponent.risk_tolerance < 0.3 and risk_level > 0.5:
            base_score += 0.1  # Exploit conservative opponent in risky situation
        
        return {
            'score': np.clip(base_score, 0.0, 1.0),
            'psychological_pressure': self.psychological_pressure,
            'bluff_detected': bluff_detected,
            'bluff_probability': bluff_probability,
            'emotional_state': emotional_state,
            'psychological_advantage': psychological_advantage,
            'apply_pressure': apply_pressure
        }
    
    def _game_theory_analysis(self, game_state: GameState) -> Dict[str, Any]:
        """
        Apply game theory principles for optimal strategic positioning.
        
        Calculate Nash equilibrium and optimal mixed strategies like a true
        strategic mastermind!
        """
        # Define strategy space
        strategies = ['aggressive', 'conservative', 'balanced', 'bluff', 'exploit']
        
        # Calculate payoff matrix based on current game state
        payoff_matrix = self._calculate_payoff_matrix(game_state, strategies)
        
        # Find Nash equilibrium
        nash_equilibrium = self._find_nash_equilibrium(payoff_matrix)
        
        # Calculate expected value for each strategy
        strategy_values = {}
        for i, strategy in enumerate(strategies):
            strategy_values[strategy] = np.sum(payoff_matrix[i] * nash_equilibrium)
        
        # Determine optimal strategy
        optimal_strategy = max(strategy_values, key=strategy_values.get)
        
        # Calculate Nash adjustment needed
        current_frequencies = np.array([self.strategy_frequencies[s] for s in strategies])
        nash_adjustment = np.linalg.norm(nash_equilibrium - current_frequencies)
        
        # Apply mixed strategy randomization
        if random.random() < 0.3:  # 30% chance to use mixed strategy
            strategy_choice = np.random.choice(strategies, p=nash_equilibrium)
        else:
            strategy_choice = optimal_strategy
        
        # Calculate game theory score
        base_score = strategy_values[strategy_choice]
        
        # Bonus for exploiting opponent weaknesses
        opponent = self.opponent_models[self.current_opponent]
        if strategy_choice == 'exploit' and opponent.confidence_level < 0.4:
            base_score += 0.2
        
        # Bonus for successful bluffing setup
        if strategy_choice == 'bluff' and self.psychological_pressure > 0.6:
            base_score += 0.15
        
        return {
            'score': np.clip(base_score, 0.0, 1.0),
            'optimal_strategy': optimal_strategy,
            'nash_equilibrium': nash_equilibrium.tolist(),
            'strategy_values': strategy_values,
            'nash_adjustment': nash_adjustment,
            'chosen_strategy': strategy_choice
        }
    
    def _prediction_analysis(self, game_state: GameState) -> Dict[str, Any]:
        """
        Predict future outcomes using pattern recognition and worldline analysis.
        
        Channel the power of Reading Steiner to see the convergence of probability!
        """
        # Pattern-based prediction
        pattern_prediction = self._pattern_based_prediction(game_state)
        
        # Probability-based prediction
        probability_prediction = self._probability_based_prediction(game_state)
        
        # Worldline convergence prediction (Okabe's special ability)
        worldline_prediction = self._worldline_convergence_prediction(game_state)
        
        # Opponent behavior prediction
        opponent_prediction = self._predict_opponent_behavior(game_state)
        
        # Combine predictions with confidence weighting
        predictions = [pattern_prediction, probability_prediction, worldline_prediction, opponent_prediction]
        weights = [0.25, 0.35, 0.2, 0.2]
        
        combined_prediction = sum(p * w for p, w in zip(predictions, weights))
        
        # Calculate prediction confidence
        prediction_variance = np.var(predictions)
        self.prediction_confidence = max(0.1, 1.0 - prediction_variance)
        
        # Adjust score based on confidence
        final_score = combined_prediction * self.prediction_confidence
        
        return {
            'score': np.clip(final_score, 0.0, 1.0),
            'pattern_prediction': pattern_prediction,
            'probability_prediction': probability_prediction,
            'worldline_prediction': worldline_prediction,
            'opponent_prediction': opponent_prediction,
            'combined_prediction': combined_prediction,
            'prediction_confidence': self.prediction_confidence,
            'prediction_variance': prediction_variance
        }
    
    def _calculate_worldline_convergence(self, game_state: GameState) -> float:
        """
        Calculate the convergence of probability worldlines.
        
        Okabe's special ability to perceive the flow of causality and probability!
        """
        # Base convergence on multiple factors
        risk_level = game_state.get_risk_level()
        session_performance = game_state.session_profit / max(1, game_state.bankroll)
        time_factor = min(1.0, game_state.time_elapsed / 3600)  # Normalize to hours
        
        # Pattern recognition strength
        pattern_strength = len(self.pattern_memory) / 1000.0
        
        # Opponent predictability
        opponent = self.opponent_models[self.current_opponent]
        opponent_predictability = 1.0 - np.std([action[2] for action in opponent.last_actions]) if opponent.last_actions else 0.5
        
        # Calculate convergence
        convergence = (
            0.3 * (1.0 - risk_level) +  # Lower risk = higher convergence
            0.2 * max(0, session_performance) +  # Positive performance increases convergence
            0.2 * pattern_strength +  # More patterns = better convergence
            0.2 * opponent_predictability +  # Predictable opponent = higher convergence
            0.1 * time_factor  # Time increases understanding
        )
        
        return np.clip(convergence, 0.0, 1.0)
    
    def _calculate_psychological_pressure(self, game_state: GameState, opponent: OpponentModel) -> float:
        """Calculate the psychological pressure in the current situation."""
        risk_level = game_state.get_risk_level()
        multiplier_pressure = min(1.0, game_state.current_multiplier / 10.0)
        bankroll_pressure = max(0, (game_state.bankroll - 1000) / 5000)  # Normalized bankroll pressure
        
        # Opponent-specific pressure
        opponent_pressure = 0.0
        if opponent.risk_tolerance < 0.4 and risk_level > 0.5:
            opponent_pressure = 0.3  # High pressure on conservative opponent in risky situation
        elif opponent.aggression_level > 0.7 and risk_level < 0.3:
            opponent_pressure = 0.2  # Moderate pressure on aggressive opponent in safe situation
        
        total_pressure = (
            0.4 * risk_level +
            0.3 * multiplier_pressure +
            0.2 * bankroll_pressure +
            0.1 * opponent_pressure
        )
        
        return np.clip(total_pressure, 0.0, 1.0)
    
    def _detect_bluff(self, opponent: OpponentModel, game_state: GameState) -> float:
        """Detect if the opponent is bluffing based on behavioral patterns."""
        if len(opponent.last_actions) < 5:
            return 0.5  # Insufficient data
        
        # Analyze recent action patterns
        recent_actions = list(opponent.last_actions)[-5:]
        risk_levels = [action[2] for action in recent_actions]
        outcomes = [action[1] for action in recent_actions]
        
        # Look for inconsistent risk-taking patterns
        risk_variance = np.var(risk_levels)
        success_rate = sum(outcomes) / len(outcomes)
        
        # High variance with low success rate suggests bluffing
        bluff_indicator = risk_variance * (1.0 - success_rate)
        
        # Adjust based on opponent's known bluff frequency
        adjusted_probability = (bluff_indicator + opponent.bluff_frequency) / 2
        
        return np.clip(adjusted_probability, 0.0, 1.0)
    
    def _estimate_emotional_state(self, opponent: OpponentModel, game_state: GameState) -> str:
        """Estimate the opponent's emotional state based on recent performance."""
        if len(opponent.last_actions) < 3:
            return "neutral"
        
        recent_outcomes = [action[1] for action in list(opponent.last_actions)[-3:]]
        win_rate = sum(recent_outcomes) / len(recent_outcomes)
        
        if win_rate >= 0.8:
            return "confident"
        elif win_rate >= 0.6:
            return "optimistic"
        elif win_rate >= 0.4:
            return "neutral"
        elif win_rate >= 0.2:
            return "frustrated"
        else:
            return "desperate"
    
    def _calculate_psychological_advantage(self, opponent: OpponentModel, emotional_state: str, pressure: float) -> float:
        """Calculate our psychological advantage over the opponent."""
        base_advantage = 0.5
        
        # Adjust based on emotional state
        emotional_adjustments = {
            "confident": -0.2,  # Confident opponents are harder to manipulate
            "optimistic": -0.1,
            "neutral": 0.0,
            "frustrated": 0.2,  # Frustrated opponents make mistakes
            "desperate": 0.3   # Desperate opponents are most exploitable
        }
        
        base_advantage += emotional_adjustments.get(emotional_state, 0.0)
        
        # Adjust based on pressure
        if pressure > 0.7:
            base_advantage += 0.15  # High pressure situations favor psychological warfare
        
        # Adjust based on opponent's confidence level
        base_advantage += (1.0 - opponent.confidence_level) * 0.2
        
        return np.clip(base_advantage, 0.0, 1.0)
    
    def _calculate_payoff_matrix(self, game_state: GameState, strategies: List[str]) -> np.ndarray:
        """Calculate the payoff matrix for game theory analysis."""
        n_strategies = len(strategies)
        payoff_matrix = np.zeros((n_strategies, n_strategies))
        
        risk_level = game_state.get_risk_level()
        multiplier = game_state.current_multiplier
        
        # Define base payoffs for each strategy combination
        strategy_values = {
            'aggressive': 0.8 if risk_level < 0.5 else 0.3,
            'conservative': 0.6 if risk_level > 0.5 else 0.4,
            'balanced': 0.5,
            'bluff': 0.7 if self.psychological_pressure > 0.6 else 0.2,
            'exploit': 0.8 if self.opponent_models[self.current_opponent].confidence_level < 0.4 else 0.3
        }
        
        # Fill payoff matrix
        for i, strategy_i in enumerate(strategies):
            for j, strategy_j in enumerate(strategies):
                # Base payoff
                payoff = strategy_values[strategy_i]
                
                # Interaction effects
                if strategy_i == 'aggressive' and strategy_j == 'conservative':
                    payoff += 0.2  # Aggressive beats conservative
                elif strategy_i == 'bluff' and strategy_j == 'aggressive':
                    payoff += 0.15  # Bluff can beat aggressive
                elif strategy_i == 'exploit' and strategy_j == 'bluff':
                    payoff += 0.1  # Exploit beats bluff
                
                payoff_matrix[i, j] = payoff
        
        return payoff_matrix
    
    def _find_nash_equilibrium(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """Find Nash equilibrium for the given payoff matrix."""
        n_strategies = payoff_matrix.shape[0]
        
        # For simplicity, use iterative best response
        # In a real implementation, you'd use linear programming
        strategy_probs = np.ones(n_strategies) / n_strategies
        
        for _ in range(10):  # Iterate to convergence
            # Calculate best responses
            expected_payoffs = payoff_matrix @ strategy_probs
            best_response = np.zeros(n_strategies)
            best_response[np.argmax(expected_payoffs)] = 1.0
            
            # Update with learning rate
            strategy_probs = 0.9 * strategy_probs + 0.1 * best_response
        
        return strategy_probs
    
    def _pattern_based_prediction(self, game_state: GameState) -> float:
        """Predict outcome based on historical patterns."""
        if len(self.pattern_memory) < 10:
            return 0.5
        
        # Find similar game states
        similar_states = []
        current_risk = game_state.get_risk_level()
        
        for memory in self.pattern_memory:
            past_risk = memory['game_state'].get_risk_level()
            if abs(current_risk - past_risk) < 0.1:  # Similar risk level
                similar_states.append(memory)
        
        if not similar_states:
            return 0.5
        
        # Calculate average success rate for similar states
        success_rate = sum(1 for state in similar_states if state['action'] == 'continue') / len(similar_states)
        
        return success_rate
    
    def _probability_based_prediction(self, game_state: GameState) -> float:
        """Predict outcome based on pure probability calculations."""
        risk_level = game_state.get_risk_level()
        safety_probability = 1.0 - risk_level
        
        # Adjust for current multiplier (higher multiplier = more tempting to continue)
        multiplier_factor = min(1.0, game_state.current_multiplier / 5.0)
        
        # Kelly criterion consideration
        if safety_probability > 0.5:
            kelly_factor = (safety_probability * game_state.current_multiplier - (1 - safety_probability)) / game_state.current_multiplier
            kelly_factor = max(0, min(1, kelly_factor))
        else:
            kelly_factor = 0
        
        # Combine factors
        prediction = (0.6 * safety_probability + 0.3 * multiplier_factor + 0.1 * kelly_factor)
        
        return np.clip(prediction, 0.0, 1.0)
    
    def _worldline_convergence_prediction(self, game_state: GameState) -> float:
        """Predict using worldline convergence analysis (Okabe's special ability)."""
        convergence = self.worldline_convergence
        
        # Higher convergence = more accurate prediction
        if convergence > 0.8:
            # High convergence: trust probability calculations more
            return self._probability_based_prediction(game_state)
        elif convergence > 0.6:
            # Medium convergence: balance probability and patterns
            prob_pred = self._probability_based_prediction(game_state)
            pattern_pred = self._pattern_based_prediction(game_state)
            return 0.6 * prob_pred + 0.4 * pattern_pred
        else:
            # Low convergence: rely on intuition (random with slight bias)
            return 0.5 + random.uniform(-0.2, 0.2)
    
    def _predict_opponent_behavior(self, game_state: GameState) -> float:
        """Predict opponent's likely behavior in this situation."""
        opponent = self.opponent_models[self.current_opponent]
        
        # Base prediction on opponent's risk tolerance
        risk_level = game_state.get_risk_level()
        
        if risk_level > opponent.risk_tolerance:
            # Opponent likely to fold/cash out
            return 0.3
        elif risk_level < opponent.risk_tolerance * 0.5:
            # Opponent likely to continue
            return 0.8
        else:
            # Uncertain situation
            return 0.5
    
    def _determine_action(self, decision_score: float, game_state: GameState) -> str:
        """Determine the final action based on decision score."""
        risk_level = game_state.get_risk_level()
        
        # Apply Okabe's decision thresholds
        if decision_score > 0.8:
            return "continue_aggressive"
        elif decision_score > 0.65:
            return "continue"
        elif decision_score > 0.45:
            if risk_level < 0.3:
                return "continue"
            else:
                return "evaluate"  # Take time to think
        elif decision_score > 0.3:
            return "cash_out"
        else:
            return "cash_out_immediately"
    
    def _update_models(self, action: str, game_state: GameState):
        """Update opponent models and strategy frequencies."""
        # Update strategy frequencies
        strategy_type = self._classify_action(action)
        self.strategy_frequencies[strategy_type] += 1
        
        # Normalize frequencies
        total_actions = sum(self.strategy_frequencies.values())
        for strategy in self.strategy_frequencies:
            self.strategy_frequencies[strategy] /= total_actions
        
        # Update opponent model (simulated)
        opponent = self.opponent_models[self.current_opponent]
        simulated_outcome = random.random() < (1.0 - game_state.get_risk_level())
        opponent.update_model(action, simulated_outcome, game_state.get_risk_level())
    
    def _classify_action(self, action: str) -> str:
        """Classify action into strategy type."""
        if "aggressive" in action:
            return "aggressive"
        elif "cash_out" in action:
            return "conservative"
        elif "evaluate" in action:
            return "balanced"
        else:
            return "balanced"
    
    def _nash_rebalance(self, action: str, game_state: GameState) -> str:
        """Apply Nash equilibrium rebalancing to the action."""
        # Simple rebalancing: occasionally switch to less frequent strategies
        if random.random() < self.nash_adjustment_rate:
            # Find least used strategy
            min_strategy = min(self.strategy_frequencies, key=self.strategy_frequencies.get)
            
            # Convert to action
            strategy_to_action = {
                "aggressive": "continue_aggressive",
                "conservative": "cash_out",
                "balanced": "continue"
            }
            
            return strategy_to_action.get(min_strategy, action)
        
        return action
    
    def _calculate_confidence(self, decision_score: float) -> float:
        """Calculate confidence in the decision."""
        base_confidence = abs(decision_score - 0.5) * 2  # Distance from neutral
        
        # Adjust for worldline convergence
        convergence_bonus = self.worldline_convergence * 0.3
        
        # Adjust for prediction confidence
        prediction_bonus = self.prediction_confidence * 0.2
        
        total_confidence = base_confidence + convergence_bonus + prediction_bonus
        
        return np.clip(total_confidence, 0.1, 1.0)
    
    def _generate_reasoning(self, psychological: Dict, game_theory: Dict, prediction: Dict) -> str:
        """Generate human-readable reasoning for the decision."""
        reasoning_parts = []
        
        # Psychological reasoning
        if psychological['bluff_detected']:
            reasoning_parts.append("Detected opponent bluff - exploiting weakness")
        if psychological['apply_pressure']:
            reasoning_parts.append("Applying psychological pressure")
        
        # Game theory reasoning
        reasoning_parts.append(f"Nash optimal strategy: {game_theory['optimal_strategy']}")
        
        # Prediction reasoning
        if self.reading_steiner_active:
            reasoning_parts.append("Reading Steiner activated - enhanced prediction")
        reasoning_parts.append(f"Prediction confidence: {prediction['prediction_confidence']:.2f}")
        
        return " | ".join(reasoning_parts)
    
    def _get_lab_member_consensus(self, game_state: GameState) -> Dict[str, str]:
        """Get consensus from other lab members (strategies)."""
        # Simulate what other strategies would do
        lab_members = {
            "Kurisu": "analytical_approach",
            "Mayuri": "intuitive_approach", 
            "Daru": "technical_approach",
            "Suzuha": "tactical_approach",
            "Faris": "charismatic_approach"
        }
        
        consensus = {}
        for member, approach in lab_members.items():
            # Simulate their decision
            if approach == "analytical_approach":
                consensus[member] = "continue" if game_state.get_risk_level() < 0.4 else "cash_out"
            elif approach == "intuitive_approach":
                consensus[member] = "continue" if random.random() > 0.5 else "cash_out"
            elif approach == "technical_approach":
                consensus[member] = "continue" if game_state.current_multiplier < 3.0 else "cash_out"
            elif approach == "tactical_approach":
                consensus[member] = "continue" if game_state.session_profit > 0 else "cash_out"
            else:  # charismatic_approach
                consensus[member] = "continue" if self.psychological_pressure > 0.5 else "cash_out"
        
        return consensus
    
    def update_performance(self, outcome: bool, profit: float):
        """Update performance tracking."""
        self.performance_history.append({
            'outcome': outcome,
            'profit': profit,
            'timestamp': time.time()
        })
        
        if outcome and self.psychological_pressure > 0.6:
            self.psychological_wins += 1
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics."""
        if not self.performance_history:
            return {"status": "No performance data available"}
        
        recent_performance = self.performance_history[-100:]  # Last 100 decisions
        win_rate = sum(1 for p in recent_performance if p['outcome']) / len(recent_performance)
        avg_profit = sum(p['profit'] for p in recent_performance) / len(recent_performance)
        
        psychological_win_rate = self.psychological_wins / max(1, self.total_decisions)
        
        return {
            "strategy_name": "Rintaro Okabe - Leader of Statistical War",
            "total_decisions": self.total_decisions,
            "win_rate": win_rate,
            "average_profit": avg_profit,
            "psychological_win_rate": psychological_win_rate,
            "worldline_convergence": self.worldline_convergence,
            "prediction_confidence": self.prediction_confidence,
            "psychological_pressure": self.psychological_pressure,
            "strategy_frequencies": dict(self.strategy_frequencies),
            "reading_steiner_activations": sum(1 for m in self.pattern_memory if 'worldline_prediction' in str(m)),
            "lab_member_count": self.lab_member_count,
            "mad_scientist_level": min(100, self.total_decisions // 10)  # Level up every 10 decisions
        }

# Example usage and testing
if __name__ == "__main__":
    # Create Rintaro Okabe strategy
    okabe = RintaroOkabeStrategy()
    
    # Test with sample game state
    test_game_state = GameState(
        board_size=(5, 5),
        mines_count=6,
        revealed_cells=[(0, 0), (0, 1), (1, 0)],
        safe_cells=[(0, 0), (0, 1), (1, 0)],
        current_multiplier=2.5,
        bankroll=1500.0,
        session_profit=250.0,
        round_number=15,
        time_elapsed=1800.0
    )
    
    # Make decision
    decision = okabe.make_decision(test_game_state)
    
    print("🧪 Rintaro Okabe Strategy Test Results:")
    print(f"Action: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.3f}")
    print(f"Reasoning: {decision['reasoning']}")
    print(f"Worldline Convergence: {decision['worldline_convergence']:.3f}")
    print(f"Lab Member Consensus: {decision['lab_member_consensus']}")
    
    # Update performance and get stats
    okabe.update_performance(True, 125.0)
    stats = okabe.get_strategy_stats()
    
    print("\n📊 Strategy Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n🎯 El Psy Kongroo! The choice of Steins Gate has been made!")

