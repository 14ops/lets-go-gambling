"""
Strategic Improvements Module

This module implements advanced strategic improvements including:
1. Hybrid Strategy Mode (Senku + Lelouch Fusion)
2. Context-Aware Risk Scaling
3. Multi-Board Portfolio Play

Like combining the analytical prowess of Senku with the tactical genius of Lelouch!
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import random
from datetime import datetime, timedelta

class RiskMode(Enum):
    """Risk scaling modes based on bankroll performance."""
    ULTRA_CONSERVATIVE = "ultra_conservative"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ULTRA_AGGRESSIVE = "ultra_aggressive"

@dataclass
class GameState:
    """Represents the current state of a game board."""
    board_size: int
    mine_count: int
    revealed_cells: List[Tuple[int, int]]
    safe_cells: List[Tuple[int, int]]
    current_multiplier: float
    is_active: bool
    expected_value: float

@dataclass
class PortfolioState:
    """Represents the state of multiple concurrent games."""
    games: List[GameState]
    total_bankroll: float
    starting_bankroll: float
    current_session_profit: float
    risk_mode: RiskMode
    hybrid_confidence: float

class HybridStrategy:
    """
    Fusion of Senku's analytical learning with Lelouch's calculated adaptive play.
    Like combining a brilliant scientist with a master strategist!
    """
    
    def __init__(self, learning_rate: float = 0.01, adaptation_speed: float = 0.05):
        self.learning_rate = learning_rate
        self.adaptation_speed = adaptation_speed
        
        # Senku's analytical components
        self.pattern_memory = {}
        self.success_rates = {}
        self.confidence_scores = {}
        
        # Lelouch's strategic components
        self.risk_tolerance = 0.5
        self.adaptation_history = []
        self.strategic_adjustments = {}
        
        # Hybrid fusion parameters
        self.analytical_weight = 0.6  # How much to trust Senku's analysis
        self.strategic_weight = 0.4   # How much to trust Lelouch's strategy
        
    def analyze_pattern(self, board_state: GameState) -> Dict[str, float]:
        """Senku's pattern analysis component."""
        pattern_key = f"{board_state.board_size}x{board_state.board_size}_{board_state.mine_count}"
        
        if pattern_key not in self.pattern_memory:
            self.pattern_memory[pattern_key] = {
                'safe_positions': {},
                'danger_zones': {},
                'optimal_paths': []
            }
        
        # Analyze current board configuration
        total_cells = board_state.board_size ** 2
        revealed_count = len(board_state.revealed_cells)
        safe_count = len(board_state.safe_cells)
        
        # Calculate pattern confidence
        if revealed_count > 0:
            success_rate = safe_count / revealed_count
            self.success_rates[pattern_key] = success_rate
            
            # Update confidence based on historical performance
            if pattern_key in self.confidence_scores:
                old_confidence = self.confidence_scores[pattern_key]
                new_confidence = old_confidence + self.learning_rate * (success_rate - old_confidence)
                self.confidence_scores[pattern_key] = min(1.0, max(0.0, new_confidence))
            else:
                self.confidence_scores[pattern_key] = success_rate
        
        return {
            'pattern_confidence': self.confidence_scores.get(pattern_key, 0.5),
            'success_rate': self.success_rates.get(pattern_key, 0.5),
            'recommended_cells': self._get_recommended_cells(board_state)
        }
    
    def strategic_adaptation(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """Lelouch's strategic adaptation component."""
        
        # Calculate performance metrics
        bankroll_change = (portfolio_state.total_bankroll - portfolio_state.starting_bankroll) / portfolio_state.starting_bankroll
        session_performance = portfolio_state.current_session_profit / portfolio_state.starting_bankroll
        
        # Lelouch's strategic assessment
        strategic_factors = {
            'bankroll_pressure': self._assess_bankroll_pressure(bankroll_change),
            'momentum': self._assess_momentum(session_performance),
            'risk_adjustment': self._calculate_risk_adjustment(portfolio_state),
            'tactical_recommendation': self._get_tactical_recommendation(portfolio_state)
        }
        
        # Update adaptation history
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'bankroll_change': bankroll_change,
            'strategic_factors': strategic_factors
        })
        
        # Keep only recent history
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]
        
        return strategic_factors
    
    def make_hybrid_decision(self, board_state: GameState, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """Combine Senku's analysis with Lelouch's strategy for optimal decision."""
        
        # Get Senku's analytical assessment
        analytical_assessment = self.analyze_pattern(board_state)
        
        # Get Lelouch's strategic assessment
        strategic_assessment = self.strategic_adaptation(portfolio_state)
        
        # Fusion algorithm - weight the assessments
        analytical_confidence = analytical_assessment['pattern_confidence']
        strategic_confidence = strategic_assessment['risk_adjustment']
        
        # Dynamic weight adjustment based on confidence levels
        if analytical_confidence > 0.8:
            self.analytical_weight = min(0.8, self.analytical_weight + 0.05)
        elif analytical_confidence < 0.3:
            self.analytical_weight = max(0.2, self.analytical_weight - 0.05)
        
        # Calculate hybrid confidence
        hybrid_confidence = (
            self.analytical_weight * analytical_confidence +
            self.strategic_weight * strategic_confidence
        )
        
        # Make decision based on hybrid assessment
        decision = {
            'action': self._determine_action(board_state, hybrid_confidence),
            'confidence': hybrid_confidence,
            'analytical_component': analytical_assessment,
            'strategic_component': strategic_assessment,
            'recommended_bet_size': self._calculate_bet_size(portfolio_state, hybrid_confidence),
            'cash_out_threshold': self._calculate_cash_out_threshold(board_state, hybrid_confidence)
        }
        
        return decision
    
    def _get_recommended_cells(self, board_state: GameState) -> List[Tuple[int, int]]:
        """Get recommended cells to click based on pattern analysis."""
        all_cells = [(i, j) for i in range(board_state.board_size) for j in range(board_state.board_size)]
        unrevealed_cells = [cell for cell in all_cells if cell not in board_state.revealed_cells]
        
        # Prioritize corner and edge cells (generally safer)
        corner_cells = [(0, 0), (0, board_state.board_size-1), 
                       (board_state.board_size-1, 0), (board_state.board_size-1, board_state.board_size-1)]
        
        edge_cells = []
        for i in range(board_state.board_size):
            for j in range(board_state.board_size):
                if (i == 0 or i == board_state.board_size-1 or j == 0 or j == board_state.board_size-1) and (i, j) not in corner_cells:
                    edge_cells.append((i, j))
        
        # Filter available cells
        available_corners = [cell for cell in corner_cells if cell in unrevealed_cells]
        available_edges = [cell for cell in edge_cells if cell in unrevealed_cells]
        
        # Return prioritized list
        recommended = available_corners + available_edges[:3]  # Top 3 edge cells
        return recommended[:5]  # Return top 5 recommendations
    
    def _assess_bankroll_pressure(self, bankroll_change: float) -> float:
        """Assess pressure based on bankroll performance."""
        if bankroll_change > 0.2:
            return 0.2  # Low pressure, winning
        elif bankroll_change > 0:
            return 0.4  # Slight pressure
        elif bankroll_change > -0.1:
            return 0.6  # Moderate pressure
        elif bankroll_change > -0.2:
            return 0.8  # High pressure
        else:
            return 1.0  # Extreme pressure
    
    def _assess_momentum(self, session_performance: float) -> str:
        """Assess current momentum."""
        if session_performance > 0.1:
            return "strong_positive"
        elif session_performance > 0.05:
            return "positive"
        elif session_performance > -0.05:
            return "neutral"
        elif session_performance > -0.1:
            return "negative"
        else:
            return "strong_negative"
    
    def _calculate_risk_adjustment(self, portfolio_state: PortfolioState) -> float:
        """Calculate risk adjustment factor."""
        bankroll_ratio = portfolio_state.total_bankroll / portfolio_state.starting_bankroll
        
        if bankroll_ratio > 1.2:
            return 0.8  # Can afford more risk
        elif bankroll_ratio > 1.1:
            return 0.6
        elif bankroll_ratio > 0.9:
            return 0.5  # Balanced
        elif bankroll_ratio > 0.8:
            return 0.3
        else:
            return 0.1  # Very conservative
    
    def _get_tactical_recommendation(self, portfolio_state: PortfolioState) -> str:
        """Get Lelouch's tactical recommendation."""
        if portfolio_state.risk_mode == RiskMode.ULTRA_AGGRESSIVE:
            return "maximum_aggression"
        elif portfolio_state.risk_mode == RiskMode.AGGRESSIVE:
            return "calculated_aggression"
        elif portfolio_state.risk_mode == RiskMode.BALANCED:
            return "balanced_approach"
        elif portfolio_state.risk_mode == RiskMode.CONSERVATIVE:
            return "defensive_play"
        else:
            return "survival_mode"
    
    def _determine_action(self, board_state: GameState, confidence: float) -> str:
        """Determine the action to take."""
        if confidence > 0.8:
            return "aggressive_reveal"
        elif confidence > 0.6:
            return "calculated_reveal"
        elif confidence > 0.4:
            return "conservative_reveal"
        elif confidence > 0.2:
            return "minimal_reveal"
        else:
            return "cash_out"
    
    def _calculate_bet_size(self, portfolio_state: PortfolioState, confidence: float) -> float:
        """Calculate optimal bet size based on Kelly Criterion and confidence."""
        base_bet = portfolio_state.total_bankroll * 0.02  # 2% base bet
        
        # Adjust based on confidence and risk mode
        confidence_multiplier = 0.5 + confidence
        
        risk_multipliers = {
            RiskMode.ULTRA_CONSERVATIVE: 0.3,
            RiskMode.CONSERVATIVE: 0.6,
            RiskMode.BALANCED: 1.0,
            RiskMode.AGGRESSIVE: 1.5,
            RiskMode.ULTRA_AGGRESSIVE: 2.0
        }
        
        risk_multiplier = risk_multipliers[portfolio_state.risk_mode]
        
        optimal_bet = base_bet * confidence_multiplier * risk_multiplier
        
        # Cap at 10% of bankroll for safety
        max_bet = portfolio_state.total_bankroll * 0.1
        return min(optimal_bet, max_bet)
    
    def _calculate_cash_out_threshold(self, board_state: GameState, confidence: float) -> float:
        """Calculate when to cash out based on risk/reward."""
        base_threshold = board_state.current_multiplier * 1.2  # 20% profit target
        
        # Adjust based on confidence
        if confidence > 0.8:
            return base_threshold * 1.5  # Higher target when confident
        elif confidence < 0.3:
            return base_threshold * 0.8  # Lower target when uncertain
        else:
            return base_threshold

class ContextAwareRiskManager:
    """
    Manages risk scaling based on bankroll performance and market conditions.
    Like having a financial advisor who adapts to your success and failures!
    """
    
    def __init__(self, initial_bankroll: float):
        self.initial_bankroll = initial_bankroll
        self.risk_thresholds = {
            'ultra_aggressive': 0.2,   # +20% bankroll
            'aggressive': 0.1,         # +10% bankroll
            'balanced': 0.0,           # Break-even
            'conservative': -0.1,      # -10% bankroll
            'ultra_conservative': -0.2 # -20% bankroll
        }
        
        self.performance_history = []
        self.current_streak = 0
        self.streak_type = None  # 'winning' or 'losing'
        
    def assess_risk_mode(self, current_bankroll: float, session_performance: float) -> RiskMode:
        """Determine appropriate risk mode based on current state."""
        
        bankroll_change = (current_bankroll - self.initial_bankroll) / self.initial_bankroll
        
        # Update performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'bankroll': current_bankroll,
            'change': bankroll_change,
            'session_performance': session_performance
        })
        
        # Keep only recent history (last 100 records)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Update streak information
        self._update_streak(session_performance)
        
        # Determine risk mode
        if bankroll_change >= self.risk_thresholds['ultra_aggressive']:
            base_mode = RiskMode.ULTRA_AGGRESSIVE
        elif bankroll_change >= self.risk_thresholds['aggressive']:
            base_mode = RiskMode.AGGRESSIVE
        elif bankroll_change >= self.risk_thresholds['balanced']:
            base_mode = RiskMode.BALANCED
        elif bankroll_change >= self.risk_thresholds['conservative']:
            base_mode = RiskMode.CONSERVATIVE
        else:
            base_mode = RiskMode.ULTRA_CONSERVATIVE
        
        # Apply streak adjustments
        adjusted_mode = self._apply_streak_adjustment(base_mode)
        
        return adjusted_mode
    
    def _update_streak(self, session_performance: float):
        """Update winning/losing streak information."""
        if session_performance > 0:
            if self.streak_type == 'winning':
                self.current_streak += 1
            else:
                self.streak_type = 'winning'
                self.current_streak = 1
        elif session_performance < 0:
            if self.streak_type == 'losing':
                self.current_streak += 1
            else:
                self.streak_type = 'losing'
                self.current_streak = 1
        else:
            # Break even, maintain current streak
            pass
    
    def _apply_streak_adjustment(self, base_mode: RiskMode) -> RiskMode:
        """Adjust risk mode based on current streak."""
        
        if self.streak_type == 'losing' and self.current_streak >= 3:
            # Reduce risk during losing streaks
            if base_mode == RiskMode.ULTRA_AGGRESSIVE:
                return RiskMode.AGGRESSIVE
            elif base_mode == RiskMode.AGGRESSIVE:
                return RiskMode.BALANCED
            elif base_mode == RiskMode.BALANCED:
                return RiskMode.CONSERVATIVE
            else:
                return RiskMode.ULTRA_CONSERVATIVE
        
        elif self.streak_type == 'winning' and self.current_streak >= 5:
            # Slightly increase risk during long winning streaks (but be careful!)
            if base_mode == RiskMode.CONSERVATIVE:
                return RiskMode.BALANCED
            elif base_mode == RiskMode.BALANCED:
                return RiskMode.AGGRESSIVE
            # Don't escalate beyond aggressive during winning streaks
        
        return base_mode
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk assessment metrics."""
        if not self.performance_history:
            return {}
        
        recent_performance = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        # Calculate volatility
        changes = [record['change'] for record in recent_performance]
        volatility = np.std(changes) if len(changes) > 1 else 0
        
        # Calculate Sharpe ratio (simplified)
        avg_return = np.mean(changes) if changes else 0
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        return {
            'current_streak': self.current_streak,
            'streak_type': self.streak_type,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'recent_avg_return': avg_return
        }

class MultiBoardPortfolioManager:
    """
    Manages multiple concurrent game boards with different risk profiles.
    Like diversifying your investment portfolio across different asset classes!
    """
    
    def __init__(self, total_bankroll: float):
        self.total_bankroll = total_bankroll
        self.active_games = []
        self.allocation_strategy = "balanced"  # balanced, aggressive, conservative
        
        # Portfolio allocation percentages
        self.allocations = {
            "conservative": {"low_risk": 0.7, "medium_risk": 0.3, "high_risk": 0.0},
            "balanced": {"low_risk": 0.4, "medium_risk": 0.4, "high_risk": 0.2},
            "aggressive": {"low_risk": 0.2, "medium_risk": 0.3, "high_risk": 0.5}
        }
        
        # Risk profiles for different board configurations
        self.risk_profiles = {
            "low_risk": {"board_size": 5, "mine_count": 2, "target_multiplier": 1.5},
            "medium_risk": {"board_size": 5, "mine_count": 3, "target_multiplier": 2.0},
            "high_risk": {"board_size": 4, "mine_count": 3, "target_multiplier": 3.0}
        }
    
    def initialize_portfolio(self, allocation_strategy: str = "balanced") -> List[GameState]:
        """Initialize a diversified portfolio of games."""
        self.allocation_strategy = allocation_strategy
        allocation = self.allocations[allocation_strategy]
        
        portfolio_games = []
        
        for risk_level, percentage in allocation.items():
            if percentage > 0:
                profile = self.risk_profiles[risk_level]
                
                # Calculate bet size for this risk level
                bet_size = self.total_bankroll * percentage
                
                game_state = GameState(
                    board_size=profile["board_size"],
                    mine_count=profile["mine_count"],
                    revealed_cells=[],
                    safe_cells=[],
                    current_multiplier=1.0,
                    is_active=True,
                    expected_value=self._calculate_expected_value(profile)
                )
                
                portfolio_games.append(game_state)
        
        self.active_games = portfolio_games
        return portfolio_games
    
    def rebalance_portfolio(self, performance_data: Dict[str, float]) -> Dict[str, Any]:
        """Rebalance portfolio based on performance."""
        
        # Analyze performance of each risk level
        performance_analysis = {}
        
        for i, game in enumerate(self.active_games):
            risk_level = self._get_risk_level(game)
            if risk_level not in performance_analysis:
                performance_analysis[risk_level] = []
            
            # Calculate game performance
            game_performance = game.current_multiplier - 1.0  # Profit/loss
            performance_analysis[risk_level].append(game_performance)
        
        # Calculate average performance per risk level
        avg_performance = {}
        for risk_level, performances in performance_analysis.items():
            avg_performance[risk_level] = np.mean(performances) if performances else 0
        
        # Determine if rebalancing is needed
        rebalancing_decision = self._should_rebalance(avg_performance)
        
        if rebalancing_decision['should_rebalance']:
            new_allocation = self._calculate_new_allocation(avg_performance)
            self._apply_rebalancing(new_allocation)
            
            return {
                'rebalanced': True,
                'new_allocation': new_allocation,
                'reason': rebalancing_decision['reason'],
                'performance_analysis': avg_performance
            }
        
        return {
            'rebalanced': False,
            'current_allocation': self.allocations[self.allocation_strategy],
            'performance_analysis': avg_performance
        }
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio performance metrics."""
        
        if not self.active_games:
            return {}
        
        # Calculate portfolio-wide metrics
        total_value = sum(game.current_multiplier for game in self.active_games)
        portfolio_return = (total_value / len(self.active_games)) - 1.0
        
        # Calculate diversification benefit
        individual_risks = [self._calculate_game_risk(game) for game in self.active_games]
        portfolio_risk = np.std([game.current_multiplier for game in self.active_games])
        diversification_ratio = np.mean(individual_risks) / portfolio_risk if portfolio_risk > 0 else 1
        
        # Risk-adjusted return (Sharpe-like ratio)
        risk_adjusted_return = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'diversification_ratio': diversification_ratio,
            'risk_adjusted_return': risk_adjusted_return,
            'active_games': len(self.active_games),
            'total_exposure': total_value
        }
    
    def _calculate_expected_value(self, profile: Dict[str, Any]) -> float:
        """Calculate expected value for a game profile."""
        board_size = profile["board_size"]
        mine_count = profile["mine_count"]
        target_multiplier = profile["target_multiplier"]
        
        total_cells = board_size ** 2
        safe_cells = total_cells - mine_count
        
        # Simplified EV calculation
        # Probability of success for revealing one safe cell
        prob_success = safe_cells / total_cells
        
        # Expected value
        ev = prob_success * (target_multiplier - 1) - (1 - prob_success) * 1
        
        return ev
    
    def _get_risk_level(self, game: GameState) -> str:
        """Determine risk level of a game based on its configuration."""
        for risk_level, profile in self.risk_profiles.items():
            if (game.board_size == profile["board_size"] and 
                game.mine_count == profile["mine_count"]):
                return risk_level
        return "unknown"
    
    def _should_rebalance(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Determine if portfolio rebalancing is needed."""
        
        # Check for significant performance divergence
        if len(performance) < 2:
            return {'should_rebalance': False, 'reason': 'insufficient_data'}
        
        performance_values = list(performance.values())
        performance_spread = max(performance_values) - min(performance_values)
        
        # Rebalance if spread is too large
        if performance_spread > 0.3:  # 30% spread threshold
            return {'should_rebalance': True, 'reason': 'performance_divergence'}
        
        # Check for consistent underperformance
        underperforming_count = sum(1 for perf in performance_values if perf < -0.1)
        if underperforming_count >= len(performance_values) / 2:
            return {'should_rebalance': True, 'reason': 'widespread_underperformance'}
        
        return {'should_rebalance': False, 'reason': 'performance_within_tolerance'}
    
    def _calculate_new_allocation(self, performance: Dict[str, float]) -> Dict[str, float]:
        """Calculate new allocation based on performance."""
        
        # Simple momentum-based reallocation
        total_performance = sum(performance.values())
        
        if total_performance > 0:
            # Increase allocation to better performing assets
            new_allocation = {}
            for risk_level, perf in performance.items():
                weight = max(0.1, perf / total_performance) if total_performance > 0 else 1/len(performance)
                new_allocation[risk_level] = weight
            
            # Normalize to sum to 1
            total_weight = sum(new_allocation.values())
            new_allocation = {k: v/total_weight for k, v in new_allocation.items()}
            
        else:
            # Default to conservative allocation during poor performance
            new_allocation = self.allocations["conservative"]
        
        return new_allocation
    
    def _apply_rebalancing(self, new_allocation: Dict[str, float]):
        """Apply new allocation to the portfolio."""
        # This would involve adjusting bet sizes and potentially closing/opening positions
        # For now, we'll update the allocation strategy
        
        # Find closest matching predefined strategy
        min_distance = float('inf')
        closest_strategy = "balanced"
        
        for strategy_name, strategy_allocation in self.allocations.items():
            distance = sum(abs(new_allocation.get(risk, 0) - strategy_allocation.get(risk, 0)) 
                          for risk in ["low_risk", "medium_risk", "high_risk"])
            
            if distance < min_distance:
                min_distance = distance
                closest_strategy = strategy_name
        
        self.allocation_strategy = closest_strategy
    
    def _calculate_game_risk(self, game: GameState) -> float:
        """Calculate risk metric for individual game."""
        # Simple risk calculation based on mine density
        mine_density = game.mine_count / (game.board_size ** 2)
        return mine_density

def create_strategic_improvements_demo():
    """Create a demonstration of the strategic improvements."""
    
    # Initialize components
    initial_bankroll = 1000.0
    hybrid_strategy = HybridStrategy()
    risk_manager = ContextAwareRiskManager(initial_bankroll)
    portfolio_manager = MultiBoardPortfolioManager(initial_bankroll)
    
    # Initialize portfolio
    portfolio_games = portfolio_manager.initialize_portfolio("balanced")
    
    # Create portfolio state
    portfolio_state = PortfolioState(
        games=portfolio_games,
        total_bankroll=initial_bankroll,
        starting_bankroll=initial_bankroll,
        current_session_profit=0.0,
        risk_mode=RiskMode.BALANCED,
        hybrid_confidence=0.5
    )
    
    # Simulate some decisions
    results = []
    
    for round_num in range(10):
        # Simulate bankroll changes
        portfolio_state.total_bankroll += random.uniform(-50, 100)
        portfolio_state.current_session_profit = portfolio_state.total_bankroll - initial_bankroll
        
        # Update risk mode
        portfolio_state.risk_mode = risk_manager.assess_risk_mode(
            portfolio_state.total_bankroll, 
            portfolio_state.current_session_profit
        )
        
        # Make hybrid decision for first game
        if portfolio_state.games:
            game_state = portfolio_state.games[0]
            
            # Simulate some revealed cells
            game_state.revealed_cells.append((random.randint(0, 4), random.randint(0, 4)))
            game_state.current_multiplier += random.uniform(0.1, 0.3)
            
            decision = hybrid_strategy.make_hybrid_decision(game_state, portfolio_state)
            
            results.append({
                'round': round_num + 1,
                'bankroll': portfolio_state.total_bankroll,
                'risk_mode': portfolio_state.risk_mode.value,
                'decision': decision['action'],
                'confidence': decision['confidence'],
                'bet_size': decision['recommended_bet_size']
            })
    
    return results

if __name__ == "__main__":
    # Run demonstration
    demo_results = create_strategic_improvements_demo()
    
    print("Strategic Improvements Demonstration:")
    print("=" * 50)
    
    for result in demo_results:
        print(f"Round {result['round']}: Bankroll=${result['bankroll']:.2f}, "
              f"Risk={result['risk_mode']}, Action={result['decision']}, "
              f"Confidence={result['confidence']:.3f}, Bet=${result['bet_size']:.2f}")
    
    print("\nStrategic improvements successfully implemented!")
    print("✓ Hybrid Strategy (Senku + Lelouch Fusion)")
    print("✓ Context-Aware Risk Scaling")
    print("✓ Multi-Board Portfolio Management")

