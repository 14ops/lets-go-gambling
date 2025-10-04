"""
Advanced Strategic Layers Module

This module implements sophisticated strategic enhancements including
detection evasion, bankroll management, and positive EV hunting.
"""

import time
import random
import math
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class DetectionLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class EvasionConfig:
    """Configuration for detection evasion techniques."""
    min_delay: float = 0.5
    max_delay: float = 2.0
    mouse_drift_enabled: bool = True
    pause_probability: float = 0.1
    pause_duration_range: Tuple[float, float] = (3.0, 8.0)

class DetectionEvasion:
    """Implements various techniques to evade automated detection."""
    
    def __init__(self, config: EvasionConfig):
        self.config = config
        self.last_action_time = time.time()
        self.action_count = 0
        self.session_start_time = time.time()
    
    def apply_human_delay(self):
        """Apply randomized delays to mimic human reaction times."""
        base_delay = random.uniform(self.config.min_delay, self.config.max_delay)
        
        # Add slight variation based on action count (fatigue simulation)
        fatigue_factor = min(1.5, 1.0 + (self.action_count * 0.001))
        delay = base_delay * fatigue_factor
        
        # Occasionally add longer pauses
        if random.random() < self.config.pause_probability:
            pause_duration = random.uniform(*self.config.pause_duration_range)
            delay += pause_duration
            print(f"Taking a human-like pause for {pause_duration:.1f} seconds...")
        
        time.sleep(delay)
        self.last_action_time = time.time()
        self.action_count += 1
    
    def get_mouse_drift(self, target_x: int, target_y: int) -> Tuple[int, int]:
        """Add slight random drift to mouse coordinates."""
        if not self.config.mouse_drift_enabled:
            return target_x, target_y
        
        # Add small random offset (±3 pixels)
        drift_x = random.randint(-3, 3)
        drift_y = random.randint(-3, 3)
        
        return target_x + drift_x, target_y + drift_y
    
    def should_take_break(self) -> bool:
        """Determine if a longer break should be taken."""
        session_duration = time.time() - self.session_start_time
        
        # Take break every 30-45 minutes
        if session_duration > random.uniform(1800, 2700):  # 30-45 minutes
            return True
        
        # Take break after many actions
        if self.action_count > random.randint(200, 400):
            return True
        
        return False
    
    def take_session_break(self):
        """Take a longer break to simulate human behavior."""
        break_duration = random.uniform(300, 900)  # 5-15 minutes
        print(f"Taking a session break for {break_duration/60:.1f} minutes...")
        time.sleep(break_duration)
        
        # Reset counters
        self.session_start_time = time.time()
        self.action_count = 0

class AdvancedBankrollManager:
    """Enhanced bankroll management with dynamic risk adjustment."""
    
    def __init__(self, initial_bankroll: float, base_bet: float):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.base_bet = base_bet
        self.win_streak = 0
        self.loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.risk_level = 0.5  # 0.0 = very conservative, 1.0 = very aggressive
        
    def calculate_bet_size(self, win_probability: float, expected_multiplier: float) -> float:
        """Calculate optimal bet size using Kelly Criterion with modifications."""
        
        # Basic Kelly Criterion
        kelly_fraction = self._calculate_kelly_fraction(win_probability, expected_multiplier)
        
        # Apply risk adjustment based on current streak
        risk_adjusted_fraction = self._apply_risk_adjustment(kelly_fraction)
        
        # Calculate bet size
        bet_size = self.current_bankroll * risk_adjusted_fraction
        
        # Apply constraints
        min_bet = self.base_bet * 0.5
        max_bet = self.base_bet * 5.0
        
        return max(min_bet, min(max_bet, bet_size))
    
    def _calculate_kelly_fraction(self, win_prob: float, multiplier: float) -> float:
        """Calculate Kelly Criterion fraction."""
        if win_prob <= 0 or win_prob >= 1 or multiplier <= 1:
            return 0.01  # Minimum bet
        
        # Kelly formula: f = (bp - q) / b
        # where b = net odds, p = win probability, q = loss probability
        b = multiplier - 1  # Net odds
        p = win_prob
        q = 1 - win_prob
        
        kelly_fraction = (b * p - q) / b
        
        # Cap at 25% of bankroll for safety
        return max(0.01, min(0.25, kelly_fraction))
    
    def _apply_risk_adjustment(self, base_fraction: float) -> float:
        """Adjust bet size based on current performance and streaks."""
        
        # Reduce bet size during loss streaks
        if self.loss_streak > 3:
            streak_penalty = 0.8 ** (self.loss_streak - 3)
            base_fraction *= streak_penalty
        
        # Slightly increase bet size during moderate win streaks
        elif self.win_streak > 2:
            streak_bonus = min(1.2, 1.0 + (self.win_streak - 2) * 0.05)
            base_fraction *= streak_bonus
        
        # Apply overall risk level
        base_fraction *= (0.5 + self.risk_level * 0.5)
        
        return base_fraction
    
    def update_after_round(self, won: bool, bet_amount: float, payout: float):
        """Update bankroll and streaks after a round."""
        if won:
            self.current_bankroll += (payout - bet_amount)
            self.win_streak += 1
            self.loss_streak = 0
            self.max_win_streak = max(self.max_win_streak, self.win_streak)
        else:
            self.current_bankroll -= bet_amount
            self.loss_streak += 1
            self.win_streak = 0
            self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)
        
        # Adjust risk level based on performance
        self._adjust_risk_level()
    
    def _adjust_risk_level(self):
        """Dynamically adjust risk level based on performance."""
        current_roi = (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
        
        # Increase risk when performing well
        if current_roi > 0.1:  # 10% profit
            self.risk_level = min(1.0, self.risk_level + 0.05)
        # Decrease risk when losing
        elif current_roi < -0.05:  # 5% loss
            self.risk_level = max(0.1, self.risk_level - 0.1)

class PositiveEVHunter:
    """Identifies and exploits positive expected value opportunities."""
    
    def __init__(self):
        self.game_history = []
        self.strategy_performance = {}
        self.current_strategy = None
        
    def analyze_game_state(self, board_state: Dict[str, Any], available_strategies: List[str]) -> Dict[str, Any]:
        """Analyze current game state for EV opportunities."""
        
        analysis = {
            'recommended_strategy': None,
            'confidence': 0.0,
            'expected_value': 0.0,
            'risk_assessment': 'medium'
        }
        
        # Calculate basic probabilities
        cells_revealed = board_state['cells_revealed']
        safe_remaining = board_state['safe_cells_remaining']
        total_cells = board_state['board_size'] ** 2
        mines = board_state['mine_count']
        
        unrevealed_cells = total_cells - cells_revealed
        
        if unrevealed_cells <= 0 or safe_remaining <= 0:
            return analysis
        
        # Calculate probability of success for next reveal
        success_probability = safe_remaining / unrevealed_cells
        
        # Estimate expected multiplier (simplified)
        estimated_multiplier = self._estimate_multiplier(cells_revealed + 1, total_cells, mines)
        
        # Calculate expected value
        expected_value = success_probability * (estimated_multiplier - 1) - (1 - success_probability) * 1
        
        # Determine risk level
        if success_probability > 0.8:
            risk_assessment = 'low'
        elif success_probability > 0.6:
            risk_assessment = 'medium'
        else:
            risk_assessment = 'high'
        
        # Recommend strategy based on conditions
        recommended_strategy = self._recommend_strategy(success_probability, expected_value, risk_assessment)
        
        analysis.update({
            'recommended_strategy': recommended_strategy,
            'confidence': success_probability,
            'expected_value': expected_value,
            'risk_assessment': risk_assessment
        })
        
        return analysis
    
    def _estimate_multiplier(self, cells_revealed: int, total_cells: int, mines: int) -> float:
        """Estimate payout multiplier for given number of revealed cells."""
        if cells_revealed <= 0:
            return 1.0
        
        # Calculate probability of revealing N cells without hitting a mine
        probability = 1.0
        remaining_cells = total_cells
        remaining_mines = mines
        
        for i in range(cells_revealed):
            if remaining_cells <= 0:
                break
            safe_prob = max(0, (remaining_cells - remaining_mines) / remaining_cells)
            probability *= safe_prob
            remaining_cells -= 1
        
        if probability <= 0:
            return 1.0
        
        # Multiplier is inverse of probability with house edge
        house_edge = 0.02
        multiplier = (1.0 / probability) * (1.0 - house_edge)
        
        return max(1.01, multiplier)
    
    def _recommend_strategy(self, success_prob: float, expected_value: float, risk_level: str) -> str:
        """Recommend optimal strategy based on current conditions."""
        
        if expected_value <= 0:
            return 'kazuya'  # Conservative when EV is negative
        
        if risk_level == 'low' and expected_value > 0.1:
            return 'takeshi'  # Aggressive when low risk, high EV
        elif risk_level == 'medium' and expected_value > 0.05:
            return 'lelouch'  # Calculated approach for medium risk
        elif success_prob > 0.7:
            return 'senku'  # Analytical for high probability situations
        else:
            return 'kazuya'  # Conservative for uncertain situations
    
    def update_strategy_performance(self, strategy: str, result: bool, expected_value: float):
        """Update performance tracking for strategies."""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'total_rounds': 0,
                'wins': 0,
                'total_ev': 0.0,
                'actual_performance': 0.0
            }
        
        perf = self.strategy_performance[strategy]
        perf['total_rounds'] += 1
        if result:
            perf['wins'] += 1
        perf['total_ev'] += expected_value
        perf['actual_performance'] = perf['wins'] / perf['total_rounds']
    
    def get_best_performing_strategy(self) -> Optional[str]:
        """Get the strategy with the best actual performance."""
        if not self.strategy_performance:
            return None
        
        best_strategy = None
        best_performance = -1
        
        for strategy, perf in self.strategy_performance.items():
            if perf['total_rounds'] >= 10:  # Minimum sample size
                if perf['actual_performance'] > best_performance:
                    best_performance = perf['actual_performance']
                    best_strategy = strategy
        
        return best_strategy

class AdaptiveStrategyManager:
    """Manages dynamic strategy switching based on performance and conditions."""
    
    def __init__(self, available_strategies: List[str]):
        self.available_strategies = available_strategies
        self.current_strategy = available_strategies[0]
        self.strategy_stats = {strategy: {'rounds': 0, 'wins': 0, 'profit': 0.0} 
                              for strategy in available_strategies}
        self.switch_threshold = 0.1  # Switch if performance drops below this
        self.min_rounds_before_switch = 50
        
    def should_switch_strategy(self) -> bool:
        """Determine if strategy should be switched."""
        current_stats = self.strategy_stats[self.current_strategy]
        
        if current_stats['rounds'] < self.min_rounds_before_switch:
            return False
        
        current_win_rate = current_stats['wins'] / current_stats['rounds']
        
        # Switch if performance is poor
        if current_win_rate < self.switch_threshold:
            return True
        
        # Switch if another strategy is performing significantly better
        best_alternative = self._find_best_alternative()
        if best_alternative:
            alt_stats = self.strategy_stats[best_alternative]
            if alt_stats['rounds'] >= self.min_rounds_before_switch:
                alt_win_rate = alt_stats['wins'] / alt_stats['rounds']
                if alt_win_rate > current_win_rate + 0.05:  # 5% better
                    return True
        
        return False
    
    def _find_best_alternative(self) -> Optional[str]:
        """Find the best performing alternative strategy."""
        best_strategy = None
        best_win_rate = -1
        
        for strategy in self.available_strategies:
            if strategy == self.current_strategy:
                continue
            
            stats = self.strategy_stats[strategy]
            if stats['rounds'] >= self.min_rounds_before_switch:
                win_rate = stats['wins'] / stats['rounds']
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_strategy = strategy
        
        return best_strategy
    
    def switch_to_best_strategy(self):
        """Switch to the best performing strategy."""
        best_strategy = self._find_best_alternative()
        if best_strategy:
            print(f"Switching strategy from {self.current_strategy} to {best_strategy}")
            self.current_strategy = best_strategy
    
    def update_strategy_performance(self, strategy: str, won: bool, profit: float):
        """Update performance statistics for a strategy."""
        stats = self.strategy_stats[strategy]
        stats['rounds'] += 1
        if won:
            stats['wins'] += 1
        stats['profit'] += profit
    
    def get_strategy_report(self) -> Dict[str, Any]:
        """Get comprehensive strategy performance report."""
        report = {}
        
        for strategy, stats in self.strategy_stats.items():
            if stats['rounds'] > 0:
                win_rate = stats['wins'] / stats['rounds']
                avg_profit = stats['profit'] / stats['rounds']
            else:
                win_rate = 0.0
                avg_profit = 0.0
            
            report[strategy] = {
                'rounds': stats['rounds'],
                'win_rate': win_rate,
                'total_profit': stats['profit'],
                'average_profit_per_round': avg_profit,
                'is_current': strategy == self.current_strategy
            }
        
        return report

