"""
Utilities Module

This module provides utility classes for bankroll management, statistics tracking,
and other supporting functionality for the betting automation framework.
"""

import math
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class RoundResult:
    """Data class for storing individual round results."""
    won: bool
    bet_amount: float
    payout: float
    bankroll_after: float
    timestamp: float = field(default_factory=lambda: 0)

class BankrollManager:
    """Manages bankroll with stop-loss and win target functionality."""
    
    def __init__(self, initial_bankroll: float, stop_loss: float, win_target: float):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.stop_loss = stop_loss
        self.win_target = win_target
        self.peak_bankroll = initial_bankroll
        self.max_drawdown = 0.0
        
    def can_continue_betting(self, bet_amount: float) -> bool:
        """Check if betting can continue based on bankroll and limits."""
        # Check if we have enough funds
        if self.current_bankroll < bet_amount:
            return False
        
        # Check stop-loss
        if self.current_bankroll <= self.stop_loss:
            return False
        
        # Check win target
        if self.current_bankroll >= self.win_target:
            return False
        
        return True
    
    def add_winnings(self, amount: float):
        """Add winnings to the bankroll."""
        self.current_bankroll += amount
        
        # Update peak bankroll
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll
        
        self._update_drawdown()
    
    def subtract_loss(self, amount: float):
        """Subtract losses from the bankroll."""
        self.current_bankroll -= amount
        self._update_drawdown()
    
    def _update_drawdown(self):
        """Update maximum drawdown calculation."""
        current_drawdown = self.peak_bankroll - self.current_bankroll
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_profit_loss(self) -> float:
        """Get current profit or loss."""
        return self.current_bankroll - self.initial_bankroll
    
    def get_roi(self) -> float:
        """Get return on investment as a percentage."""
        if self.initial_bankroll == 0:
            return 0.0
        return (self.get_profit_loss() / self.initial_bankroll) * 100

class StatisticsTracker:
    """Tracks and calculates various performance statistics."""
    
    def __init__(self):
        self.rounds: List[RoundResult] = []
        self.total_wagered = 0.0
        self.total_won = 0.0
        self.total_lost = 0.0
        self.wins = 0
        self.losses = 0
        self.bankroll_history: List[float] = []
    
    def add_round(self, won: bool, bet_amount: float, payout: float, bankroll_after: float):
        """Add a round result to the statistics."""
        round_result = RoundResult(
            won=won,
            bet_amount=bet_amount,
            payout=payout,
            bankroll_after=bankroll_after
        )
        
        self.rounds.append(round_result)
        self.bankroll_history.append(bankroll_after)
        self.total_wagered += bet_amount
        
        if won:
            self.wins += 1
            self.total_won += payout
        else:
            self.losses += 1
            self.total_lost += bet_amount
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate and return comprehensive statistics."""
        total_rounds = len(self.rounds)
        
        if total_rounds == 0:
            return self._empty_stats()
        
        # Basic statistics
        win_rate = (self.wins / total_rounds) * 100
        loss_rate = (self.losses / total_rounds) * 100
        average_bet = self.total_wagered / total_rounds
        total_profit = self.total_won - self.total_lost
        
        # Profit factor (total winnings / total losses)
        profit_factor = self.total_won / self.total_lost if self.total_lost > 0 else float('inf')
        
        # Largest win and loss
        largest_win = max((r.payout for r in self.rounds if r.won), default=0)
        largest_loss = max((r.bet_amount for r in self.rounds if not r.won), default=0)
        
        # Expected value per round
        expected_value = total_profit / total_rounds
        
        # Volatility (standard deviation of returns)
        returns = [r.payout - r.bet_amount for r in self.rounds]
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        
        # Sharpe ratio (risk-adjusted return)
        sharpe_ratio = expected_value / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        return {
            'total_rounds': total_rounds,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'total_wagered': self.total_wagered,
            'total_won': self.total_won,
            'total_lost': self.total_lost,
            'total_profit': total_profit,
            'average_bet': average_bet,
            'profit_factor': profit_factor,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'expected_value': expected_value,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics dictionary."""
        return {
            'total_rounds': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'total_wagered': 0.0,
            'total_won': 0.0,
            'total_lost': 0.0,
            'total_profit': 0.0,
            'average_bet': 0.0,
            'profit_factor': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'expected_value': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from bankroll history."""
        if len(self.bankroll_history) < 2:
            return 0.0
        
        peak = self.bankroll_history[0]
        max_drawdown = 0.0
        
        for bankroll in self.bankroll_history:
            if bankroll > peak:
                peak = bankroll
            
            drawdown = peak - bankroll
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def export_to_csv(self, filename: str):
        """Export round results to CSV file."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['round', 'won', 'bet_amount', 'payout', 'profit_loss', 'bankroll_after']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i, round_result in enumerate(self.rounds, 1):
                writer.writerow({
                    'round': i,
                    'won': round_result.won,
                    'bet_amount': round_result.bet_amount,
                    'payout': round_result.payout,
                    'profit_loss': round_result.payout - round_result.bet_amount,
                    'bankroll_after': round_result.bankroll_after
                })

class RiskCalculator:
    """Utility class for risk-related calculations."""
    
    @staticmethod
    def calculate_kelly_criterion(win_probability: float, win_amount: float, loss_amount: float) -> float:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Args:
            win_probability: Probability of winning (0-1)
            win_amount: Amount won on a winning bet
            loss_amount: Amount lost on a losing bet
            
        Returns:
            Optimal fraction of bankroll to bet
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0.0
        
        if win_amount <= 0 or loss_amount <= 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds received on the wager, p = probability of winning, q = probability of losing
        b = win_amount / loss_amount
        p = win_probability
        q = 1 - win_probability
        
        kelly_fraction = (b * p - q) / b
        
        # Return 0 if Kelly suggests not to bet
        return max(0, kelly_fraction)
    
    @staticmethod
    def calculate_risk_of_ruin(win_probability: float, bankroll: float, bet_size: float, target_loss: float) -> float:
        """
        Calculate the probability of losing a specified amount (risk of ruin).
        
        This is a simplified calculation for demonstration purposes.
        """
        if win_probability <= 0 or win_probability >= 1:
            return 1.0
        
        if bankroll <= target_loss or bet_size <= 0:
            return 1.0
        
        # Simplified risk of ruin calculation
        # In practice, this would use more sophisticated mathematical models
        
        advantage = (2 * win_probability) - 1  # Player advantage
        
        if advantage <= 0:
            return 1.0  # Negative expectation = certain ruin
        
        # Approximate calculation
        ruin_probability = math.exp(-2 * advantage * (bankroll - target_loss) / bet_size)
        
        return min(1.0, max(0.0, ruin_probability))
    
    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) for a given confidence level.
        
        Args:
            returns: List of return values
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR value
        """
        if not returns:
            return 0.0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        
        return abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0

