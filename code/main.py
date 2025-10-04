#!/usr/bin/env python3
"""
Applied Probability and Automation Framework for High-RTP Games
Python Backend - Main Entry Point

This module serves as the main entry point for the Python backend,
handling configuration, strategy execution, and simulation.
"""

import json
import time
import random
import sys
import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game_simulator import MinesSimulator
from strategies import StrategyFactory
from utils import BankrollManager, StatisticsTracker

class GameMode(Enum):
    SIMULATION = "simulation"
    LIVE = "live"

@dataclass
class GameConfig:
    """Configuration for the game session."""
    board_size: int
    mine_count: int
    bet_amount: float
    strategy: str
    mode: str
    initial_bankroll: float = 1000.0
    max_rounds: int = 1000
    stop_loss: float = 100.0
    win_target: float = 2000.0

class GameController:
    """Main controller for the game automation and simulation."""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.bankroll_manager = BankrollManager(config.initial_bankroll, config.stop_loss, config.win_target)
        self.stats_tracker = StatisticsTracker()
        self.simulator = MinesSimulator(config.board_size, config.mine_count)
        self.strategy = StrategyFactory.create_strategy(config.strategy, config.board_size, config.mine_count)
        self.running = True
        
    def run(self):
        """Main execution loop."""
        print(f"Starting {self.config.mode} mode...")
        print(f"Strategy: {self.config.strategy}")
        print(f"Board: {self.config.board_size}x{self.config.board_size}, Mines: {self.config.mine_count}")
        print(f"Bet Amount: ${self.config.bet_amount}")
        print(f"Initial Bankroll: ${self.config.initial_bankroll}")
        print("-" * 50)
        
        if self.config.mode == GameMode.SIMULATION.value:
            self._run_simulation()
        else:
            self._run_live()
    
    def _run_simulation(self):
        """Run the simulation mode."""
        round_count = 0
        
        while (round_count < self.config.max_rounds and 
               self.bankroll_manager.can_continue_betting(self.config.bet_amount) and 
               self.running):
            
            round_count += 1
            
            # Execute strategy
            moves = self.strategy.get_moves(self.simulator.get_board_state())
            
            # Simulate the round
            result = self.simulator.play_round(moves)
            
            # Update bankroll
            if result.won:
                payout = self.config.bet_amount * result.multiplier
                self.bankroll_manager.add_winnings(payout - self.config.bet_amount)
            else:
                self.bankroll_manager.subtract_loss(self.config.bet_amount)
            
            # Update statistics
            self.stats_tracker.add_round(result.won, self.config.bet_amount, 
                                       payout if result.won else 0, 
                                       self.bankroll_manager.current_bankroll)
            
            # Print progress every 100 rounds
            if round_count % 100 == 0:
                self._print_status(round_count)
            
            # Add small delay to prevent overwhelming output
            time.sleep(0.01)
        
        # Final statistics
        self._print_final_results(round_count)
    
    def _run_live(self):
        """Run the live automation mode (placeholder)."""
        print("Live mode is not implemented in this demo version.")
        print("This would integrate with actual game interfaces using automation libraries.")
        print("For safety and demonstration purposes, only simulation mode is available.")
    
    def _print_status(self, round_count: int):
        """Print current status."""
        stats = self.stats_tracker.get_statistics()
        print(f"Round {round_count:4d} | "
              f"Bankroll: ${self.bankroll_manager.current_bankroll:8.2f} | "
              f"Win Rate: {stats['win_rate']:5.1f}% | "
              f"Profit: ${stats['total_profit']:8.2f}")
    
    def _print_final_results(self, total_rounds: int):
        """Print final simulation results."""
        stats = self.stats_tracker.get_statistics()
        
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"Total Rounds: {total_rounds}")
        print(f"Final Bankroll: ${self.bankroll_manager.current_bankroll:.2f}")
        print(f"Total Profit/Loss: ${stats['total_profit']:.2f}")
        print(f"Win Rate: {stats['win_rate']:.2f}%")
        print(f"Average Bet: ${stats['average_bet']:.2f}")
        print(f"Largest Win: ${stats['largest_win']:.2f}")
        print(f"Largest Loss: ${stats['largest_loss']:.2f}")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        
        # Risk metrics
        print(f"\nRisk Metrics:")
        print(f"Max Drawdown: ${stats['max_drawdown']:.2f}")
        print(f"Volatility: {stats['volatility']:.2f}")
        
        # Strategy performance
        print(f"\nStrategy Performance ({self.config.strategy}):")
        print(f"Expected Value per Round: ${stats['expected_value']:.4f}")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.4f}")

def load_config() -> GameConfig:
    """Load configuration from JSON file."""
    try:
        with open('config.json', 'r') as f:
            config_data = json.load(f)
        
        # Map strategy names
        strategy_mapping = {
            "Takeshi (Aggressive)": "takeshi",
            "Lelouch (Calculated)": "lelouch", 
            "Kazuya (Conservative)": "kazuya",
            "Senku (Analytical)": "senku"
        }
        
        strategy_name = strategy_mapping.get(config_data['strategy'], 'takeshi')
        
        return GameConfig(
            board_size=int(config_data['board_size']),
            mine_count=int(config_data['mine_count']),
            bet_amount=float(config_data['bet_amount']),
            strategy=strategy_name,
            mode=config_data.get('mode', 'simulation')
        )
    
    except FileNotFoundError:
        print("Configuration file not found. Using default configuration.")
        return GameConfig(
            board_size=5,
            mine_count=3,
            bet_amount=1.0,
            strategy='takeshi',
            mode='simulation'
        )
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration.")
        return GameConfig(
            board_size=5,
            mine_count=3,
            bet_amount=1.0,
            strategy='takeshi',
            mode='simulation'
        )

def main():
    """Main entry point."""
    try:
        # Load configuration
        config = load_config()
        
        # Create and run game controller
        controller = GameController(config)
        controller.run()
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

