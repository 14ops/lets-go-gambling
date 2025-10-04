import random
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class StrategyPerformance:
    name: str
    win_rate: float
    avg_profit: float
    confidence: float
    weight: float = 0.0

@dataclass
class EnsembleConfig:
    strategies: List[str] = field(default_factory=lambda: [
        "takeshi", "lelouch", "kazuya", "senku", "hybrid", "rintaro_okabe"
    ])
    # Simplified win/loss probabilities for each strategy (for demonstration)
    strategy_win_rates: Dict[str, float] = field(default_factory=lambda: {
        "takeshi": 0.52,  # Aggressive
        "lelouch": 0.64,  # Calculated
        "kazuya": 0.78,   # Conservative
        "senku": 0.72,    # Analytical
        "hybrid": 0.75,   # Balanced
        "rintaro_okabe": 0.85 # Game Theory Master
    })
    # Confidence levels for each strategy (how certain we are about their performance)
    strategy_confidences: Dict[str, float] = field(default_factory=lambda: {
        "takeshi": 0.6,
        "lelouch": 0.8,
        "kazuya": 0.9,
        "senku": 0.85,
        "hybrid": 0.75,
        "rintaro_okabe": 0.95
    })
    profit_per_win: float = 15.0
    loss_per_loss: float = -10.0
    num_games: int = 1000

class ConfidenceWeightedEnsemble:
    """
    Implements a confidence-weighted ensemble that combines multiple strategies
    based on their performance and confidence levels.
    
    Like assembling the ultimate team where each member's contribution
    is weighted by their expertise and reliability!
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        self.ensemble_history: List[Dict[str, Any]] = []
        
        # Initialize strategy performances
        for strategy in self.config.strategies:
            self.strategy_performances[strategy] = StrategyPerformance(
                name=strategy,
                win_rate=self.config.strategy_win_rates.get(strategy, 0.5),
                avg_profit=0.0,
                confidence=self.config.strategy_confidences.get(strategy, 0.5)
            )
    
    def _calculate_weights(self) -> Dict[str, float]:
        """Calculate confidence-weighted ensemble weights."""
        weights = {}
        total_weighted_performance = 0.0
        
        # Calculate weighted performance for each strategy
        for strategy, perf in self.strategy_performances.items():
            # Weight = (win_rate * confidence) normalized
            weighted_performance = perf.win_rate * perf.confidence
            weights[strategy] = weighted_performance
            total_weighted_performance += weighted_performance
        
        # Normalize weights to sum to 1
        if total_weighted_performance > 0:
            for strategy in weights:
                weights[strategy] /= total_weighted_performance
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(weights)
            for strategy in weights:
                weights[strategy] = equal_weight
        
        return weights
    
    def _simulate_strategy_game(self, strategy_name: str) -> Tuple[float, str]:
        """Simulate a single game for a given strategy."""
        win_rate = self.config.strategy_win_rates.get(strategy_name, 0.5)
        
        if random.random() < win_rate:
            return self.config.profit_per_win, "win"
        else:
            return self.config.loss_per_loss, "loss"
    
    def _ensemble_decision(self, weights: Dict[str, float]) -> str:
        """Make an ensemble decision based on weighted voting."""
        # For simplicity, we'll use weighted random selection
        strategies = list(weights.keys())
        strategy_weights = list(weights.values())
        
        return np.random.choice(strategies, p=strategy_weights)
    
    def run_ensemble_simulation(self):
        """Run the confidence-weighted ensemble simulation."""
        print(f"🤖 Starting Confidence-Weighted Ensemble Simulation with {self.config.num_games} games!\n")
        
        total_profit = 0.0
        wins = 0
        losses = 0
        strategy_usage = defaultdict(int)
        
        for game_num in range(self.config.num_games):
            # Calculate current weights
            weights = self._calculate_weights()
            
            # Make ensemble decision
            chosen_strategy = self._ensemble_decision(weights)
            strategy_usage[chosen_strategy] += 1
            
            # Simulate game with chosen strategy
            profit, outcome = self._simulate_strategy_game(chosen_strategy)
            total_profit += profit
            
            if outcome == "win":
                wins += 1
            else:
                losses += 1
            
            # Update strategy performance (simplified online learning)
            perf = self.strategy_performances[chosen_strategy]
            perf.avg_profit = (perf.avg_profit * strategy_usage[chosen_strategy] + profit) / (strategy_usage[chosen_strategy] + 1)
            
            # Store ensemble state for analysis
            if game_num % 100 == 0:  # Store every 100 games
                self.ensemble_history.append({
                    "game": game_num,
                    "total_profit": total_profit,
                    "weights": weights.copy(),
                    "chosen_strategy": chosen_strategy,
                    "win_rate": wins / max(game_num + 1, 1)
                })
            
            if (game_num + 1) % (self.config.num_games // 10) == 0:
                print(f"Game {game_num + 1}/{self.config.num_games} completed. Current profit: ${total_profit:.2f}")
        
        final_win_rate = wins / self.config.num_games
        avg_profit_per_game = total_profit / self.config.num_games
        
        print(f"\n🏆 Ensemble Simulation Complete!")
        print(f"Total Profit: ${total_profit:.2f}")
        print(f"Win Rate: {final_win_rate:.2%}")
        print(f"Average Profit per Game: ${avg_profit_per_game:.2f}")
        print(f"Wins: {wins}, Losses: {losses}")
        
        # Calculate final weights
        final_weights = self._calculate_weights()
        
        print(f"\n--- Final Strategy Weights ---")
        for strategy, weight in final_weights.items():
            usage_percentage = strategy_usage[strategy] / self.config.num_games * 100
            print(f"{strategy.upper()}: Weight = {weight:.3f}, Usage = {usage_percentage:.1f}%")
        
        self._analyze_ensemble_performance()
        self._visualize_ensemble_results(strategy_usage, final_weights)
        
        return {
            "total_profit": total_profit,
            "win_rate": final_win_rate,
            "avg_profit_per_game": avg_profit_per_game,
            "strategy_usage": dict(strategy_usage),
            "final_weights": final_weights
        }
    
    def _analyze_ensemble_performance(self):
        """Analyze the ensemble performance over time."""
        print(f"\n--- Ensemble Performance Analysis ---")
        
        if len(self.ensemble_history) > 1:
            initial_profit = self.ensemble_history[0]["total_profit"]
            final_profit = self.ensemble_history[-1]["total_profit"]
            profit_growth = final_profit - initial_profit
            
            initial_win_rate = self.ensemble_history[0]["win_rate"]
            final_win_rate = self.ensemble_history[-1]["win_rate"]
            
            print(f"Profit Growth: ${profit_growth:.2f}")
            print(f"Win Rate Change: {initial_win_rate:.2%} → {final_win_rate:.2%}")
            
            # Analyze weight evolution
            weight_changes = {}
            for strategy in self.config.strategies:
                initial_weight = self.ensemble_history[0]["weights"].get(strategy, 0)
                final_weight = self.ensemble_history[-1]["weights"].get(strategy, 0)
                weight_changes[strategy] = final_weight - initial_weight
            
            print(f"\n--- Weight Evolution ---")
            for strategy, change in weight_changes.items():
                direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"{strategy.upper()}: {direction} {change:+.3f}")
        
        # Export detailed results
        export_data = {
            "ensemble_config": self.config.__dict__,
            "strategy_performances": {k: v.__dict__ for k, v in self.strategy_performances.items()},
            "ensemble_history": self.ensemble_history,
            "final_analysis": {
                "total_strategies": len(self.config.strategies),
                "simulation_games": self.config.num_games,
                "timestamp": time.time()
            }
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"/home/ubuntu/fusion-project/python-backend/results/ensemble_results_{timestamp}.json"
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\nDetailed ensemble results exported to: {filepath}")
    
    def _visualize_ensemble_results(self, strategy_usage: Dict[str, int], final_weights: Dict[str, float]):
        """Visualize ensemble results and weight evolution."""
        
        # 1. Strategy Usage Pie Chart
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 2, 1)
        strategies = list(strategy_usage.keys())
        usage_counts = list(strategy_usage.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        
        plt.pie(usage_counts, labels=[s.upper() for s in strategies], autopct='%1.1f%%', colors=colors)
        plt.title('Strategy Usage Distribution')
        
        # 2. Final Weights Bar Chart
        plt.subplot(2, 2, 2)
        strategies = list(final_weights.keys())
        weights = list(final_weights.values())
        
        bars = plt.bar(strategies, weights, color=colors[:len(strategies)])
        plt.title('Final Strategy Weights')
        plt.xlabel('Strategy')
        plt.ylabel('Weight')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{weight:.3f}', ha='center', va='bottom')
        
        # 3. Profit Evolution
        plt.subplot(2, 2, 3)
        games = [entry["game"] for entry in self.ensemble_history]
        profits = [entry["total_profit"] for entry in self.ensemble_history]
        
        plt.plot(games, profits, marker='o', linewidth=2, markersize=4)
        plt.title('Profit Evolution')
        plt.xlabel('Game Number')
        plt.ylabel('Total Profit ($)')
        plt.grid(True, alpha=0.3)
        
        # 4. Win Rate Evolution
        plt.subplot(2, 2, 4)
        win_rates = [entry["win_rate"] for entry in self.ensemble_history]
        
        plt.plot(games, win_rates, marker='s', linewidth=2, markersize=4, color='green')
        plt.title('Win Rate Evolution')
        plt.xlabel('Game Number')
        plt.ylabel('Win Rate')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"/home/ubuntu/fusion-project/python-backend/visualizations/ensemble_analysis_{timestamp}.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Ensemble analysis chart saved to: {filepath}")
        
        # Weight Evolution Heatmap
        if len(self.ensemble_history) > 1:
            plt.figure(figsize=(12, 6))
            
            # Prepare data for heatmap
            strategies = self.config.strategies
            time_points = [entry["game"] for entry in self.ensemble_history]
            weight_matrix = []
            
            for entry in self.ensemble_history:
                weights_at_time = [entry["weights"].get(strategy, 0) for strategy in strategies]
                weight_matrix.append(weights_at_time)
            
            weight_matrix = np.array(weight_matrix).T  # Transpose for proper orientation
            
            sns.heatmap(weight_matrix, 
                       xticklabels=[f"Game {g}" for g in time_points[::2]], # Show every other game
                       yticklabels=[s.upper() for s in strategies],
                       cmap="YlOrRd", 
                       annot=False, 
                       cbar_kws={'label': 'Weight'})
            
            plt.title('Strategy Weight Evolution Over Time')
            plt.xlabel('Game Progress')
            plt.ylabel('Strategy')
            plt.xticks(rotation=45, ha='right')
            
            filepath = f"/home/ubuntu/fusion-project/python-backend/visualizations/ensemble_weight_evolution_{timestamp}.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Weight evolution heatmap saved to: {filepath}")

# Example Usage
if __name__ == "__main__":
    # Ensure the results and visualizations directories exist
    import os
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../visualizations", exist_ok=True)
    
    # Create an ensemble configuration
    ensemble_config = EnsembleConfig(
        num_games=5000,
        strategies=["takeshi", "lelouch", "kazuya", "senku", "hybrid", "rintaro_okabe"]
    )
    
    # Initialize and run the ensemble
    ensemble = ConfidenceWeightedEnsemble(ensemble_config)
    results = ensemble.run_ensemble_simulation()
    
    print("\nConfidence-weighted ensemble simulation complete. Check the 'results' and 'visualizations' folders!")

