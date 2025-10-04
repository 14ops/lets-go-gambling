
import random
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ABTestConfig:
    num_simulations: int = 1000  # Number of A/B test simulations
    num_games_per_simulation: int = 1000 # Number of games in each simulation
    initial_bankroll: float = 1000.0
    bet_size: float = 10.0
    # Simplified win/loss probabilities for each strategy (for demonstration)
    # In a real scenario, these would come from actual strategy implementations
    strategy_win_rates: Dict[str, float] = field(default_factory=lambda: {
        "strategy_A": 0.55,  # Control group
        "strategy_B": 0.57   # Test group (slightly better)
    })
    profit_per_win: float = 15.0
    loss_per_loss: float = -10.0
    confidence_level: float = 0.95 # For statistical significance

class ABTestingFramework:
    """
    A framework for conducting A/B tests on different betting strategies.
    
    Like a scientific experiment to determine the superior strategy,
    ensuring data-driven decisions!
    """
    
    def __init__(self, config: ABTestConfig = None):
        self.config = config or ABTestConfig()
        self.results: Dict[str, List[float]] = defaultdict(list) # Stores final profits for each strategy
        self.p_values: List[float] = [] # Stores p-values from each simulation
        
    def _simulate_single_strategy(self, strategy_name: str) -> float:
        """Simulate a single run for a given strategy and return final profit."""
        bankroll = self.config.initial_bankroll
        win_rate = self.config.strategy_win_rates.get(strategy_name, 0.5)
        
        for _ in range(self.config.num_games_per_simulation):
            if random.random() < win_rate:
                bankroll += self.config.profit_per_win
            else:
                bankroll += self.config.loss_per_loss
        return bankroll - self.config.initial_bankroll # Return total profit
            
    def run_ab_test(self):
        """Runs the A/B test simulations."""
        print(f"🧪 Starting A/B Testing Framework with {self.config.num_simulations} simulations!\n")
        
        strategy_A = list(self.config.strategy_win_rates.keys())[0]
        strategy_B = list(self.config.strategy_win_rates.keys())[1]
        
        for sim_num in range(self.config.num_simulations):
            profit_A = self._simulate_single_strategy(strategy_A)
            profit_B = self._simulate_single_strategy(strategy_B)
            
            self.results[strategy_A].append(profit_A)
            self.results[strategy_B].append(profit_B)
            
            # Perform t-test for each simulation (or accumulate data for one large test)
            # For simplicity, we'll do a t-test on the accumulated data at the end
            
            if (sim_num + 1) % (self.config.num_simulations // 10) == 0:
                print(f"Simulation {sim_num + 1}/{self.config.num_simulations} completed.")
                
        print("\n📊 A/B Test Simulations Concluded! Analyzing results...")
        self._analyze_results()
        self._visualize_results()
        
    def _analyze_results(self):
        """Analyzes the accumulated results and performs statistical tests."""
        strategy_A = list(self.config.strategy_win_rates.keys())[0]
        strategy_B = list(self.config.strategy_win_rates.keys())[1]
        
        data_A = np.array(self.results[strategy_A])
        data_B = np.array(self.results[strategy_B])
        
        # Perform independent t-test
        t_stat, p_value = stats.ttest_ind(data_A, data_B, equal_var=False) # Welch's t-test
        self.p_values.append(p_value) # Store for overall analysis
        
        print(f"\n--- A/B Test Results ---")
        print(f"Strategy A ({strategy_A}): Mean Profit = ${np.mean(data_A):.2f}, Std Dev = ${np.std(data_A):.2f}")
        print(f"Strategy B ({strategy_B}): Mean Profit = ${np.mean(data_B):.2f}, Std Dev = ${np.std(data_B):.2f}")
        print(f"T-statistic: {t_stat:.2f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < (1 - self.config.confidence_level):
            print(f"Conclusion: With {self.config.confidence_level:.0%} confidence, there is a statistically significant difference between Strategy A and Strategy B.")
            if np.mean(data_B) > np.mean(data_A):
                print(f"Strategy B ({strategy_B}) is likely better than Strategy A ({strategy_A}).")
            else:
                print(f"Strategy A ({strategy_A}) is likely better than Strategy B ({strategy_B}).")
        else:
            print(f"Conclusion: With {self.config.confidence_level:.0%} confidence, there is NO statistically significant difference between Strategy A and Strategy B.")
            print("Further testing or optimization may be required.")
            
        # Export results to JSON
        export_data = {
            "ab_test_config": self.config.__dict__,
            "strategy_A_results": data_A.tolist(),
            "strategy_B_results": data_B.tolist(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "conclusion": "Statistically significant difference" if p_value < (1 - self.config.confidence_level) else "No statistically significant difference"
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"/home/ubuntu/fusion-project/python-backend/results/ab_test_results_{timestamp}.json"
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\nDetailed A/B test results exported to: {filepath}")
        
    def _visualize_results(self):
        """Visualizes the distribution of profits for each strategy."""
        strategy_A = list(self.config.strategy_win_rates.keys())[0]
        strategy_B = list(self.config.strategy_win_rates.keys())[1]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(self.results[strategy_A], color="blue", label=strategy_A.upper(), kde=True, stat="density", alpha=0.5)
        sns.histplot(self.results[strategy_B], color="red", label=strategy_B.upper(), kde=True, stat="density", alpha=0.5)
        
        plt.title("A/B Test: Distribution of Profits")
        plt.xlabel("Total Profit ($)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"/home/ubuntu/fusion-project/python-backend/visualizations/ab_test_profit_distribution_{timestamp}.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Profit distribution chart saved to: {filepath}")
        
        # Box plot for comparison
        data_to_plot = [self.results[strategy_A], self.results[strategy_B]]
        labels = [strategy_A.upper(), strategy_B.upper()]
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data_to_plot, palette=["blue", "red"])
        plt.xticks(ticks=[0, 1], labels=labels)
        plt.title("A/B Test: Profit Comparison (Box Plot)")
        plt.xlabel("Strategy")
        plt.ylabel("Total Profit ($)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        filepath = f"/home/ubuntu/fusion-project/python-backend/visualizations/ab_test_profit_boxplot_{timestamp}.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Profit box plot saved to: {filepath}")

# Example Usage
if __name__ == "__main__":
    # Ensure the results and visualizations directories exist
    import os
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../visualizations", exist_ok=True)
    
    # Create an A/B test configuration
    ab_test_config = ABTestConfig(
        num_simulations=500, 
        num_games_per_simulation=2000,
        strategy_win_rates={
            "strategy_A": 0.55,  # Control group
            "strategy_B": 0.56   # Test group (slightly better)
        }
    )
    
    # Initialize and run the A/B test
    ab_tester = ABTestingFramework(ab_test_config)
    ab_tester.run_ab_test()
    
    print("\nA/B testing simulation complete. Check the \'results\' and \'visualizations\' folders!")


