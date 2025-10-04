"""
Comprehensive Visualization Module

This module creates the maximum variety of visualizations possible based on all
available data and concepts from the betting strategy framework.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import seaborn as sns
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from scipy import stats
from scipy.stats import norm, beta, gamma
import warnings
warnings.filterwarnings('ignore')

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveVisualizer:
    """Creates comprehensive visualization suite."""
    
    def __init__(self, output_dir: str = "../visualizations"):
        self.output_dir = output_dir
        self.colors = {
            'takeshi': '#FF6B6B',    # Red - Aggressive
            'lelouch': '#4ECDC4',    # Teal - Calculated  
            'kazuya': '#45B7D1',     # Blue - Conservative
            'senku': '#96CEB4'       # Green - Analytical
        }
        self.strategies = ['Takeshi', 'Lelouch', 'Kazuya', 'Senku']
        
    def create_probability_theory_visualization(self, save_path: str = None) -> str:
        """Visualize the mathematical foundations and probability theory."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Hypergeometric Distribution for Mines Game
        ax = axes[0, 0]
        total_cells = 25
        mines = 3
        safe_cells = total_cells - mines
        
        reveals = np.arange(1, 11)
        probabilities = []
        
        for r in reveals:
            # Probability of revealing r safe cells without hitting a mine
            from math import comb
            prob = comb(safe_cells, r) / comb(total_cells, r)
            probabilities.append(prob)
        
        ax.bar(reveals, probabilities, color='skyblue', alpha=0.7, edgecolor='black')
        ax.set_title('Hypergeometric Distribution\n(5x5 board, 3 mines)', fontweight='bold')
        ax.set_xlabel('Number of Cells Revealed')
        ax.set_ylabel('Probability of Success')
        ax.grid(True, alpha=0.3)
        
        # Add probability values on bars
        for i, (r, p) in enumerate(zip(reveals, probabilities)):
            ax.text(r, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Expected Value vs Risk Trade-off
        ax = axes[0, 1]
        risk_levels = np.linspace(0.1, 0.8, 50)
        
        # Different payout structures
        conservative_ev = []
        moderate_ev = []
        aggressive_ev = []
        
        for risk in risk_levels:
            # Conservative: Lower multiplier, higher win rate
            win_rate_cons = 1 - risk * 0.8
            multiplier_cons = 1 + risk * 0.5
            ev_cons = win_rate_cons * (multiplier_cons - 1) - (1 - win_rate_cons)
            conservative_ev.append(ev_cons)
            
            # Moderate: Balanced
            win_rate_mod = 1 - risk
            multiplier_mod = 1 + risk * 1.2
            ev_mod = win_rate_mod * (multiplier_mod - 1) - (1 - win_rate_mod)
            moderate_ev.append(ev_mod)
            
            # Aggressive: Higher multiplier, lower win rate
            win_rate_agg = 1 - risk * 1.3
            multiplier_agg = 1 + risk * 2.0
            ev_agg = win_rate_agg * (multiplier_agg - 1) - (1 - win_rate_agg)
            aggressive_ev.append(ev_agg)
        
        ax.plot(risk_levels, conservative_ev, label='Conservative', linewidth=2, color='blue')
        ax.plot(risk_levels, moderate_ev, label='Moderate', linewidth=2, color='green')
        ax.plot(risk_levels, aggressive_ev, label='Aggressive', linewidth=2, color='red')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_title('Expected Value vs Risk Trade-off', fontweight='bold')
        ax.set_xlabel('Risk Level')
        ax.set_ylabel('Expected Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Kelly Criterion Visualization
        ax = axes[0, 2]
        win_probs = np.linspace(0.4, 0.8, 100)
        payout_odds = 1.5  # 1.5:1 payout
        
        kelly_fractions = []
        for p in win_probs:
            # Kelly formula: f = (bp - q) / b where b = odds, p = win prob, q = lose prob
            b = payout_odds
            q = 1 - p
            kelly_f = (b * p - q) / b
            kelly_fractions.append(max(0, kelly_f))  # Don't bet if negative
        
        ax.plot(win_probs, kelly_fractions, linewidth=3, color='purple')
        ax.fill_between(win_probs, kelly_fractions, alpha=0.3, color='purple')
        ax.set_title('Kelly Criterion Optimal Bet Size\n(1.5:1 payout odds)', fontweight='bold')
        ax.set_xlabel('Win Probability')
        ax.set_ylabel('Optimal Fraction of Bankroll')
        ax.grid(True, alpha=0.3)
        
        # 4. House Edge Impact Over Time
        ax = axes[1, 0]
        rounds = np.arange(1, 1001)
        house_edges = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10%
        
        for edge in house_edges:
            # Expected bankroll after n rounds with house edge
            expected_bankroll = 1000 * (1 - edge) ** rounds
            ax.plot(rounds, expected_bankroll, label=f'{edge*100:.0f}% House Edge', linewidth=2)
        
        ax.set_title('Impact of House Edge Over Time', fontweight='bold')
        ax.set_xlabel('Number of Rounds')
        ax.set_ylabel('Expected Bankroll ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 5. Variance and Standard Deviation
        ax = axes[1, 1]
        bet_amounts = np.linspace(0.1, 5.0, 50)
        
        # Different strategies have different variance profiles
        takeshi_variance = bet_amounts ** 2 * 0.8  # High variance
        lelouch_variance = bet_amounts ** 2 * 0.3  # Moderate variance
        kazuya_variance = bet_amounts ** 2 * 0.1   # Low variance
        senku_variance = bet_amounts ** 2 * 0.25   # Optimized variance
        
        ax.plot(bet_amounts, np.sqrt(takeshi_variance), label='Takeshi', color=self.colors['takeshi'], linewidth=2)
        ax.plot(bet_amounts, np.sqrt(lelouch_variance), label='Lelouch', color=self.colors['lelouch'], linewidth=2)
        ax.plot(bet_amounts, np.sqrt(kazuya_variance), label='Kazuya', color=self.colors['kazuya'], linewidth=2)
        ax.plot(bet_amounts, np.sqrt(senku_variance), label='Senku', color=self.colors['senku'], linewidth=2)
        
        ax.set_title('Strategy Variance vs Bet Size', fontweight='bold')
        ax.set_xlabel('Bet Amount ($)')
        ax.set_ylabel('Standard Deviation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Probability Density Functions
        ax = axes[1, 2]
        x = np.linspace(-0.1, 0.1, 1000)
        
        # Different return distributions for each strategy
        takeshi_returns = norm.pdf(x, -0.02, 0.08)
        lelouch_returns = norm.pdf(x, 0.015, 0.03)
        kazuya_returns = norm.pdf(x, -0.005, 0.01)
        senku_returns = norm.pdf(x, 0.025, 0.025)
        
        ax.plot(x, takeshi_returns, label='Takeshi', color=self.colors['takeshi'], linewidth=2)
        ax.plot(x, lelouch_returns, label='Lelouch', color=self.colors['lelouch'], linewidth=2)
        ax.plot(x, kazuya_returns, label='Kazuya', color=self.colors['kazuya'], linewidth=2)
        ax.plot(x, senku_returns, label='Senku', color=self.colors['senku'], linewidth=2)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_title('Return Distribution Probability Density', fontweight='bold')
        ax.set_xlabel('Return per Round')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Mathematical Foundations and Probability Theory', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/probability_theory_visualization.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_game_mechanics_analysis(self, save_path: str = None) -> str:
        """Visualize game mechanics and board analysis."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Board Size Impact Analysis
        ax = axes[0, 0]
        board_sizes = [3, 4, 5, 6, 7, 8]
        mine_densities = [0.1, 0.15, 0.2, 0.25]
        
        # Create heatmap of difficulty
        difficulty_matrix = np.zeros((len(board_sizes), len(mine_densities)))
        
        for i, size in enumerate(board_sizes):
            for j, density in enumerate(mine_densities):
                total_cells = size * size
                mines = int(total_cells * density)
                # Difficulty increases with mine density and decreases with board size
                difficulty = density * 100 + (1 / size) * 20
                difficulty_matrix[i, j] = difficulty
        
        im = ax.imshow(difficulty_matrix, cmap='Reds', aspect='auto')
        ax.set_xticks(range(len(mine_densities)))
        ax.set_yticks(range(len(board_sizes)))
        ax.set_xticklabels([f'{d*100:.0f}%' for d in mine_densities])
        ax.set_yticklabels([f'{s}x{s}' for s in board_sizes])
        ax.set_title('Game Difficulty Matrix', fontweight='bold')
        ax.set_xlabel('Mine Density')
        ax.set_ylabel('Board Size')
        
        # Add text annotations
        for i in range(len(board_sizes)):
            for j in range(len(mine_densities)):
                text = ax.text(j, i, f'{difficulty_matrix[i, j]:.1f}',
                              ha="center", va="center", color="white", fontweight='bold')
        
        # 2. Optimal Cell Revelation Strategy
        ax = axes[0, 1]
        cells_to_reveal = np.arange(1, 11)
        
        # Different board configurations
        configs = [
            {'size': 5, 'mines': 3, 'label': '5x5, 3 mines'},
            {'size': 4, 'mines': 2, 'label': '4x4, 2 mines'},
            {'size': 6, 'mines': 4, 'label': '6x6, 4 mines'}
        ]
        
        for config in configs:
            expected_values = []
            for cells in cells_to_reveal:
                total_cells = config['size'] ** 2
                safe_cells = total_cells - config['mines']
                
                if cells <= safe_cells:
                    from math import comb
                    prob_success = comb(safe_cells, cells) / comb(total_cells, cells)
                    # Assume multiplier increases with cells revealed
                    multiplier = 1 + cells * 0.2
                    ev = prob_success * (multiplier - 1) - (1 - prob_success)
                    expected_values.append(ev)
                else:
                    expected_values.append(-1)  # Impossible
            
            ax.plot(cells_to_reveal, expected_values, 'o-', label=config['label'], linewidth=2, markersize=6)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Expected Value by Cells Revealed', fontweight='bold')
        ax.set_xlabel('Number of Cells to Reveal')
        ax.set_ylabel('Expected Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Position Strategy Heatmap
        ax = axes[0, 2]
        
        # Create a 5x5 board showing optimal cell selection order
        board = np.zeros((5, 5))
        
        # Corner cells (highest priority)
        corners = [(0,0), (0,4), (4,0), (4,4)]
        for corner in corners:
            board[corner] = 4
        
        # Edge cells (medium priority)
        edges = [(0,1), (0,2), (0,3), (1,0), (1,4), (2,0), (2,4), (3,0), (3,4), (4,1), (4,2), (4,3)]
        for edge in edges:
            board[edge] = 3
        
        # Inner edge cells
        inner_edges = [(1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,3)]
        for inner in inner_edges:
            board[inner] = 2
        
        # Center cell (lowest priority)
        board[2,2] = 1
        
        im = ax.imshow(board, cmap='RdYlGn', aspect='equal')
        ax.set_title('Optimal Cell Selection Priority\n(5x5 Board)', fontweight='bold')
        
        # Add grid and labels
        for i in range(5):
            for j in range(5):
                priority = int(board[i, j])
                priority_text = ['Center', 'Inner', 'Edge', 'Corner'][priority-1] if priority > 0 else ''
                ax.text(j, i, priority_text, ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(range(1, 6))
        ax.set_yticklabels(range(1, 6))
        
        # 4. Multiplier vs Probability Relationship
        ax = axes[1, 0]
        probabilities = np.linspace(0.1, 0.9, 50)
        
        # Different payout structures
        fair_multipliers = 1 / probabilities
        house_edge_1 = fair_multipliers * 0.99  # 1% house edge
        house_edge_5 = fair_multipliers * 0.95  # 5% house edge
        
        ax.plot(probabilities, fair_multipliers, label='Fair (0% house edge)', linewidth=2, color='green')
        ax.plot(probabilities, house_edge_1, label='1% house edge', linewidth=2, color='orange')
        ax.plot(probabilities, house_edge_5, label='5% house edge', linewidth=2, color='red')
        
        ax.set_title('Multiplier vs Win Probability', fontweight='bold')
        ax.set_xlabel('Win Probability')
        ax.set_ylabel('Payout Multiplier')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1, 10)
        
        # 5. Risk-Reward Efficiency Frontier
        ax = axes[1, 1]
        
        # Generate efficient frontier data
        risks = np.linspace(0.05, 0.5, 50)
        max_returns = []
        
        for risk in risks:
            # Maximum theoretical return for given risk level
            max_return = risk * 0.8 - 0.02  # Simplified model
            max_returns.append(max_return)
        
        ax.plot(risks, max_returns, linewidth=3, color='purple', label='Efficient Frontier')
        ax.fill_between(risks, max_returns, alpha=0.3, color='purple')
        
        # Plot strategy positions
        strategy_data = {
            'Takeshi': {'risk': 0.35, 'return': -0.02},
            'Lelouch': {'risk': 0.18, 'return': 0.015},
            'Kazuya': {'risk': 0.08, 'return': -0.005},
            'Senku': {'risk': 0.15, 'return': 0.025}
        }
        
        for strategy, data in strategy_data.items():
            ax.scatter(data['risk'], data['return'], s=100, 
                      color=self.colors[strategy.lower()], label=strategy, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Risk-Reward Efficiency Frontier', fontweight='bold')
        ax.set_xlabel('Risk (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Bankroll Requirements Analysis
        ax = axes[1, 2]
        
        # Different strategies require different bankroll sizes
        confidence_levels = [90, 95, 99]
        strategies = ['Takeshi', 'Lelouch', 'Kazuya', 'Senku']
        
        # Estimated bankroll requirements (as multiple of bet size)
        bankroll_requirements = {
            'Takeshi': [50, 80, 150],
            'Lelouch': [30, 45, 70],
            'Kazuya': [15, 20, 30],
            'Senku': [25, 35, 55]
        }
        
        x = np.arange(len(confidence_levels))
        width = 0.2
        
        for i, strategy in enumerate(strategies):
            ax.bar(x + i * width, bankroll_requirements[strategy], width, 
                  label=strategy, color=self.colors[strategy.lower()], alpha=0.8)
        
        ax.set_title('Recommended Bankroll Requirements\n(Multiple of Bet Size)', fontweight='bold')
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('Bankroll Multiple')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f'{c}%' for c in confidence_levels])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Game Mechanics and Strategic Analysis', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/game_mechanics_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_detection_evasion_analysis(self, save_path: str = None) -> str:
        """Visualize detection evasion strategies and patterns."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Human vs Bot Behavior Patterns
        ax = axes[0, 0]
        
        # Simulate reaction times
        np.random.seed(42)
        human_times = np.random.gamma(2, 0.5, 1000) + 0.5  # Human reaction times
        bot_times = np.random.normal(0.1, 0.02, 1000)      # Bot reaction times
        evasive_bot_times = np.random.gamma(1.5, 0.3, 1000) + 0.3  # Evasive bot
        
        ax.hist(human_times, bins=50, alpha=0.6, label='Human', color='blue', density=True)
        ax.hist(bot_times, bins=50, alpha=0.6, label='Basic Bot', color='red', density=True)
        ax.hist(evasive_bot_times, bins=50, alpha=0.6, label='Evasive Bot', color='green', density=True)
        
        ax.set_title('Reaction Time Distributions', fontweight='bold')
        ax.set_xlabel('Reaction Time (seconds)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Mouse Movement Patterns
        ax = axes[0, 1]
        
        # Generate mouse movement data
        t = np.linspace(0, 1, 100)
        
        # Human movement (curved, imperfect)
        human_x = t + 0.1 * np.sin(5 * t) + np.random.normal(0, 0.02, 100)
        human_y = 0.5 + 0.2 * np.cos(3 * t) + np.random.normal(0, 0.02, 100)
        
        # Bot movement (straight line)
        bot_x = t
        bot_y = np.full_like(t, 0.5)
        
        # Evasive bot movement (slightly curved)
        evasive_x = t + 0.05 * np.sin(2 * t)
        evasive_y = 0.5 + 0.1 * np.cos(1.5 * t)
        
        ax.plot(human_x, human_y, label='Human', color='blue', linewidth=2, alpha=0.8)
        ax.plot(bot_x, bot_y, label='Basic Bot', color='red', linewidth=2, alpha=0.8)
        ax.plot(evasive_x, evasive_y, label='Evasive Bot', color='green', linewidth=2, alpha=0.8)
        
        ax.set_title('Mouse Movement Patterns', fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Session Duration Patterns
        ax = axes[0, 2]
        
        # Different session duration patterns
        hours = np.arange(0, 24)
        human_activity = 10 * np.exp(-((hours - 14)**2) / 50) + 2  # Peak afternoon
        bot_activity = np.full_like(hours, 8.0)  # Constant
        evasive_activity = 8 + 4 * np.sin((hours - 6) * np.pi / 12) * (hours > 6) * (hours < 22)
        
        ax.plot(hours, human_activity, 'o-', label='Human', color='blue', linewidth=2)
        ax.plot(hours, bot_activity, 's-', label='Basic Bot', color='red', linewidth=2)
        ax.plot(hours, evasive_activity, '^-', label='Evasive Bot', color='green', linewidth=2)
        
        ax.set_title('Activity Patterns by Hour', fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Activity Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Betting Pattern Regularity
        ax = axes[1, 0]
        
        # Generate betting pattern data
        rounds = np.arange(1, 101)
        
        # Human: irregular patterns
        human_bets = 1 + 0.5 * np.random.random(100) + 0.3 * np.sin(rounds / 10)
        
        # Bot: very regular
        bot_bets = np.full_like(rounds, 1.0, dtype=float)
        
        # Evasive bot: slightly irregular
        evasive_bets = 1 + 0.1 * np.random.random(100) + 0.05 * np.sin(rounds / 15)
        
        ax.plot(rounds, human_bets, label='Human', color='blue', alpha=0.7)
        ax.plot(rounds, bot_bets, label='Basic Bot', color='red', alpha=0.7)
        ax.plot(rounds, evasive_bets, label='Evasive Bot', color='green', alpha=0.7)
        
        ax.set_title('Betting Amount Patterns', fontweight='bold')
        ax.set_xlabel('Round Number')
        ax.set_ylabel('Bet Amount ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Detection Risk Score Over Time
        ax = axes[1, 1]
        
        # Simulate detection risk accumulation
        time_points = np.arange(0, 1000)
        
        # Basic bot: risk increases linearly
        basic_risk = np.minimum(time_points * 0.001, 1.0)
        
        # Evasive bot: risk increases more slowly with periodic resets
        evasive_risk = []
        current_risk = 0
        for t in time_points:
            current_risk += 0.0003
            # Periodic "cooling off" periods
            if t % 200 == 0 and t > 0:
                current_risk *= 0.5
            evasive_risk.append(min(current_risk, 1.0))
        
        ax.plot(time_points, basic_risk, label='Basic Bot', color='red', linewidth=2)
        ax.plot(time_points, evasive_risk, label='Evasive Bot', color='green', linewidth=2)
        ax.axhline(y=0.8, color='orange', linestyle='--', label='High Risk Threshold')
        ax.axhline(y=0.95, color='red', linestyle='--', label='Ban Threshold')
        
        ax.set_title('Detection Risk Accumulation', fontweight='bold')
        ax.set_xlabel('Time (rounds)')
        ax.set_ylabel('Detection Risk Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Evasion Technique Effectiveness
        ax = axes[1, 2]
        
        techniques = ['Random\nDelays', 'Mouse\nDrift', 'Session\nBreaks', 'Bet\nVariation', 'Combined\nApproach']
        effectiveness = [0.3, 0.4, 0.6, 0.5, 0.85]
        implementation_difficulty = [0.2, 0.3, 0.4, 0.3, 0.8]
        
        # Create scatter plot
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        sizes = [eff * 500 for eff in effectiveness]
        
        scatter = ax.scatter(implementation_difficulty, effectiveness, s=sizes, 
                           c=colors, alpha=0.7, edgecolors='black')
        
        # Add labels
        for i, technique in enumerate(techniques):
            ax.annotate(technique, (implementation_difficulty[i], effectiveness[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_title('Evasion Technique Analysis', fontweight='bold')
        ax.set_xlabel('Implementation Difficulty')
        ax.set_ylabel('Effectiveness')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.suptitle('Detection Evasion Strategies and Analysis', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/detection_evasion_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_portfolio_optimization_analysis(self, save_path: str = None) -> str:
        """Visualize portfolio optimization and multi-strategy approaches."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Efficient Frontier with Strategy Combinations
        ax = axes[0, 0]
        
        # Generate efficient frontier
        n_portfolios = 1000
        np.random.seed(42)
        
        # Strategy returns and risks
        strategy_returns = np.array([-0.02, 0.015, -0.005, 0.025])  # Takeshi, Lelouch, Kazuya, Senku
        strategy_risks = np.array([0.35, 0.18, 0.08, 0.15])
        
        portfolio_returns = []
        portfolio_risks = []
        
        for _ in range(n_portfolios):
            # Random weights that sum to 1
            weights = np.random.random(4)
            weights /= weights.sum()
            
            # Portfolio return and risk
            port_return = np.sum(weights * strategy_returns)
            port_risk = np.sqrt(np.sum((weights * strategy_risks) ** 2))  # Simplified
            
            portfolio_returns.append(port_return)
            portfolio_risks.append(port_risk)
        
        # Plot efficient frontier
        ax.scatter(portfolio_risks, portfolio_returns, alpha=0.3, s=10, color='lightblue')
        
        # Plot individual strategies
        for i, strategy in enumerate(self.strategies):
            ax.scatter(strategy_risks[i], strategy_returns[i], s=100, 
                      color=self.colors[strategy.lower()], label=strategy, alpha=0.8)
        
        # Highlight optimal portfolio
        optimal_idx = np.argmax(np.array(portfolio_returns) / np.array(portfolio_risks))
        ax.scatter(portfolio_risks[optimal_idx], portfolio_returns[optimal_idx], 
                  s=200, color='gold', marker='*', label='Optimal Portfolio', edgecolor='black')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Portfolio Efficient Frontier', fontweight='bold')
        ax.set_xlabel('Portfolio Risk')
        ax.set_ylabel('Portfolio Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Correlation Matrix
        ax = axes[0, 1]
        
        # Simulated correlation matrix between strategies
        correlation_matrix = np.array([
            [1.00, 0.15, -0.20, 0.25],  # Takeshi
            [0.15, 1.00, 0.30, 0.60],   # Lelouch
            [-0.20, 0.30, 1.00, 0.40],  # Kazuya
            [0.25, 0.60, 0.40, 1.00]    # Senku
        ])
        
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(self.strategies)
        ax.set_yticklabels(self.strategies)
        ax.set_title('Strategy Correlation Matrix', fontweight='bold')
        
        # Add correlation values
        for i in range(4):
            for j in range(4):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                              ha="center", va="center", color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
                              fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 3. Optimal Allocation Pie Chart
        ax = axes[0, 2]
        
        # Optimal portfolio weights (example)
        optimal_weights = [0.10, 0.35, 0.25, 0.30]  # Takeshi, Lelouch, Kazuya, Senku
        colors = [self.colors[s.lower()] for s in self.strategies]
        
        wedges, texts, autotexts = ax.pie(optimal_weights, labels=self.strategies, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title('Optimal Portfolio Allocation', fontweight='bold')
        
        # 4. Risk Contribution Analysis
        ax = axes[1, 0]
        
        # Risk contribution of each strategy to portfolio
        individual_risks = [0.35, 0.18, 0.08, 0.15]
        weights = optimal_weights
        
        # Marginal risk contribution
        marginal_contributions = []
        for i in range(4):
            # Simplified marginal risk calculation
            marginal_risk = individual_risks[i] * weights[i]
            marginal_contributions.append(marginal_risk)
        
        bars = ax.bar(self.strategies, marginal_contributions, 
                     color=[self.colors[s.lower()] for s in self.strategies], alpha=0.8)
        ax.set_title('Risk Contribution by Strategy', fontweight='bold')
        ax.set_ylabel('Risk Contribution')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, contrib in zip(bars, marginal_contributions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{contrib:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Rebalancing Frequency Analysis
        ax = axes[1, 1]
        
        rebalancing_periods = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Never']
        transaction_costs = [0.05, 0.02, 0.008, 0.003, 0.0]
        portfolio_drift = [0.001, 0.005, 0.02, 0.08, 0.15]
        net_benefit = [tc - drift for tc, drift in zip(transaction_costs, portfolio_drift)]
        
        x = np.arange(len(rebalancing_periods))
        
        bars1 = ax.bar(x - 0.2, transaction_costs, 0.4, label='Transaction Costs', color='red', alpha=0.7)
        bars2 = ax.bar(x + 0.2, portfolio_drift, 0.4, label='Portfolio Drift Cost', color='orange', alpha=0.7)
        
        ax2 = ax.twinx()
        line = ax2.plot(x, net_benefit, 'go-', linewidth=3, markersize=8, label='Net Benefit')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_title('Rebalancing Frequency Analysis', fontweight='bold')
        ax.set_xlabel('Rebalancing Frequency')
        ax.set_ylabel('Cost (%)', color='red')
        ax2.set_ylabel('Net Benefit (%)', color='green')
        ax.set_xticks(x)
        ax.set_xticklabels(rebalancing_periods, rotation=45)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax.grid(True, alpha=0.3)
        
        # 6. Dynamic Allocation Over Market Conditions
        ax = axes[1, 2]
        
        market_conditions = ['Bull\nMarket', 'Bear\nMarket', 'High\nVolatility', 'Low\nVolatility', 'Crisis']
        
        # Allocation changes for different conditions
        takeshi_alloc = [0.20, 0.05, 0.02, 0.15, 0.01]
        lelouch_alloc = [0.30, 0.40, 0.35, 0.35, 0.30]
        kazuya_alloc = [0.20, 0.35, 0.45, 0.25, 0.60]
        senku_alloc = [0.30, 0.20, 0.18, 0.25, 0.09]
        
        x = np.arange(len(market_conditions))
        width = 0.6
        
        p1 = ax.bar(x, takeshi_alloc, width, label='Takeshi', color=self.colors['takeshi'])
        p2 = ax.bar(x, lelouch_alloc, width, bottom=takeshi_alloc, label='Lelouch', color=self.colors['lelouch'])
        p3 = ax.bar(x, kazuya_alloc, width, bottom=np.array(takeshi_alloc) + np.array(lelouch_alloc), 
                   label='Kazuya', color=self.colors['kazuya'])
        p4 = ax.bar(x, senku_alloc, width, 
                   bottom=np.array(takeshi_alloc) + np.array(lelouch_alloc) + np.array(kazuya_alloc),
                   label='Senku', color=self.colors['senku'])
        
        ax.set_title('Dynamic Allocation by Market Condition', fontweight='bold')
        ax.set_xlabel('Market Condition')
        ax.set_ylabel('Allocation (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(market_conditions)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.suptitle('Portfolio Optimization and Multi-Strategy Analysis', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/portfolio_optimization_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_machine_learning_analysis(self, save_path: str = None) -> str:
        """Visualize machine learning aspects and adaptive algorithms."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Learning Curves for Different Algorithms
        ax = axes[0, 0]
        
        epochs = np.arange(1, 101)
        
        # Different learning algorithms
        np.random.seed(42)
        
        # Q-Learning
        q_learning = 0.3 + 0.4 * (1 - np.exp(-epochs/30)) + np.random.normal(0, 0.02, 100)
        
        # Neural Network
        nn_learning = 0.2 + 0.5 * (1 - np.exp(-epochs/20)) + np.random.normal(0, 0.03, 100)
        
        # Genetic Algorithm
        ga_learning = 0.25 + 0.45 * (1 - np.exp(-epochs/40)) + np.random.normal(0, 0.025, 100)
        
        # Random Forest
        rf_learning = 0.35 + 0.35 * (1 - np.exp(-epochs/15)) + np.random.normal(0, 0.015, 100)
        
        ax.plot(epochs, q_learning, label='Q-Learning', linewidth=2, color='red')
        ax.plot(epochs, nn_learning, label='Neural Network', linewidth=2, color='blue')
        ax.plot(epochs, ga_learning, label='Genetic Algorithm', linewidth=2, color='green')
        ax.plot(epochs, rf_learning, label='Random Forest', linewidth=2, color='purple')
        
        ax.set_title('Learning Algorithm Performance', fontweight='bold')
        ax.set_xlabel('Training Epochs')
        ax.set_ylabel('Win Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Feature Importance Analysis
        ax = axes[0, 1]
        
        features = ['Board\nPosition', 'Historical\nWin Rate', 'Recent\nPerformance', 
                   'Risk\nScore', 'Volatility', 'Time of\nDay', 'Bankroll\nLevel', 'Streak\nLength']
        importance_scores = [0.18, 0.16, 0.14, 0.12, 0.11, 0.09, 0.08, 0.07]
        
        bars = ax.barh(features, importance_scores, color='skyblue', alpha=0.8)
        ax.set_title('Feature Importance in ML Model', fontweight='bold')
        ax.set_xlabel('Importance Score')
        
        # Add value labels
        for bar, score in zip(bars, importance_scores):
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        # 3. Model Performance Comparison
        ax = axes[0, 2]
        
        models = ['Linear\nRegression', 'Random\nForest', 'SVM', 'Neural\nNetwork', 'Ensemble']
        accuracy = [0.62, 0.71, 0.68, 0.74, 0.78]
        precision = [0.58, 0.69, 0.65, 0.72, 0.76]
        recall = [0.60, 0.68, 0.66, 0.71, 0.75]
        
        x = np.arange(len(models))
        width = 0.25
        
        bars1 = ax.bar(x - width, accuracy, width, label='Accuracy', alpha=0.8, color='red')
        bars2 = ax.bar(x, precision, width, label='Precision', alpha=0.8, color='green')
        bars3 = ax.bar(x + width, recall, width, label='Recall', alpha=0.8, color='blue')
        
        ax.set_title('Model Performance Comparison', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Hyperparameter Optimization
        ax = axes[1, 0]
        
        # 3D surface plot for hyperparameter optimization
        from mpl_toolkits.mplot3d import Axes3D
        ax.remove()
        ax = fig.add_subplot(2, 3, 4, projection='3d')
        
        # Create meshgrid for hyperparameters
        learning_rate = np.linspace(0.001, 0.1, 20)
        batch_size = np.linspace(16, 128, 20)
        LR, BS = np.meshgrid(learning_rate, batch_size)
        
        # Simulate performance surface
        performance = 0.7 + 0.1 * np.sin(LR * 50) * np.cos(BS / 20) - 0.05 * (LR - 0.01)**2 - 0.02 * (BS - 64)**2 / 1000
        
        surf = ax.plot_surface(LR, BS, performance, cmap='viridis', alpha=0.8)
        ax.set_title('Hyperparameter Optimization', fontweight='bold')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Batch Size')
        ax.set_zlabel('Performance')
        
        # 5. Overfitting Analysis
        ax = axes[1, 1]
        
        epochs = np.arange(1, 101)
        
        # Training and validation curves
        train_loss = 0.8 * np.exp(-epochs/20) + 0.1 + np.random.normal(0, 0.02, 100)
        val_loss = 0.8 * np.exp(-epochs/25) + 0.15 + 0.1 * np.maximum(0, epochs - 50) / 50 + np.random.normal(0, 0.03, 100)
        
        ax.plot(epochs, train_loss, label='Training Loss', linewidth=2, color='blue')
        ax.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='red')
        
        # Mark overfitting point
        overfitting_point = 50
        ax.axvline(x=overfitting_point, color='orange', linestyle='--', linewidth=2, label='Overfitting Starts')
        
        ax.set_title('Training vs Validation Loss', fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Ensemble Method Performance
        ax = axes[1, 2]
        
        # Different ensemble methods
        methods = ['Voting', 'Bagging', 'Boosting', 'Stacking']
        individual_performance = [0.68, 0.70, 0.69, 0.71]
        ensemble_performance = [0.73, 0.75, 0.77, 0.79]
        improvement = [ep - ip for ep, ip in zip(ensemble_performance, individual_performance)]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, individual_performance, width, label='Individual Models', alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x + width/2, ensemble_performance, width, label='Ensemble', alpha=0.8, color='lightgreen')
        
        # Add improvement annotations
        for i, imp in enumerate(improvement):
            ax.annotate(f'+{imp:.2f}', xy=(i, ensemble_performance[i]), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom', fontweight='bold', color='green')
        
        ax.set_title('Ensemble Method Performance', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Machine Learning and Adaptive Algorithm Analysis', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/machine_learning_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

def create_all_comprehensive_visualizations(output_dir: str = "../visualizations") -> List[str]:
    """Create all comprehensive visualization files and return list of file paths."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = ComprehensiveVisualizer(output_dir)
    created_files = []
    
    # Create probability theory visualization
    file_path = visualizer.create_probability_theory_visualization()
    created_files.append(file_path)
    
    # Create game mechanics analysis
    file_path = visualizer.create_game_mechanics_analysis()
    created_files.append(file_path)
    
    # Create detection evasion analysis
    file_path = visualizer.create_detection_evasion_analysis()
    created_files.append(file_path)
    
    # Create portfolio optimization analysis
    file_path = visualizer.create_portfolio_optimization_analysis()
    created_files.append(file_path)
    
    # Create machine learning analysis
    file_path = visualizer.create_machine_learning_analysis()
    created_files.append(file_path)
    
    return created_files

