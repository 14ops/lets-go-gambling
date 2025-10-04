"""
Visualization Module

This module provides comprehensive data visualization capabilities for the
betting automation framework, including performance charts, strategy comparisons,
and risk analysis graphs.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import seaborn as sns
from datetime import datetime, timedelta

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PerformanceVisualizer:
    """Creates various performance visualization charts."""
    
    def __init__(self, output_dir: str = "../visualizations"):
        self.output_dir = output_dir
        self.colors = {
            'takeshi': '#FF6B6B',    # Red - Aggressive
            'lelouch': '#4ECDC4',    # Teal - Calculated  
            'kazuya': '#45B7D1',     # Blue - Conservative
            'senku': '#96CEB4'       # Green - Analytical
        }
        
    def create_bankroll_chart(self, bankroll_history: List[float], strategy_name: str, 
                            save_path: str = None) -> str:
        """Create a bankroll progression chart."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        rounds = list(range(1, len(bankroll_history) + 1))
        
        # Main bankroll line
        ax.plot(rounds, bankroll_history, linewidth=2.5, 
               color=self.colors.get(strategy_name.lower(), '#333333'),
               label=f'{strategy_name.title()} Strategy')
        
        # Add initial bankroll reference line
        initial_bankroll = bankroll_history[0] if bankroll_history else 1000
        ax.axhline(y=initial_bankroll, color='gray', linestyle='--', alpha=0.7, 
                  label='Initial Bankroll')
        
        # Fill area for profit/loss
        ax.fill_between(rounds, bankroll_history, initial_bankroll, 
                       where=[b >= initial_bankroll for b in bankroll_history],
                       color='green', alpha=0.3, label='Profit Zone')
        ax.fill_between(rounds, bankroll_history, initial_bankroll,
                       where=[b < initial_bankroll for b in bankroll_history], 
                       color='red', alpha=0.3, label='Loss Zone')
        
        # Styling
        ax.set_xlabel('Round Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Bankroll ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'Bankroll Progression - {strategy_name.title()} Strategy', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key points
        if bankroll_history:
            max_bankroll = max(bankroll_history)
            min_bankroll = min(bankroll_history)
            final_bankroll = bankroll_history[-1]
            
            # Annotate peak
            max_idx = bankroll_history.index(max_bankroll)
            ax.annotate(f'Peak: ${max_bankroll:.2f}', 
                       xy=(max_idx + 1, max_bankroll),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Annotate lowest point
            min_idx = bankroll_history.index(min_bankroll)
            ax.annotate(f'Lowest: ${min_bankroll:.2f}',
                       xy=(min_idx + 1, min_bankroll),
                       xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/bankroll_progression_{strategy_name.lower()}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_strategy_comparison(self, strategy_stats: Dict[str, Dict], 
                                 save_path: str = None) -> str:
        """Create a comprehensive strategy comparison chart."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        strategies = list(strategy_stats.keys())
        
        # 1. Win Rate Comparison
        win_rates = [strategy_stats[s]['win_rate'] * 100 for s in strategies]
        colors = [self.colors.get(s.lower(), '#333333') for s in strategies]
        
        bars1 = ax1.bar(strategies, win_rates, color=colors, alpha=0.8)
        ax1.set_ylabel('Win Rate (%)', fontweight='bold')
        ax1.set_title('Win Rate by Strategy', fontweight='bold', fontsize=14)
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, win_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Total Profit Comparison
        profits = [strategy_stats[s]['total_profit'] for s in strategies]
        bars2 = ax2.bar(strategies, profits, color=colors, alpha=0.8)
        ax2.set_ylabel('Total Profit ($)', fontweight='bold')
        ax2.set_title('Total Profit by Strategy', fontweight='bold', fontsize=14)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, profit in zip(bars2, profits):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (max(profits) * 0.02),
                    f'${profit:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Average Profit per Round
        avg_profits = [strategy_stats[s]['average_profit_per_round'] for s in strategies]
        bars3 = ax3.bar(strategies, avg_profits, color=colors, alpha=0.8)
        ax3.set_ylabel('Avg Profit per Round ($)', fontweight='bold')
        ax3.set_title('Average Profit per Round', fontweight='bold', fontsize=14)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 4. Number of Rounds
        rounds = [strategy_stats[s]['rounds'] for s in strategies]
        bars4 = ax4.bar(strategies, rounds, color=colors, alpha=0.8)
        ax4.set_ylabel('Number of Rounds', fontweight='bold')
        ax4.set_title('Rounds Played by Strategy', fontweight='bold', fontsize=14)
        
        # Overall styling
        for ax in [ax1, ax2, ax3, ax4]:
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Strategy Performance Comparison', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/strategy_comparison.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_risk_analysis_chart(self, statistics: Dict[str, Any], 
                                 save_path: str = None) -> str:
        """Create a risk analysis visualization."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Profit/Loss Distribution (simulated for demo)
        np.random.seed(42)
        returns = np.random.normal(statistics.get('expected_value', 0), 
                                 statistics.get('volatility', 1), 1000)
        
        ax1.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.axvline(x=statistics.get('expected_value', 0), color='green', 
                   linestyle='-', linewidth=2, label='Expected Value')
        ax1.set_xlabel('Return per Round ($)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Return Distribution', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Risk Metrics Radar Chart
        metrics = ['Win Rate', 'Profit Factor', 'Sharpe Ratio', 'Volatility (inv)', 'Max Drawdown (inv)']
        values = [
            min(statistics.get('win_rate', 0) * 100, 100),
            min(statistics.get('profit_factor', 0) * 20, 100),
            min(abs(statistics.get('sharpe_ratio', 0)) * 50, 100),
            max(0, 100 - statistics.get('volatility', 0) * 10),
            max(0, 100 - statistics.get('max_drawdown', 0) / 10)
        ]
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax2.plot(angles, values, 'o-', linewidth=2, color='#FF6B6B')
        ax2.fill(angles, values, alpha=0.25, color='#FF6B6B')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics, fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.set_title('Risk Profile', fontweight='bold', fontsize=14)
        ax2.grid(True)
        
        # 3. Drawdown Analysis
        # Simulate drawdown data
        bankroll_sim = [1000]
        for i in range(100):
            change = np.random.normal(statistics.get('expected_value', -1), 
                                    statistics.get('volatility', 5))
            bankroll_sim.append(max(0, bankroll_sim[-1] + change))
        
        peak = np.maximum.accumulate(bankroll_sim)
        drawdown = [(peak[i] - bankroll_sim[i]) / peak[i] * 100 for i in range(len(bankroll_sim))]
        
        ax3.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color='red')
        ax3.set_xlabel('Time Period', fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontweight='bold')
        ax3.set_title('Drawdown Analysis', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # 4. Key Statistics Table
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = [
            ['Metric', 'Value'],
            ['Total Rounds', f"{statistics.get('total_rounds', 0):,}"],
            ['Win Rate', f"{statistics.get('win_rate', 0)*100:.2f}%"],
            ['Total Profit', f"${statistics.get('total_profit', 0):.2f}"],
            ['Max Drawdown', f"${statistics.get('max_drawdown', 0):.2f}"],
            ['Profit Factor', f"{statistics.get('profit_factor', 0):.2f}"],
            ['Sharpe Ratio', f"{statistics.get('sharpe_ratio', 0):.4f}"],
            ['Volatility', f"{statistics.get('volatility', 0):.2f}"]
        ]
        
        table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data)):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4ECDC4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
        
        ax4.set_title('Performance Summary', fontweight='bold', fontsize=14)
        
        plt.suptitle('Risk Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/risk_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

class CharacterCardGenerator:
    """Generates character cards for each strategy."""
    
    def __init__(self, output_dir: str = "../visualizations"):
        self.output_dir = output_dir
        self.character_info = {
            'takeshi': {
                'title': 'The Relentless Aggressor',
                'description': 'Maximizes short-term gains with high-risk plays',
                'color': '#FF6B6B',
                'traits': ['High Risk', 'High Reward', 'Aggressive', 'Bold']
            },
            'lelouch': {
                'title': 'The Strategic Mastermind', 
                'description': 'Uses calculated probability analysis',
                'color': '#4ECDC4',
                'traits': ['Calculated', 'Balanced', 'Strategic', 'Adaptive']
            },
            'kazuya': {
                'title': 'The Cautious Survivor',
                'description': 'Prioritizes capital preservation',
                'color': '#45B7D1', 
                'traits': ['Conservative', 'Safe', 'Steady', 'Defensive']
            },
            'senku': {
                'title': 'The Data Scientist',
                'description': 'Employs mathematical optimization',
                'color': '#96CEB4',
                'traits': ['Analytical', 'Scientific', 'Logical', 'Precise']
            }
        }
    
    def create_character_card(self, strategy_name: str, stats: Dict[str, Any], 
                            save_path: str = None) -> str:
        """Create a character card for a strategy."""
        
        strategy_name = strategy_name.lower()
        if strategy_name not in self.character_info:
            return None
        
        info = self.character_info[strategy_name]
        
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Background
        bg_rect = patches.Rectangle((0.5, 0.5), 9, 11, linewidth=3, 
                                  edgecolor=info['color'], facecolor='white')
        ax.add_patch(bg_rect)
        
        # Header section
        header_rect = patches.Rectangle((0.5, 9.5), 9, 2, linewidth=0,
                                      facecolor=info['color'], alpha=0.8)
        ax.add_patch(header_rect)
        
        # Character name
        ax.text(5, 10.7, strategy_name.upper(), fontsize=24, fontweight='bold',
               ha='center', va='center', color='white')
        
        # Title
        ax.text(5, 10.2, info['title'], fontsize=14, fontweight='bold',
               ha='center', va='center', color='white')
        
        # Description
        ax.text(5, 8.8, info['description'], fontsize=12, ha='center', va='center',
               wrap=True, style='italic')
        
        # Stats section
        ax.text(5, 8.2, 'PERFORMANCE STATS', fontsize=16, fontweight='bold',
               ha='center', va='center', color=info['color'])
        
        # Performance metrics
        y_pos = 7.5
        metrics = [
            ('Win Rate', f"{stats.get('win_rate', 0)*100:.1f}%"),
            ('Total Profit', f"${stats.get('total_profit', 0):.2f}"),
            ('Rounds Played', f"{stats.get('rounds', 0):,}"),
            ('Avg Profit/Round', f"${stats.get('average_profit_per_round', 0):.3f}")
        ]
        
        for metric, value in metrics:
            ax.text(2, y_pos, metric + ':', fontsize=12, fontweight='bold', ha='left')
            ax.text(8, y_pos, value, fontsize=12, ha='right', 
                   color=info['color'], fontweight='bold')
            y_pos -= 0.5
        
        # Traits section
        ax.text(5, 5.2, 'STRATEGY TRAITS', fontsize=16, fontweight='bold',
               ha='center', va='center', color=info['color'])
        
        # Trait badges
        trait_y = 4.5
        for i, trait in enumerate(info['traits']):
            x_pos = 2.5 + (i % 2) * 5
            if i >= 2:
                trait_y = 3.8
            
            # Trait badge
            badge_rect = patches.Rectangle((x_pos-1, trait_y-0.2), 2, 0.4,
                                         linewidth=1, edgecolor=info['color'],
                                         facecolor=info['color'], alpha=0.2)
            ax.add_patch(badge_rect)
            
            ax.text(x_pos, trait_y, trait, fontsize=10, fontweight='bold',
                   ha='center', va='center', color=info['color'])
        
        # Rating section
        ax.text(5, 2.8, 'RISK RATING', fontsize=16, fontweight='bold',
               ha='center', va='center', color=info['color'])
        
        # Risk level visualization
        risk_levels = {'takeshi': 5, 'lelouch': 3, 'kazuya': 1, 'senku': 2}
        risk_level = risk_levels.get(strategy_name, 3)
        
        for i in range(5):
            star_color = info['color'] if i < risk_level else 'lightgray'
            ax.text(3 + i * 0.8, 2.2, '★', fontsize=20, ha='center', va='center',
                   color=star_color)
        
        # Footer
        ax.text(5, 1.2, f'Generated on {datetime.now().strftime("%Y-%m-%d")}',
               fontsize=10, ha='center', va='center', style='italic', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/character_card_{strategy_name}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path

def create_all_visualizations(strategy_stats: Dict[str, Dict], 
                            bankroll_histories: Dict[str, List[float]],
                            overall_stats: Dict[str, Any],
                            output_dir: str = "../visualizations") -> List[str]:
    """Create all visualization files and return list of file paths."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = PerformanceVisualizer(output_dir)
    card_generator = CharacterCardGenerator(output_dir)
    
    created_files = []
    
    # Create bankroll charts for each strategy
    for strategy, history in bankroll_histories.items():
        if history:
            file_path = visualizer.create_bankroll_chart(history, strategy)
            created_files.append(file_path)
    
    # Create strategy comparison
    if strategy_stats:
        file_path = visualizer.create_strategy_comparison(strategy_stats)
        created_files.append(file_path)
    
    # Create risk analysis
    if overall_stats:
        file_path = visualizer.create_risk_analysis_chart(overall_stats)
        created_files.append(file_path)
    
    # Create character cards
    for strategy, stats in strategy_stats.items():
        file_path = card_generator.create_character_card(strategy, stats)
        if file_path:
            created_files.append(file_path)
    
    return created_files

