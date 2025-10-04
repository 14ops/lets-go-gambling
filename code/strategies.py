"""
Betting Strategies Module

This module implements various betting strategies inspired by anime characters,
each with distinct risk profiles and decision-making approaches.
"""

import random
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

class BaseStrategy(ABC):
    """Abstract base class for all betting strategies."""
    
    def __init__(self, board_size: int, mine_count: int):
        self.board_size = board_size
        self.mine_count = mine_count
        self.total_cells = board_size * board_size
        self.safe_cells = self.total_cells - mine_count
        
    @abstractmethod
    def get_moves(self, board_state: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Generate moves based on the current board state.
        
        Args:
            board_state: Current state of the game board
            
        Returns:
            List of (row, col) tuples representing cells to reveal
        """
        pass
    
    def _get_unrevealed_cells(self, revealed_matrix) -> List[Tuple[int, int]]:
        """Get list of unrevealed cell coordinates."""
        unrevealed = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if not revealed_matrix[i, j]:
                    unrevealed.append((i, j))
        return unrevealed
    
    def _calculate_risk_score(self, cells_revealed: int, safe_cells_remaining: int) -> float:
        """Calculate risk score for current position."""
        if safe_cells_remaining <= 0:
            return 1.0
        
        # Probability of hitting a mine on next reveal
        unrevealed_cells = self.total_cells - cells_revealed
        mines_remaining = self.mine_count
        
        if unrevealed_cells <= 0:
            return 0.0
        
        return mines_remaining / unrevealed_cells

class TakeshiStrategy(BaseStrategy):
    """
    Takeshi (Aggressive) Strategy
    
    Inspired by aggressive anime protagonists who rush into battle.
    - High risk, high reward approach
    - Reveals many cells per round
    - Doesn't back down easily
    """
    
    def get_moves(self, board_state: Dict[str, Any]) -> List[Tuple[int, int]]:
        unrevealed = self._get_unrevealed_cells(board_state['revealed'])
        
        if not unrevealed:
            return []
        
        # Takeshi goes for 40-60% of remaining safe cells
        safe_remaining = board_state['safe_cells_remaining']
        target_reveals = max(1, min(len(unrevealed), int(safe_remaining * random.uniform(0.4, 0.6))))
        
        # Prefer corner and edge cells (more "aggressive" positioning)
        prioritized_cells = []
        regular_cells = []
        
        for row, col in unrevealed:
            is_corner = (row in [0, self.board_size-1]) and (col in [0, self.board_size-1])
            is_edge = (row in [0, self.board_size-1]) or (col in [0, self.board_size-1])
            
            if is_corner:
                prioritized_cells.append((row, col))
            elif is_edge:
                prioritized_cells.append((row, col))
            else:
                regular_cells.append((row, col))
        
        # Select moves with preference for edges/corners
        moves = []
        all_candidates = prioritized_cells + regular_cells
        
        for i in range(min(target_reveals, len(all_candidates))):
            if i < len(prioritized_cells):
                moves.append(prioritized_cells[i])
            else:
                moves.append(regular_cells[i - len(prioritized_cells)])
        
        return moves

class LelouchStrategy(BaseStrategy):
    """
    Lelouch (Calculated) Strategy
    
    Inspired by strategic masterminds who plan every move.
    - Balanced risk-reward approach
    - Uses probability calculations
    - Adapts based on current situation
    """
    
    def get_moves(self, board_state: Dict[str, Any]) -> List[Tuple[int, int]]:
        unrevealed = self._get_unrevealed_cells(board_state['revealed'])
        
        if not unrevealed:
            return []
        
        cells_revealed = board_state['cells_revealed']
        safe_remaining = board_state['safe_cells_remaining']
        
        # Calculate optimal number of reveals based on risk
        risk_score = self._calculate_risk_score(cells_revealed, safe_remaining)
        
        if risk_score < 0.2:  # Low risk
            target_reveals = min(len(unrevealed), max(3, int(safe_remaining * 0.4)))
        elif risk_score < 0.4:  # Medium risk
            target_reveals = min(len(unrevealed), max(2, int(safe_remaining * 0.25)))
        else:  # High risk
            target_reveals = min(len(unrevealed), max(1, int(safe_remaining * 0.15)))
        
        # Use center-out strategy (start from center, work outward)
        center_row, center_col = self.board_size // 2, self.board_size // 2
        
        # Sort cells by distance from center
        cells_with_distance = []
        for row, col in unrevealed:
            distance = abs(row - center_row) + abs(col - center_col)
            cells_with_distance.append((row, col, distance))
        
        cells_with_distance.sort(key=lambda x: x[2])
        
        return [(row, col) for row, col, _ in cells_with_distance[:target_reveals]]

class KazuyaStrategy(BaseStrategy):
    """
    Kazuya (Conservative) Strategy
    
    Inspired by cautious characters who prioritize survival.
    - Low risk approach
    - Reveals few cells per round
    - Focuses on consistent small gains
    """
    
    def get_moves(self, board_state: Dict[str, Any]) -> List[Tuple[int, int]]:
        unrevealed = self._get_unrevealed_cells(board_state['revealed'])
        
        if not unrevealed:
            return []
        
        cells_revealed = board_state['cells_revealed']
        safe_remaining = board_state['safe_cells_remaining']
        risk_score = self._calculate_risk_score(cells_revealed, safe_remaining)
        
        # Very conservative approach
        if risk_score < 0.15:
            target_reveals = min(len(unrevealed), 2)
        elif risk_score < 0.3:
            target_reveals = min(len(unrevealed), 1)
        else:
            # Too risky, cash out early (return empty moves)
            return []
        
        # Prefer cells that are statistically safer (simplified heuristic)
        # In a real implementation, this would use advanced probability analysis
        
        # For now, use random selection from available cells
        selected_cells = random.sample(unrevealed, min(target_reveals, len(unrevealed)))
        
        return selected_cells

class SenkuStrategy(BaseStrategy):
    """
    Senku (Analytical) Strategy
    
    Inspired by scientific/analytical characters who use data and logic.
    - Data-driven approach
    - Uses mathematical optimization
    - Adapts strategy based on statistical analysis
    """
    
    def __init__(self, board_size: int, mine_count: int):
        super().__init__(board_size, mine_count)
        self.round_history = []
        self.performance_metrics = {
            'total_rounds': 0,
            'successful_rounds': 0,
            'average_cells_revealed': 0
        }
    
    def get_moves(self, board_state: Dict[str, Any]) -> List[Tuple[int, int]]:
        unrevealed = self._get_unrevealed_cells(board_state['revealed'])
        
        if not unrevealed:
            return []
        
        cells_revealed = board_state['cells_revealed']
        safe_remaining = board_state['safe_cells_remaining']
        
        # Calculate expected value for different numbers of reveals
        optimal_reveals = self._calculate_optimal_reveals(cells_revealed, safe_remaining, len(unrevealed))
        
        # Use pattern analysis to select best cells
        selected_cells = self._select_optimal_cells(unrevealed, optimal_reveals)
        
        return selected_cells
    
    def _calculate_optimal_reveals(self, cells_revealed: int, safe_remaining: int, unrevealed_count: int) -> int:
        """Calculate optimal number of cells to reveal using expected value analysis."""
        
        if safe_remaining <= 0 or unrevealed_count <= 0:
            return 0
        
        best_ev = -float('inf')
        optimal_count = 1
        
        # Test different numbers of reveals (up to 5 for performance)
        for num_reveals in range(1, min(6, unrevealed_count + 1, safe_remaining + 1)):
            # Calculate probability of success
            prob_success = 1.0
            remaining_cells = unrevealed_count
            remaining_mines = self.mine_count
            
            for i in range(num_reveals):
                if remaining_cells <= 0:
                    break
                safe_prob = max(0, (remaining_cells - remaining_mines) / remaining_cells)
                prob_success *= safe_prob
                remaining_cells -= 1
            
            # Calculate expected multiplier (simplified)
            multiplier = 1.0 + (num_reveals * 0.1)  # Simplified multiplier calculation
            
            # Expected value = probability * (multiplier - 1) - (1 - probability) * 1
            expected_value = prob_success * (multiplier - 1) - (1 - prob_success) * 1
            
            if expected_value > best_ev:
                best_ev = expected_value
                optimal_count = num_reveals
        
        return optimal_count
    
    def _select_optimal_cells(self, unrevealed: List[Tuple[int, int]], num_cells: int) -> List[Tuple[int, int]]:
        """Select optimal cells using pattern analysis."""
        
        if num_cells >= len(unrevealed):
            return unrevealed
        
        # Score each cell based on various factors
        cell_scores = []
        
        for row, col in unrevealed:
            score = 0
            
            # Factor 1: Distance from edges (center cells might be safer)
            edge_distance = min(row, col, self.board_size - 1 - row, self.board_size - 1 - col)
            score += edge_distance * 0.1
            
            # Factor 2: Avoid clustering (spread out selections)
            # This is a simplified heuristic
            score += random.uniform(0, 0.5)
            
            cell_scores.append((row, col, score))
        
        # Sort by score and select top cells
        cell_scores.sort(key=lambda x: x[2], reverse=True)
        
        return [(row, col) for row, col, _ in cell_scores[:num_cells]]

class StrategyFactory:
    """Factory class for creating strategy instances."""
    
    @staticmethod
    def create_strategy(strategy_name: str, board_size: int, mine_count: int) -> BaseStrategy:
        """
        Create a strategy instance based on the strategy name.
        
        Args:
            strategy_name: Name of the strategy to create
            board_size: Size of the game board
            mine_count: Number of mines on the board
            
        Returns:
            Strategy instance
        """
        strategy_map = {
            'takeshi': TakeshiStrategy,
            'lelouch': LelouchStrategy,
            'kazuya': KazuyaStrategy,
            'senku': SenkuStrategy
        }
        
        strategy_class = strategy_map.get(strategy_name.lower(), TakeshiStrategy)
        return strategy_class(board_size, mine_count)
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategy names."""
        return ['takeshi', 'lelouch', 'kazuya', 'senku']
    
    @staticmethod
    def get_strategy_description(strategy_name: str) -> str:
        """Get description of a strategy."""
        descriptions = {
            'takeshi': "Aggressive strategy with high risk, high reward approach. Reveals many cells per round.",
            'lelouch': "Calculated strategy using probability analysis and balanced risk-reward approach.",
            'kazuya': "Conservative strategy focusing on survival and consistent small gains.",
            'senku': "Analytical strategy using mathematical optimization and data-driven decisions."
        }
        
        return descriptions.get(strategy_name.lower(), "Unknown strategy")

