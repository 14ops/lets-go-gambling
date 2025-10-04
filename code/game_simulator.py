"""
Game Simulator Module

This module provides a complete simulation environment for Mines-like games,
allowing for offline testing and statistical analysis of betting strategies.
"""

import random
import numpy as np
from typing import List, Tuple, Set
from dataclasses import dataclass
from enum import Enum

class CellState(Enum):
    HIDDEN = 0
    REVEALED = 1
    MINE = 2

@dataclass
class GameResult:
    """Result of a single game round."""
    won: bool
    multiplier: float
    cells_revealed: int
    total_cells: int
    mines_hit: int

class MinesSimulator:
    """Simulates a Mines-like game for strategy testing."""
    
    def __init__(self, board_size: int, mine_count: int):
        self.board_size = board_size
        self.mine_count = mine_count
        self.total_cells = board_size * board_size
        self.safe_cells = self.total_cells - mine_count
        
        # Validate configuration
        if mine_count >= self.total_cells:
            raise ValueError("Mine count cannot exceed total cells")
        if mine_count < 1:
            raise ValueError("Must have at least one mine")
        
        self.reset_board()
    
    def reset_board(self):
        """Reset the game board for a new round."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.revealed = np.zeros((self.board_size, self.board_size), dtype=bool)
        self.mines_positions = set()
        self.cells_revealed = 0
        self.game_over = False
        self.won = False
        
        # Place mines randomly
        self._place_mines()
    
    def _place_mines(self):
        """Randomly place mines on the board."""
        positions = [(i, j) for i in range(self.board_size) for j in range(self.board_size)]
        mine_positions = random.sample(positions, self.mine_count)
        
        for i, j in mine_positions:
            self.board[i, j] = 1  # 1 represents a mine
            self.mines_positions.add((i, j))
    
    def get_board_state(self) -> dict:
        """Get current board state for strategy decision making."""
        return {
            'board_size': self.board_size,
            'mine_count': self.mine_count,
            'revealed': self.revealed.copy(),
            'cells_revealed': self.cells_revealed,
            'safe_cells_remaining': self.safe_cells - self.cells_revealed,
            'total_safe_cells': self.safe_cells
        }
    
    def reveal_cell(self, row: int, col: int) -> bool:
        """
        Reveal a cell and return True if safe, False if mine.
        
        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            
        Returns:
            True if cell is safe, False if mine
        """
        if self.game_over:
            return False
        
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            raise ValueError("Cell coordinates out of bounds")
        
        if self.revealed[row, col]:
            return True  # Already revealed, assume safe
        
        self.revealed[row, col] = True
        
        if self.board[row, col] == 1:  # Hit a mine
            self.game_over = True
            self.won = False
            return False
        else:  # Safe cell
            self.cells_revealed += 1
            
            # Check win condition (all safe cells revealed)
            if self.cells_revealed >= self.safe_cells:
                self.game_over = True
                self.won = True
            
            return True
    
    def calculate_multiplier(self, cells_revealed: int) -> float:
        """
        Calculate the payout multiplier based on cells revealed.
        
        This uses a simplified multiplier calculation based on probability.
        In real games, this would be determined by the game's payout table.
        """
        if cells_revealed == 0:
            return 1.0
        
        # Calculate probability of revealing N safe cells without hitting a mine
        probability = 1.0
        remaining_cells = self.total_cells
        remaining_mines = self.mine_count
        
        for i in range(cells_revealed):
            safe_probability = (remaining_cells - remaining_mines) / remaining_cells
            probability *= safe_probability
            remaining_cells -= 1
        
        # Multiplier is inverse of probability (with house edge)
        house_edge = 0.02  # 2% house edge
        multiplier = (1.0 / probability) * (1.0 - house_edge)
        
        return max(1.01, multiplier)  # Minimum multiplier of 1.01x
    
    def play_round(self, moves: List[Tuple[int, int]]) -> GameResult:
        """
        Play a complete round with the given moves.
        
        Args:
            moves: List of (row, col) tuples representing cells to reveal
            
        Returns:
            GameResult object with round outcome
        """
        self.reset_board()
        
        cells_revealed = 0
        mines_hit = 0
        
        for row, col in moves:
            if self.game_over:
                break
                
            is_safe = self.reveal_cell(row, col)
            
            if is_safe:
                cells_revealed += 1
            else:
                mines_hit += 1
                break
        
        # Calculate final result
        multiplier = self.calculate_multiplier(cells_revealed) if self.won or cells_revealed > 0 else 1.0
        
        return GameResult(
            won=self.won,
            multiplier=multiplier,
            cells_revealed=cells_revealed,
            total_cells=self.total_cells,
            mines_hit=mines_hit
        )
    
    def get_safe_cell_probability(self, row: int, col: int) -> float:
        """
        Calculate the probability that a specific cell is safe.
        
        This is a simplified calculation for demonstration.
        In practice, this would use more sophisticated probability analysis.
        """
        if self.revealed[row, col]:
            return 1.0 if self.board[row, col] == 0 else 0.0
        
        # Basic probability: (total safe cells - revealed safe cells) / (total unrevealed cells)
        unrevealed_cells = self.total_cells - np.sum(self.revealed)
        unrevealed_safe_cells = self.safe_cells - self.cells_revealed
        
        if unrevealed_cells <= 0:
            return 0.0
        
        return unrevealed_safe_cells / unrevealed_cells
    
    def get_optimal_cells(self, num_cells: int) -> List[Tuple[int, int]]:
        """
        Get the optimal cells to reveal based on current board state.
        
        This is a simplified implementation for demonstration.
        Real implementations would use advanced probability analysis.
        """
        available_cells = []
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if not self.revealed[i, j]:
                    probability = self.get_safe_cell_probability(i, j)
                    available_cells.append((i, j, probability))
        
        # Sort by probability (highest first) and return top N
        available_cells.sort(key=lambda x: x[2], reverse=True)
        
        return [(row, col) for row, col, _ in available_cells[:num_cells]]

