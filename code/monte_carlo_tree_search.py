"""
Monte Carlo Tree Search (MCTS) Implementation

This module implements Monte Carlo Tree Search for board-based Mines logic where decisions
branch per click. Like having a strategic AI that can see all possible futures and
choose the path with the highest win probability!
"""

import numpy as np
import random
import math
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import copy
from enum import Enum
import json

class ActionType(Enum):
    """Types of actions available in the game."""
    REVEAL_CELL = "reveal_cell"
    CASH_OUT = "cash_out"
    CONTINUE = "continue"

@dataclass
class GameAction:
    """Represents an action in the game."""
    action_type: ActionType
    position: Optional[Tuple[int, int]] = None
    confidence: float = 1.0
    expected_value: float = 0.0
    
    def __hash__(self):
        return hash((self.action_type, self.position))
    
    def __eq__(self, other):
        return (self.action_type == other.action_type and 
                self.position == other.position)

@dataclass
class MCTSGameState:
    """Represents the current state of the game for MCTS."""
    board_size: Tuple[int, int]
    mines_count: int
    revealed_cells: Set[Tuple[int, int]]
    mine_positions: Set[Tuple[int, int]]  # Known mine positions
    safe_cells: Set[Tuple[int, int]]
    current_multiplier: float
    bankroll: float
    bet_amount: float
    is_terminal: bool = False
    is_win: bool = False
    
    def __post_init__(self):
        # Ensure sets are properly initialized
        if not isinstance(self.revealed_cells, set):
            self.revealed_cells = set(self.revealed_cells)
        if not isinstance(self.mine_positions, set):
            self.mine_positions = set(self.mine_positions)
        if not isinstance(self.safe_cells, set):
            self.safe_cells = set(self.safe_cells)
    
    def copy(self) -> 'MCTSGameState':
        """Create a deep copy of the game state."""
        return MCTSGameState(
            board_size=self.board_size,
            mines_count=self.mines_count,
            revealed_cells=self.revealed_cells.copy(),
            mine_positions=self.mine_positions.copy(),
            safe_cells=self.safe_cells.copy(),
            current_multiplier=self.current_multiplier,
            bankroll=self.bankroll,
            bet_amount=self.bet_amount,
            is_terminal=self.is_terminal,
            is_win=self.is_win
        )
    
    def get_available_actions(self) -> List[GameAction]:
        """Get all available actions from this state."""
        if self.is_terminal:
            return []
        
        actions = []
        
        # Always can cash out (unless already terminal)
        actions.append(GameAction(
            action_type=ActionType.CASH_OUT,
            expected_value=self.current_multiplier * self.bet_amount
        ))
        
        # Can reveal unrevealed cells
        total_cells = self.board_size[0] * self.board_size[1]
        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                position = (row, col)
                if position not in self.revealed_cells:
                    # Calculate expected value for revealing this cell
                    ev = self._calculate_reveal_expected_value(position)
                    actions.append(GameAction(
                        action_type=ActionType.REVEAL_CELL,
                        position=position,
                        expected_value=ev
                    ))
        
        return actions
    
    def _calculate_reveal_expected_value(self, position: Tuple[int, int]) -> float:
        """Calculate expected value of revealing a specific cell."""
        # Calculate probability that this cell is safe
        total_cells = self.board_size[0] * self.board_size[1]
        unrevealed_cells = total_cells - len(self.revealed_cells)
        remaining_mines = self.mines_count - len(self.mine_positions)
        
        if unrevealed_cells <= 0:
            return 0.0
        
        safe_probability = max(0.0, (unrevealed_cells - remaining_mines) / unrevealed_cells)
        
        # Expected multiplier if safe (simplified)
        next_multiplier = self.current_multiplier * 1.1  # Approximate multiplier increase
        
        # Expected value calculation
        ev_safe = safe_probability * next_multiplier * self.bet_amount
        ev_mine = (1 - safe_probability) * (-self.bet_amount)
        
        return ev_safe + ev_mine
    
    def apply_action(self, action: GameAction) -> 'MCTSGameState':
        """Apply an action and return the resulting state."""
        new_state = self.copy()
        
        if action.action_type == ActionType.CASH_OUT:
            new_state.is_terminal = True
            new_state.is_win = True
            new_state.bankroll += self.current_multiplier * self.bet_amount
            
        elif action.action_type == ActionType.REVEAL_CELL:
            position = action.position
            new_state.revealed_cells.add(position)
            
            # Check if it's a mine (simplified simulation)
            if self._is_mine(position):
                new_state.mine_positions.add(position)
                new_state.is_terminal = True
                new_state.is_win = False
                new_state.bankroll -= self.bet_amount
            else:
                new_state.safe_cells.add(position)
                new_state.current_multiplier *= 1.1  # Simplified multiplier increase
                
                # Check if all safe cells are revealed
                total_cells = self.board_size[0] * self.board_size[1]
                if len(new_state.safe_cells) >= total_cells - self.mines_count:
                    new_state.is_terminal = True
                    new_state.is_win = True
                    new_state.bankroll += new_state.current_multiplier * self.bet_amount
        
        return new_state
    
    def _is_mine(self, position: Tuple[int, int]) -> bool:
        """Determine if a position contains a mine (probabilistic simulation)."""
        total_cells = self.board_size[0] * self.board_size[1]
        unrevealed_cells = total_cells - len(self.revealed_cells)
        remaining_mines = self.mines_count - len(self.mine_positions)
        
        if unrevealed_cells <= 0 or remaining_mines <= 0:
            return False
        
        mine_probability = remaining_mines / unrevealed_cells
        return random.random() < mine_probability
    
    def get_reward(self) -> float:
        """Get the reward for reaching this state."""
        if not self.is_terminal:
            return 0.0
        
        if self.is_win:
            return self.current_multiplier * self.bet_amount
        else:
            return -self.bet_amount

class MCTSNode:
    """Node in the Monte Carlo Tree Search tree."""
    
    def __init__(self, state: MCTSGameState, parent: Optional['MCTSNode'] = None, 
                 action: Optional[GameAction] = None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        
        # MCTS statistics
        self.visits = 0
        self.total_reward = 0.0
        self.children: Dict[GameAction, 'MCTSNode'] = {}
        self.untried_actions = state.get_available_actions()
        
        # UCB1 parameters
        self.exploration_constant = math.sqrt(2)
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return self.state.is_terminal
    
    def ucb1_value(self) -> float:
        """Calculate UCB1 value for action selection."""
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0:
            return self.total_reward / self.visits
        
        exploitation = self.total_reward / self.visits
        exploration = self.exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        
        return exploitation + exploration
    
    def select_child(self) -> 'MCTSNode':
        """Select child with highest UCB1 value."""
        return max(self.children.values(), key=lambda child: child.ucb1_value())
    
    def expand(self) -> 'MCTSNode':
        """Expand the tree by adding a new child node."""
        if not self.untried_actions:
            return self
        
        action = self.untried_actions.pop()
        new_state = self.state.apply_action(action)
        child_node = MCTSNode(new_state, parent=self, action=action)
        self.children[action] = child_node
        
        return child_node
    
    def simulate(self) -> float:
        """Simulate a random playout from this state."""
        current_state = self.state.copy()
        total_reward = 0.0
        simulation_depth = 0
        max_depth = 50  # Prevent infinite simulations
        
        while not current_state.is_terminal and simulation_depth < max_depth:
            available_actions = current_state.get_available_actions()
            if not available_actions:
                break
            
            # Random action selection with some bias towards higher EV actions
            action = self._select_simulation_action(available_actions)
            current_state = current_state.apply_action(action)
            simulation_depth += 1
        
        return current_state.get_reward()
    
    def _select_simulation_action(self, actions: List[GameAction]) -> GameAction:
        """Select action for simulation with some intelligence."""
        if not actions:
            return actions[0]
        
        # Bias towards actions with higher expected value
        weights = []
        for action in actions:
            if action.action_type == ActionType.CASH_OUT:
                # Cash out becomes more attractive as multiplier increases
                weight = max(1.0, self.state.current_multiplier / 2.0)
            else:
                # Reveal actions weighted by expected value
                weight = max(0.1, action.expected_value + 1.0)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            return np.random.choice(actions, p=weights)
        else:
            return random.choice(actions)
    
    def backpropagate(self, reward: float):
        """Backpropagate the reward up the tree."""
        self.visits += 1
        self.total_reward += reward
        
        if self.parent:
            self.parent.backpropagate(reward)
    
    def best_action(self) -> Optional[GameAction]:
        """Get the best action based on visit count."""
        if not self.children:
            return None
        
        return max(self.children.keys(), 
                  key=lambda action: self.children[action].visits)
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics for all child actions."""
        stats = {}
        for action, child in self.children.items():
            action_key = f"{action.action_type.value}"
            if action.position:
                action_key += f"_{action.position}"
            
            stats[action_key] = {
                'visits': child.visits,
                'avg_reward': child.total_reward / max(1, child.visits),
                'ucb1_value': child.ucb1_value(),
                'expected_value': action.expected_value
            }
        
        return stats

class MonteCarloTreeSearch:
    """
    Monte Carlo Tree Search Implementation for Mines Game
    
    Uses MCTS to find optimal strategies by building a search tree of possible
    game states and actions. Like having a crystal ball that shows all possible
    futures and their probabilities!
    
    Features:
    - UCB1 selection for exploration vs exploitation
    - Intelligent simulation with biased random playouts
    - Depth-limited search to prevent infinite loops
    - Action statistics and confidence measures
    - Integration with existing strategy framework
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # MCTS parameters
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.max_time = self.config.get('max_time', 5.0)  # seconds
        self.exploration_constant = self.config.get('exploration_constant', math.sqrt(2))
        self.simulation_depth = self.config.get('simulation_depth', 50)
        
        # Performance tracking
        self.search_statistics = defaultdict(list)
        self.total_searches = 0
        self.total_search_time = 0.0
        
        # Tree reuse
        self.reuse_tree = self.config.get('reuse_tree', True)
        self.previous_root = None
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for MCTS."""
        return {
            'max_iterations': 1000,
            'max_time': 5.0,
            'exploration_constant': math.sqrt(2),
            'simulation_depth': 50,
            'reuse_tree': True,
            'min_visits_for_selection': 10,
            'confidence_threshold': 0.8,
            'early_termination': True,
            'progressive_widening': False
        }
    
    def search(self, initial_state: MCTSGameState) -> Dict[str, Any]:
        """
        Perform MCTS to find the best action.
        
        Like running thousands of simulations to see which path leads to victory!
        """
        start_time = time.time()
        
        # Initialize or reuse root node
        if self.reuse_tree and self.previous_root:
            root = self._find_reusable_node(initial_state)
        else:
            root = MCTSNode(initial_state)
        
        # MCTS main loop
        iterations = 0
        while (iterations < self.max_iterations and 
               time.time() - start_time < self.max_time):
            
            # Selection: traverse tree using UCB1
            node = self._select(root)
            
            # Expansion: add new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # Simulation: random playout from current node
            reward = node.simulate()
            
            # Backpropagation: update statistics
            node.backpropagate(reward)
            
            iterations += 1
            
            # Early termination if confident
            if (self.config.get('early_termination', True) and 
                iterations > 100 and 
                self._should_terminate_early(root)):
                break
        
        search_time = time.time() - start_time
        
        # Update statistics
        self.total_searches += 1
        self.total_search_time += search_time
        self.search_statistics['iterations'].append(iterations)
        self.search_statistics['search_time'].append(search_time)
        
        # Store root for potential reuse
        self.previous_root = root
        
        # Get best action and statistics
        best_action = root.best_action()
        action_stats = root.get_action_statistics()
        
        return {
            'best_action': best_action,
            'action_statistics': action_stats,
            'search_iterations': iterations,
            'search_time': search_time,
            'tree_size': self._count_tree_nodes(root),
            'confidence': self._calculate_confidence(root, best_action),
            'expected_value': action_stats.get(f"{best_action.action_type.value}", {}).get('avg_reward', 0.0) if best_action else 0.0
        }
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCB1."""
        current = root
        
        while not current.is_terminal() and current.is_fully_expanded():
            if not current.children:
                break
            current = current.select_child()
        
        return current
    
    def _find_reusable_node(self, state: MCTSGameState) -> MCTSNode:
        """Find a reusable node from previous search tree."""
        if not self.previous_root:
            return MCTSNode(state)
        
        # Simple state matching (could be improved)
        if self._states_match(self.previous_root.state, state):
            return self.previous_root
        
        # Search children for matching state
        for child in self.previous_root.children.values():
            if self._states_match(child.state, state):
                child.parent = None  # Detach from old tree
                return child
        
        # No reusable node found
        return MCTSNode(state)
    
    def _states_match(self, state1: MCTSGameState, state2: MCTSGameState) -> bool:
        """Check if two states are equivalent."""
        return (state1.board_size == state2.board_size and
                state1.mines_count == state2.mines_count and
                state1.revealed_cells == state2.revealed_cells and
                abs(state1.current_multiplier - state2.current_multiplier) < 0.01)
    
    def _should_terminate_early(self, root: MCTSNode) -> bool:
        """Check if search should terminate early due to confidence."""
        if not root.children:
            return False
        
        # Get visit counts for all children
        visit_counts = [child.visits for child in root.children.values()]
        total_visits = sum(visit_counts)
        
        if total_visits < self.config.get('min_visits_for_selection', 10):
            return False
        
        # Check if one action is clearly dominant
        max_visits = max(visit_counts)
        second_max = sorted(visit_counts, reverse=True)[1] if len(visit_counts) > 1 else 0
        
        dominance_ratio = max_visits / max(second_max, 1)
        confidence_threshold = self.config.get('confidence_threshold', 0.8)
        
        return dominance_ratio > (1 / (1 - confidence_threshold))
    
    def _count_tree_nodes(self, root: MCTSNode) -> int:
        """Count total nodes in the search tree."""
        count = 1
        for child in root.children.values():
            count += self._count_tree_nodes(child)
        return count
    
    def _calculate_confidence(self, root: MCTSNode, best_action: Optional[GameAction]) -> float:
        """Calculate confidence in the best action."""
        if not best_action or not root.children:
            return 0.0
        
        best_child = root.children.get(best_action)
        if not best_child:
            return 0.0
        
        total_visits = sum(child.visits for child in root.children.values())
        if total_visits == 0:
            return 0.0
        
        # Confidence based on visit proportion
        visit_confidence = best_child.visits / total_visits
        
        # Confidence based on reward difference
        rewards = [child.total_reward / max(1, child.visits) for child in root.children.values()]
        if len(rewards) > 1:
            best_reward = best_child.total_reward / max(1, best_child.visits)
            avg_other_reward = np.mean([r for r in rewards if r != best_reward])
            reward_confidence = max(0.0, (best_reward - avg_other_reward) / max(abs(best_reward), 1.0))
        else:
            reward_confidence = 1.0
        
        # Combined confidence
        return (visit_confidence + reward_confidence) / 2.0
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        if not self.search_statistics['iterations']:
            return {"status": "No search data available"}
        
        return {
            "total_searches": self.total_searches,
            "avg_iterations": np.mean(self.search_statistics['iterations']),
            "avg_search_time": np.mean(self.search_statistics['search_time']),
            "total_search_time": self.total_search_time,
            "max_iterations": self.max_iterations,
            "exploration_constant": self.exploration_constant,
            "tree_reuse_enabled": self.reuse_tree,
            "last_search_iterations": self.search_statistics['iterations'][-1] if self.search_statistics['iterations'] else 0,
            "last_search_time": self.search_statistics['search_time'][-1] if self.search_statistics['search_time'] else 0.0
        }
    
    def visualize_tree(self, root: MCTSNode, max_depth: int = 3) -> Dict[str, Any]:
        """Create a visualization of the search tree."""
        def node_to_dict(node: MCTSNode, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth:
                return {"truncated": True}
            
            node_data = {
                "visits": node.visits,
                "avg_reward": node.total_reward / max(1, node.visits),
                "ucb1_value": node.ucb1_value(),
                "is_terminal": node.is_terminal(),
                "children_count": len(node.children),
                "children": {}
            }
            
            for action, child in node.children.items():
                action_key = f"{action.action_type.value}"
                if action.position:
                    action_key += f"_{action.position}"
                node_data["children"][action_key] = node_to_dict(child, depth + 1)
            
            return node_data
        
        return node_to_dict(root)

class MCTSStrategy:
    """Integration of MCTS with the existing strategy framework."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.mcts = MonteCarloTreeSearch(self.config.get('mcts_config', {}))
        
        # Strategy parameters
        self.mcts_weight = self.config.get('mcts_weight', 0.7)
        self.fallback_weight = self.config.get('fallback_weight', 0.3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # Performance tracking
        self.decisions_made = 0
        self.mcts_decisions = 0
        self.fallback_decisions = 0
    
    def make_decision(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision using MCTS analysis."""
        self.decisions_made += 1
        
        # Convert to MCTS game state
        mcts_state = self._convert_to_mcts_state(game_state)
        
        # Perform MCTS search
        search_result = self.mcts.search(mcts_state)
        
        # Check confidence
        if search_result['confidence'] >= self.confidence_threshold:
            self.mcts_decisions += 1
            action = search_result['best_action']
            reasoning = f"MCTS recommendation (confidence: {search_result['confidence']:.3f})"
        else:
            self.fallback_decisions += 1
            action = self._fallback_decision(mcts_state)
            reasoning = f"Fallback decision (MCTS confidence too low: {search_result['confidence']:.3f})"
        
        return {
            'action': self._convert_action_to_strategy_format(action),
            'confidence': search_result['confidence'],
            'reasoning': reasoning,
            'mcts_statistics': search_result,
            'expected_value': search_result['expected_value'],
            'search_time': search_result['search_time'],
            'tree_size': search_result['tree_size']
        }
    
    def _convert_to_mcts_state(self, game_state: Dict[str, Any]) -> MCTSGameState:
        """Convert strategy game state to MCTS game state."""
        return MCTSGameState(
            board_size=game_state.get('board_size', (5, 5)),
            mines_count=game_state.get('mines_count', 6),
            revealed_cells=set(game_state.get('revealed_cells', [])),
            mine_positions=set(game_state.get('mine_positions', [])),
            safe_cells=set(game_state.get('safe_cells', [])),
            current_multiplier=game_state.get('current_multiplier', 1.0),
            bankroll=game_state.get('bankroll', 1000.0),
            bet_amount=game_state.get('bet_amount', 100.0)
        )
    
    def _convert_action_to_strategy_format(self, action: Optional[GameAction]) -> str:
        """Convert MCTS action to strategy format."""
        if not action:
            return "cash_out"
        
        if action.action_type == ActionType.CASH_OUT:
            return "cash_out"
        elif action.action_type == ActionType.REVEAL_CELL:
            return f"reveal_{action.position[0]}_{action.position[1]}"
        else:
            return "continue"
    
    def _fallback_decision(self, state: MCTSGameState) -> GameAction:
        """Fallback decision when MCTS confidence is low."""
        # Simple heuristic: cash out if multiplier is high or risk is high
        total_cells = state.board_size[0] * state.board_size[1]
        unrevealed_cells = total_cells - len(state.revealed_cells)
        remaining_mines = state.mines_count - len(state.mine_positions)
        
        if unrevealed_cells <= 0:
            return GameAction(ActionType.CASH_OUT)
        
        mine_probability = remaining_mines / unrevealed_cells
        
        if state.current_multiplier > 3.0 or mine_probability > 0.5:
            return GameAction(ActionType.CASH_OUT)
        else:
            # Pick a random safe-looking cell
            available_positions = []
            for row in range(state.board_size[0]):
                for col in range(state.board_size[1]):
                    pos = (row, col)
                    if pos not in state.revealed_cells:
                        available_positions.append(pos)
            
            if available_positions:
                position = random.choice(available_positions)
                return GameAction(ActionType.REVEAL_CELL, position=position)
            else:
                return GameAction(ActionType.CASH_OUT)
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        mcts_stats = self.mcts.get_search_statistics()
        
        return {
            "strategy_name": "Monte Carlo Tree Search",
            "total_decisions": self.decisions_made,
            "mcts_decisions": self.mcts_decisions,
            "fallback_decisions": self.fallback_decisions,
            "mcts_usage_rate": self.mcts_decisions / max(1, self.decisions_made),
            "mcts_weight": self.mcts_weight,
            "confidence_threshold": self.confidence_threshold,
            **mcts_stats
        }

# Example usage and testing
if __name__ == "__main__":
    # Create MCTS strategy
    mcts_strategy = MCTSStrategy({
        'mcts_config': {
            'max_iterations': 500,
            'max_time': 2.0,
            'exploration_constant': 1.4
        }
    })
    
    # Test with sample game state
    test_game_state = {
        'board_size': (5, 5),
        'mines_count': 6,
        'revealed_cells': [(0, 0), (0, 1), (1, 0)],
        'mine_positions': [],
        'safe_cells': [(0, 0), (0, 1), (1, 0)],
        'current_multiplier': 2.5,
        'bankroll': 1500.0,
        'bet_amount': 100.0
    }
    
    print("🌳 Monte Carlo Tree Search Test")
    
    # Make decision
    start_time = time.time()
    decision = mcts_strategy.make_decision(test_game_state)
    decision_time = time.time() - start_time
    
    print(f"Action: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.3f}")
    print(f"Reasoning: {decision['reasoning']}")
    print(f"Expected Value: {decision['expected_value']:.2f}")
    print(f"Decision Time: {decision_time:.3f}s")
    print(f"Search Iterations: {decision['mcts_statistics']['search_iterations']}")
    print(f"Tree Size: {decision['mcts_statistics']['tree_size']}")
    
    # Get strategy stats
    stats = mcts_strategy.get_strategy_stats()
    print(f"\n📊 MCTS Strategy Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎯 MCTS analysis complete! The tree of possibilities has been explored!")

