"""
Markov Decision Process (MDP) Models

This module implements MDP models to formalize Mines gameplay as a finite MDP
and compare policy evaluation performance to RL approaches. Like creating the
ultimate strategic framework that Senku would use to analyze probability spaces!
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import copy
from enum import Enum
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import itertools

class MDPAction(Enum):
    """Actions available in the MDP."""
    REVEAL_CELL = "reveal_cell"
    CASH_OUT = "cash_out"

@dataclass
class MDPState:
    """State representation for the MDP."""
    revealed_cells: frozenset
    mines_hit: int
    current_multiplier: float
    is_terminal: bool = False
    
    def __hash__(self):
        return hash((self.revealed_cells, self.mines_hit, 
                    round(self.current_multiplier, 2), self.is_terminal))
    
    def __eq__(self, other):
        return (self.revealed_cells == other.revealed_cells and
                self.mines_hit == other.mines_hit and
                abs(self.current_multiplier - other.current_multiplier) < 0.01 and
                self.is_terminal == other.is_terminal)

@dataclass
class MDPTransition:
    """Transition in the MDP."""
    from_state: MDPState
    action: MDPAction
    to_state: MDPState
    probability: float
    reward: float
    cell_position: Optional[Tuple[int, int]] = None

class MinesGameMDP:
    """
    Markov Decision Process model for the Mines game.
    
    Formalizes the Mines game as a finite MDP with:
    - States: (revealed_cells, mines_hit, multiplier, terminal_flag)
    - Actions: {reveal_cell(x,y), cash_out}
    - Transitions: Probabilistic based on mine distribution
    - Rewards: Based on game outcomes and multipliers
    
    Like creating the ultimate probability map that shows every possible
    path through the game space!
    """
    
    def __init__(self, board_size: Tuple[int, int] = (5, 5), mines_count: int = 6,
                 config: Dict[str, Any] = None):
        self.board_size = board_size
        self.mines_count = mines_count
        self.config = config or self._default_config()
        
        # MDP components
        self.states: Set[MDPState] = set()
        self.actions: Set[Tuple[MDPAction, Optional[Tuple[int, int]]]] = set()
        self.transitions: Dict[Tuple[MDPState, Tuple[MDPAction, Optional[Tuple[int, int]]]], List[MDPTransition]] = defaultdict(list)
        self.rewards: Dict[Tuple[MDPState, Tuple[MDPAction, Optional[Tuple[int, int]]]], float] = {}
        
        # Policy and value functions
        self.policy: Dict[MDPState, Tuple[MDPAction, Optional[Tuple[int, int]]]] = {}
        self.value_function: Dict[MDPState, float] = {}
        self.q_function: Dict[Tuple[MDPState, Tuple[MDPAction, Optional[Tuple[int, int]]]], float] = {}
        
        # MDP parameters
        self.discount_factor = self.config.get('discount_factor', 0.95)
        self.convergence_threshold = self.config.get('convergence_threshold', 1e-6)
        self.max_iterations = self.config.get('max_iterations', 1000)
        
        # State space management
        self.max_states = self.config.get('max_states', 10000)
        self.state_abstraction = self.config.get('state_abstraction', True)
        
        # Build MDP
        self._build_mdp()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for MDP."""
        return {
            'discount_factor': 0.95,
            'convergence_threshold': 1e-6,
            'max_iterations': 1000,
            'max_states': 10000,
            'state_abstraction': True,
            'reward_shaping': True,
            'multiplier_increment': 0.1,
            'base_bet': 100.0
        }
    
    def _build_mdp(self):
        """Build the complete MDP model."""
        print("🏗️ Building MDP model...")
        
        # Generate state space
        self._generate_state_space()
        
        # Generate action space
        self._generate_action_space()
        
        # Generate transitions and rewards
        self._generate_transitions_and_rewards()
        
        print(f"✅ MDP built: {len(self.states)} states, {len(self.actions)} actions")
    
    def _generate_state_space(self):
        """Generate the state space for the MDP."""
        total_cells = self.board_size[0] * self.board_size[1]
        
        # Use state abstraction to keep state space manageable
        if self.state_abstraction:
            # Abstract states based on number of revealed cells and multiplier ranges
            for revealed_count in range(min(total_cells - self.mines_count + 1, 15)):
                for mines_hit in range(min(self.mines_count + 1, 3)):
                    for multiplier_level in range(10):  # Discretize multiplier
                        multiplier = 1.0 + multiplier_level * 0.5
                        
                        # Create abstract state
                        revealed_cells = frozenset(range(revealed_count))  # Abstract representation
                        state = MDPState(
                            revealed_cells=revealed_cells,
                            mines_hit=mines_hit,
                            current_multiplier=multiplier,
                            is_terminal=(mines_hit > 0 or revealed_count >= total_cells - self.mines_count)
                        )
                        self.states.add(state)
                        
                        if len(self.states) >= self.max_states:
                            break
                    if len(self.states) >= self.max_states:
                        break
                if len(self.states) >= self.max_states:
                    break
        else:
            # Full state space (exponential - only for small boards)
            if total_cells <= 16:  # Only for very small boards
                for revealed_subset in itertools.combinations(range(total_cells), 
                                                            min(total_cells - self.mines_count, 8)):
                    revealed_cells = frozenset(revealed_subset)
                    multiplier = 1.0 + len(revealed_cells) * 0.1
                    
                    state = MDPState(
                        revealed_cells=revealed_cells,
                        mines_hit=0,
                        current_multiplier=multiplier,
                        is_terminal=False
                    )
                    self.states.add(state)
                    
                    # Terminal state (hit mine)
                    terminal_state = MDPState(
                        revealed_cells=revealed_cells,
                        mines_hit=1,
                        current_multiplier=multiplier,
                        is_terminal=True
                    )
                    self.states.add(terminal_state)
    
    def _generate_action_space(self):
        """Generate the action space for the MDP."""
        # Cash out action (always available)
        self.actions.add((MDPAction.CASH_OUT, None))
        
        # Reveal cell actions
        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                self.actions.add((MDPAction.REVEAL_CELL, (row, col)))
    
    def _generate_transitions_and_rewards(self):
        """Generate transition probabilities and rewards."""
        total_cells = self.board_size[0] * self.board_size[1]
        base_bet = self.config.get('base_bet', 100.0)
        
        # Create a copy of states to avoid modification during iteration
        states_copy = list(self.states)
        
        for state in states_copy:
            if state.is_terminal:
                continue
            
            for action_tuple in self.actions:
                action, position = action_tuple
                
                if action == MDPAction.CASH_OUT:
                    # Cash out: deterministic transition to terminal state
                    terminal_state = MDPState(
                        revealed_cells=state.revealed_cells,
                        mines_hit=state.mines_hit,
                        current_multiplier=state.current_multiplier,
                        is_terminal=True
                    )
                    
                    # Add to states if not exists
                    if terminal_state not in self.states:
                        self.states.add(terminal_state)
                        # Initialize value function for new state
                        self.value_function[terminal_state] = 0.0
                    
                    reward = state.current_multiplier * base_bet
                    transition = MDPTransition(
                        from_state=state,
                        action=action,
                        to_state=terminal_state,
                        probability=1.0,
                        reward=reward
                    )
                    
                    self.transitions[(state, action_tuple)].append(transition)
                    self.rewards[(state, action_tuple)] = reward
                
                elif action == MDPAction.REVEAL_CELL:
                    # Reveal cell: probabilistic transition
                    if position in state.revealed_cells:
                        continue  # Can't reveal already revealed cell
                    
                    # Calculate mine probability
                    unrevealed_count = total_cells - len(state.revealed_cells)
                    remaining_mines = self.mines_count - state.mines_hit
                    
                    if unrevealed_count <= 0:
                        continue
                    
                    mine_probability = remaining_mines / unrevealed_count
                    safe_probability = 1.0 - mine_probability
                    
                    # Safe transition
                    if safe_probability > 0:
                        new_revealed = state.revealed_cells | {len(state.revealed_cells)}  # Abstract
                        new_multiplier = state.current_multiplier * (1 + self.config.get('multiplier_increment', 0.1))
                        
                        safe_state = MDPState(
                            revealed_cells=new_revealed,
                            mines_hit=state.mines_hit,
                            current_multiplier=new_multiplier,
                            is_terminal=(len(new_revealed) >= total_cells - self.mines_count)
                        )
                        
                        # Add to states if not exists
                        if safe_state not in self.states:
                            self.states.add(safe_state)
                            # Initialize value function for new state
                            self.value_function[safe_state] = 0.0
                        
                        safe_reward = 0.0  # Intermediate reward
                        if safe_state.is_terminal:
                            safe_reward = new_multiplier * base_bet  # Win reward
                        
                        safe_transition = MDPTransition(
                            from_state=state,
                            action=action,
                            to_state=safe_state,
                            probability=safe_probability,
                            reward=safe_reward,
                            cell_position=position
                        )
                        
                        self.transitions[(state, action_tuple)].append(safe_transition)
                    
                    # Mine transition
                    if mine_probability > 0:
                        mine_state = MDPState(
                            revealed_cells=state.revealed_cells,
                            mines_hit=state.mines_hit + 1,
                            current_multiplier=state.current_multiplier,
                            is_terminal=True
                        )
                        
                        # Add to states if not exists
                        if mine_state not in self.states:
                            self.states.add(mine_state)
                            # Initialize value function for new state
                            self.value_function[mine_state] = 0.0
                        
                        mine_reward = -base_bet  # Loss
                        
                        mine_transition = MDPTransition(
                            from_state=state,
                            action=action,
                            to_state=mine_state,
                            probability=mine_probability,
                            reward=mine_reward,
                            cell_position=position
                        )
                        
                        self.transitions[(state, action_tuple)].append(mine_transition)
                    
                    # Calculate expected reward for this action
                    expected_reward = sum(t.probability * t.reward 
                                        for t in self.transitions[(state, action_tuple)])
                    self.rewards[(state, action_tuple)] = expected_reward
    
    def value_iteration(self) -> Dict[str, Any]:
        """
        Solve the MDP using value iteration.
        
        Like running the ultimate optimization algorithm to find the perfect
        strategy for every possible situation!
        """
        print("🔄 Running value iteration...")
        
        # Initialize value function
        for state in self.states:
            self.value_function[state] = 0.0
        
        iteration = 0
        converged = False
        
        while iteration < self.max_iterations and not converged:
            old_values = self.value_function.copy()
            max_change = 0.0
            
            for state in self.states:
                if state.is_terminal:
                    continue
                
                # Calculate Q-values for all actions
                q_values = {}
                for action_tuple in self.actions:
                    if (state, action_tuple) in self.transitions:
                        q_value = 0.0
                        for transition in self.transitions[(state, action_tuple)]:
                            q_value += transition.probability * (
                                transition.reward + 
                                self.discount_factor * self.value_function[transition.to_state]
                            )
                        q_values[action_tuple] = q_value
                        self.q_function[(state, action_tuple)] = q_value
                
                # Update value function (Bellman equation)
                if q_values:
                    new_value = max(q_values.values())
                    self.value_function[state] = new_value
                    
                    # Update policy (greedy)
                    best_action = max(q_values.keys(), key=lambda a: q_values[a])
                    self.policy[state] = best_action
                    
                    # Track convergence
                    change = abs(new_value - old_values[state])
                    max_change = max(max_change, change)
            
            # Check convergence
            if max_change < self.convergence_threshold:
                converged = True
            
            iteration += 1
        
        print(f"✅ Value iteration completed in {iteration} iterations")
        
        return {
            'converged': converged,
            'iterations': iteration,
            'max_change': max_change,
            'optimal_value': max(self.value_function.values()) if self.value_function else 0.0
        }
    
    def policy_iteration(self) -> Dict[str, Any]:
        """
        Solve the MDP using policy iteration.
        
        Alternative solution method that iterates between policy evaluation
        and policy improvement.
        """
        print("🔄 Running policy iteration...")
        
        # Initialize random policy
        for state in self.states:
            if not state.is_terminal:
                available_actions = [a for a in self.actions 
                                   if (state, a) in self.transitions]
                if available_actions:
                    self.policy[state] = random.choice(available_actions)
        
        iteration = 0
        policy_stable = False
        
        while iteration < self.max_iterations and not policy_stable:
            # Policy evaluation
            self._policy_evaluation()
            
            # Policy improvement
            policy_changed = self._policy_improvement()
            
            if not policy_changed:
                policy_stable = True
            
            iteration += 1
        
        print(f"✅ Policy iteration completed in {iteration} iterations")
        
        return {
            'converged': policy_stable,
            'iterations': iteration,
            'optimal_value': max(self.value_function.values()) if self.value_function else 0.0
        }
    
    def _policy_evaluation(self):
        """Evaluate the current policy."""
        # Solve system of linear equations for policy evaluation
        for _ in range(100):  # Limited iterations for efficiency
            old_values = self.value_function.copy()
            
            for state in self.states:
                if state.is_terminal:
                    continue
                
                if state in self.policy:
                    action_tuple = self.policy[state]
                    if (state, action_tuple) in self.transitions:
                        value = 0.0
                        for transition in self.transitions[(state, action_tuple)]:
                            value += transition.probability * (
                                transition.reward + 
                                self.discount_factor * self.value_function[transition.to_state]
                            )
                        self.value_function[state] = value
            
            # Check convergence
            max_change = max(abs(self.value_function[s] - old_values[s]) 
                           for s in self.states if s in old_values)
            if max_change < self.convergence_threshold:
                break
    
    def _policy_improvement(self) -> bool:
        """Improve the current policy."""
        policy_changed = False
        
        for state in self.states:
            if state.is_terminal:
                continue
            
            old_action = self.policy.get(state)
            
            # Find best action
            best_action = None
            best_value = float('-inf')
            
            for action_tuple in self.actions:
                if (state, action_tuple) in self.transitions:
                    value = 0.0
                    for transition in self.transitions[(state, action_tuple)]:
                        value += transition.probability * (
                            transition.reward + 
                            self.discount_factor * self.value_function[transition.to_state]
                        )
                    
                    if value > best_value:
                        best_value = value
                        best_action = action_tuple
            
            if best_action and best_action != old_action:
                self.policy[state] = best_action
                policy_changed = True
        
        return policy_changed
    
    def get_optimal_action(self, state: MDPState) -> Optional[Tuple[MDPAction, Optional[Tuple[int, int]]]]:
        """Get the optimal action for a given state."""
        return self.policy.get(state)
    
    def get_state_value(self, state: MDPState) -> float:
        """Get the value of a given state."""
        return self.value_function.get(state, 0.0)
    
    def simulate_episode(self, initial_state: MDPState, max_steps: int = 100) -> Dict[str, Any]:
        """Simulate an episode using the optimal policy."""
        current_state = initial_state
        total_reward = 0.0
        steps = 0
        trajectory = []
        
        while not current_state.is_terminal and steps < max_steps:
            action_tuple = self.get_optimal_action(current_state)
            if not action_tuple:
                break
            
            # Get transitions for this state-action pair
            transitions = self.transitions.get((current_state, action_tuple), [])
            if not transitions:
                break
            
            # Sample next state based on transition probabilities
            probs = [t.probability for t in transitions]
            if sum(probs) > 0:
                probs = [p / sum(probs) for p in probs]  # Normalize
                transition = np.random.choice(transitions, p=probs)
                
                trajectory.append({
                    'state': current_state,
                    'action': action_tuple,
                    'reward': transition.reward,
                    'next_state': transition.to_state
                })
                
                total_reward += transition.reward
                current_state = transition.to_state
                steps += 1
            else:
                break
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'final_state': current_state,
            'trajectory': trajectory,
            'success': current_state.is_terminal and current_state.mines_hit == 0
        }
    
    def compare_with_random_policy(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """Compare optimal policy with random policy."""
        print(f"🎲 Comparing policies over {num_episodes} episodes...")
        
        # Get initial state
        initial_states = [s for s in self.states 
                         if len(s.revealed_cells) == 0 and not s.is_terminal]
        if not initial_states:
            return {"error": "No initial states found"}
        
        initial_state = initial_states[0]
        
        # Simulate with optimal policy
        optimal_rewards = []
        optimal_successes = 0
        
        for _ in range(num_episodes):
            result = self.simulate_episode(initial_state)
            optimal_rewards.append(result['total_reward'])
            if result['success']:
                optimal_successes += 1
        
        # Simulate with random policy
        random_rewards = []
        random_successes = 0
        
        for _ in range(num_episodes):
            result = self._simulate_random_episode(initial_state)
            random_rewards.append(result['total_reward'])
            if result['success']:
                random_successes += 1
        
        return {
            'optimal_policy': {
                'avg_reward': np.mean(optimal_rewards),
                'std_reward': np.std(optimal_rewards),
                'success_rate': optimal_successes / num_episodes,
                'max_reward': max(optimal_rewards),
                'min_reward': min(optimal_rewards)
            },
            'random_policy': {
                'avg_reward': np.mean(random_rewards),
                'std_reward': np.std(random_rewards),
                'success_rate': random_successes / num_episodes,
                'max_reward': max(random_rewards),
                'min_reward': min(random_rewards)
            },
            'improvement': {
                'reward_improvement': np.mean(optimal_rewards) - np.mean(random_rewards),
                'success_improvement': (optimal_successes - random_successes) / num_episodes
            }
        }
    
    def _simulate_random_episode(self, initial_state: MDPState, max_steps: int = 100) -> Dict[str, Any]:
        """Simulate an episode with random policy."""
        current_state = initial_state
        total_reward = 0.0
        steps = 0
        
        while not current_state.is_terminal and steps < max_steps:
            # Random action selection
            available_actions = [a for a in self.actions 
                               if (current_state, a) in self.transitions]
            if not available_actions:
                break
            
            action_tuple = random.choice(available_actions)
            
            # Get transitions
            transitions = self.transitions.get((current_state, action_tuple), [])
            if not transitions:
                break
            
            # Sample next state
            probs = [t.probability for t in transitions]
            if sum(probs) > 0:
                probs = [p / sum(probs) for p in probs]
                transition = np.random.choice(transitions, p=probs)
                
                total_reward += transition.reward
                current_state = transition.to_state
                steps += 1
            else:
                break
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'final_state': current_state,
            'success': current_state.is_terminal and current_state.mines_hit == 0
        }
    
    def visualize_policy(self, save_path: str = None) -> str:
        """Visualize the optimal policy."""
        # Create policy visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Policy action distribution
        action_counts = defaultdict(int)
        for state, action_tuple in self.policy.items():
            action, position = action_tuple
            action_counts[action.value] += 1
        
        if action_counts:
            actions = list(action_counts.keys())
            counts = list(action_counts.values())
            ax1.bar(actions, counts, color=['skyblue', 'lightcoral'])
            ax1.set_title('Policy Action Distribution')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
        
        # Value function distribution
        values = list(self.value_function.values())
        if values:
            ax2.hist(values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.axvline(x=np.mean(values), color='red', linestyle='--', linewidth=2, label='Mean')
            ax2.set_title('Value Function Distribution')
            ax2.set_xlabel('State Value')
            ax2.set_ylabel('Count')
            ax2.legend()
        
        # State space visualization (revealed cells vs multiplier)
        revealed_counts = []
        multipliers = []
        values_for_scatter = []
        
        for state in self.states:
            if not state.is_terminal:
                revealed_counts.append(len(state.revealed_cells))
                multipliers.append(state.current_multiplier)
                values_for_scatter.append(self.value_function.get(state, 0.0))
        
        if revealed_counts and multipliers:
            scatter = ax3.scatter(revealed_counts, multipliers, c=values_for_scatter, 
                                cmap='viridis', alpha=0.6)
            ax3.set_xlabel('Revealed Cells')
            ax3.set_ylabel('Current Multiplier')
            ax3.set_title('State Space (colored by value)')
            plt.colorbar(scatter, ax=ax3, label='State Value')
        
        # Q-function heatmap (sample)
        if self.q_function:
            # Sample Q-values for visualization
            sample_q_values = list(self.q_function.values())[:100]  # First 100
            if sample_q_values:
                q_matrix = np.array(sample_q_values).reshape(-1, 1)
                sns.heatmap(q_matrix, ax=ax4, cmap='RdYlBu_r', cbar=True)
                ax4.set_title('Sample Q-Values Heatmap')
                ax4.set_xlabel('Action Index')
                ax4.set_ylabel('State Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            save_path = f"/home/ubuntu/fusion-project/python-backend/visualizations/mdp_policy_visualization.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
    
    def get_mdp_statistics(self) -> Dict[str, Any]:
        """Get comprehensive MDP statistics."""
        return {
            "mdp_name": "Mines Game MDP",
            "board_size": self.board_size,
            "mines_count": self.mines_count,
            "state_space_size": len(self.states),
            "action_space_size": len(self.actions),
            "transition_count": sum(len(transitions) for transitions in self.transitions.values()),
            "discount_factor": self.discount_factor,
            "convergence_threshold": self.convergence_threshold,
            "state_abstraction": self.state_abstraction,
            "terminal_states": sum(1 for s in self.states if s.is_terminal),
            "non_terminal_states": sum(1 for s in self.states if not s.is_terminal),
            "avg_state_value": np.mean(list(self.value_function.values())) if self.value_function else 0.0,
            "max_state_value": max(self.value_function.values()) if self.value_function else 0.0,
            "min_state_value": min(self.value_function.values()) if self.value_function else 0.0,
            "policy_size": len(self.policy)
        }

class MDPStrategy:
    """Integration of MDP with the existing strategy framework."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize MDP
        board_size = self.config.get('board_size', (5, 5))
        mines_count = self.config.get('mines_count', 6)
        self.mdp = MinesGameMDP(board_size, mines_count, self.config.get('mdp_config', {}))
        
        # Solve MDP
        self.solution_method = self.config.get('solution_method', 'value_iteration')
        self._solve_mdp()
        
        # Performance tracking
        self.decisions_made = 0
        self.mdp_decisions = 0
        self.fallback_decisions = 0
    
    def _solve_mdp(self):
        """Solve the MDP using the specified method."""
        if self.solution_method == 'value_iteration':
            self.solution_result = self.mdp.value_iteration()
        elif self.solution_method == 'policy_iteration':
            self.solution_result = self.mdp.policy_iteration()
        else:
            raise ValueError(f"Unknown solution method: {self.solution_method}")
    
    def make_decision(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision using MDP policy."""
        self.decisions_made += 1
        
        # Convert to MDP state
        mdp_state = self._convert_to_mdp_state(game_state)
        
        # Get optimal action from MDP
        optimal_action = self.mdp.get_optimal_action(mdp_state)
        state_value = self.mdp.get_state_value(mdp_state)
        
        if optimal_action:
            self.mdp_decisions += 1
            action = self._convert_action_to_strategy_format(optimal_action)
            reasoning = f"MDP optimal policy (state value: {state_value:.2f})"
            confidence = min(1.0, abs(state_value) / 100.0)  # Normalize confidence
        else:
            self.fallback_decisions += 1
            action = "cash_out"
            reasoning = "Fallback decision (no MDP policy available)"
            confidence = 0.5
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'state_value': state_value,
            'mdp_state': mdp_state,
            'optimal_action': optimal_action
        }
    
    def _convert_to_mdp_state(self, game_state: Dict[str, Any]) -> MDPState:
        """Convert strategy game state to MDP state."""
        revealed_cells = frozenset(range(len(game_state.get('revealed_cells', []))))  # Abstract
        mines_hit = len(game_state.get('mine_positions', []))
        current_multiplier = game_state.get('current_multiplier', 1.0)
        
        # Check if terminal
        total_cells = game_state.get('board_size', (5, 5))[0] * game_state.get('board_size', (5, 5))[1]
        mines_count = game_state.get('mines_count', 6)
        is_terminal = (mines_hit > 0 or len(revealed_cells) >= total_cells - mines_count)
        
        return MDPState(
            revealed_cells=revealed_cells,
            mines_hit=mines_hit,
            current_multiplier=current_multiplier,
            is_terminal=is_terminal
        )
    
    def _convert_action_to_strategy_format(self, action_tuple: Tuple[MDPAction, Optional[Tuple[int, int]]]) -> str:
        """Convert MDP action to strategy format."""
        action, position = action_tuple
        
        if action == MDPAction.CASH_OUT:
            return "cash_out"
        elif action == MDPAction.REVEAL_CELL and position:
            return f"reveal_{position[0]}_{position[1]}"
        else:
            return "continue"
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        mdp_stats = self.mdp.get_mdp_statistics()
        
        return {
            "strategy_name": "Markov Decision Process",
            "solution_method": self.solution_method,
            "solution_converged": self.solution_result.get('converged', False),
            "solution_iterations": self.solution_result.get('iterations', 0),
            "total_decisions": self.decisions_made,
            "mdp_decisions": self.mdp_decisions,
            "fallback_decisions": self.fallback_decisions,
            "mdp_usage_rate": self.mdp_decisions / max(1, self.decisions_made),
            **mdp_stats
        }

# Example usage and testing
if __name__ == "__main__":
    # Create MDP strategy
    mdp_strategy = MDPStrategy({
        'board_size': (4, 4),  # Smaller board for testing
        'mines_count': 4,
        'solution_method': 'value_iteration',
        'mdp_config': {
            'discount_factor': 0.9,
            'max_iterations': 500,
            'state_abstraction': True
        }
    })
    
    print("🎯 Markov Decision Process Test")
    
    # Test with sample game state
    test_game_state = {
        'board_size': (4, 4),
        'mines_count': 4,
        'revealed_cells': [(0, 0), (0, 1)],
        'mine_positions': [],
        'safe_cells': [(0, 0), (0, 1)],
        'current_multiplier': 1.5,
        'bankroll': 1500.0,
        'bet_amount': 100.0
    }
    
    # Make decision
    decision = mdp_strategy.make_decision(test_game_state)
    
    print(f"Action: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.3f}")
    print(f"Reasoning: {decision['reasoning']}")
    print(f"State Value: {decision['state_value']:.2f}")
    
    # Get strategy stats
    stats = mdp_strategy.get_strategy_stats()
    print(f"\n📊 MDP Strategy Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Compare policies
    comparison = mdp_strategy.mdp.compare_with_random_policy(100)
    print(f"\n🎲 Policy Comparison:")
    print(f"  Optimal Policy Avg Reward: {comparison['optimal_policy']['avg_reward']:.2f}")
    print(f"  Random Policy Avg Reward: {comparison['random_policy']['avg_reward']:.2f}")
    print(f"  Improvement: {comparison['improvement']['reward_improvement']:.2f}")
    
    # Create visualization
    viz_path = mdp_strategy.mdp.visualize_policy()
    print(f"\n📊 MDP visualization saved to: {viz_path}")
    
    print("\n🎯 MDP analysis complete! The optimal policy has been computed!")

