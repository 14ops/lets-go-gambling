"""
Strategy Auto-Evolution Module

This module implements evolutionary algorithms to breed new strategies by cross-mutating
parameters of existing ones. Like creating the ultimate fusion warriors through genetic
algorithms and natural selection!
"""

import numpy as np
import random
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import copy
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@dataclass
class StrategyGenes:
    """Genetic representation of a strategy's parameters."""
    # Core behavioral genes
    risk_tolerance: float = 0.5
    aggression_factor: float = 0.5
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    
    # Decision-making genes
    confidence_threshold: float = 0.7
    adaptation_speed: float = 0.05
    patience_level: float = 0.5
    analytical_weight: float = 0.5
    
    # Risk management genes
    max_position_size: float = 0.1
    stop_loss_threshold: float = 0.05
    take_profit_threshold: float = 0.2
    kelly_multiplier: float = 0.25
    
    # Behavioral pattern genes
    bluff_probability: float = 0.1
    deception_threshold: float = 0.6
    psychological_weight: float = 0.3
    pattern_recognition_depth: int = 100
    
    # Meta-learning genes
    meta_learning_rate: float = 0.001
    ensemble_weight: float = 0.5
    strategy_switching_threshold: float = 0.8
    performance_memory: int = 1000
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2) -> 'StrategyGenes':
        """Apply random mutations to the genes."""
        mutated = copy.deepcopy(self)
        
        for field_name, field_value in asdict(self).items():
            if random.random() < mutation_rate:
                if isinstance(field_value, float):
                    # Gaussian mutation for float values
                    mutation = np.random.normal(0, mutation_strength)
                    new_value = field_value + mutation
                    
                    # Clamp to reasonable bounds
                    if field_name.endswith('_rate') or field_name.endswith('_threshold') or field_name.endswith('_weight'):
                        new_value = np.clip(new_value, 0.0, 1.0)
                    elif field_name == 'kelly_multiplier':
                        new_value = np.clip(new_value, 0.0, 0.5)
                    else:
                        new_value = max(0.0, new_value)
                    
                    setattr(mutated, field_name, new_value)
                
                elif isinstance(field_value, int):
                    # Integer mutation
                    mutation = random.randint(-10, 10)
                    new_value = max(1, field_value + mutation)
                    setattr(mutated, field_name, new_value)
        
        return mutated
    
    def crossover(self, other: 'StrategyGenes', crossover_rate: float = 0.5) -> Tuple['StrategyGenes', 'StrategyGenes']:
        """Create two offspring through genetic crossover."""
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other)
        
        for field_name in asdict(self).keys():
            if random.random() < crossover_rate:
                # Swap genes between parents
                value1 = getattr(child1, field_name)
                value2 = getattr(child2, field_name)
                setattr(child1, field_name, value2)
                setattr(child2, field_name, value1)
        
        return child1, child2
    
    def distance(self, other: 'StrategyGenes') -> float:
        """Calculate genetic distance between two strategies."""
        total_distance = 0.0
        field_count = 0
        
        for field_name, value1 in asdict(self).items():
            value2 = getattr(other, field_name)
            if isinstance(value1, (int, float)):
                # Normalize by expected range
                if field_name.endswith('_rate') or field_name.endswith('_threshold') or field_name.endswith('_weight'):
                    distance = abs(value1 - value2)  # Already normalized 0-1
                else:
                    distance = abs(value1 - value2) / max(abs(value1), abs(value2), 1.0)
                
                total_distance += distance
                field_count += 1
        
        return total_distance / field_count if field_count > 0 else 0.0

@dataclass
class EvolutionaryStrategy:
    """A strategy individual in the evolutionary population."""
    genes: StrategyGenes
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    strategy_id: str = ""
    performance_history: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.strategy_id:
            self.strategy_id = f"evolved_{random.randint(1000, 9999)}"
    
    def evaluate_fitness(self, performance_data: Dict[str, Any]) -> float:
        """Evaluate fitness based on performance metrics."""
        # Multi-objective fitness function
        win_rate = performance_data.get('win_rate', 0.0)
        avg_return = performance_data.get('avg_return', 0.0)
        sharpe_ratio = performance_data.get('sharpe_ratio', 0.0)
        max_drawdown = performance_data.get('max_drawdown', 1.0)
        consistency = performance_data.get('consistency', 0.0)
        
        # Weighted fitness calculation
        fitness = (
            0.25 * win_rate +
            0.25 * max(0, avg_return) +  # Only positive returns contribute
            0.20 * max(0, sharpe_ratio) +
            0.15 * (1.0 - max_drawdown) +  # Lower drawdown is better
            0.15 * consistency
        )
        
        # Age penalty to encourage fresh strategies
        age_penalty = min(0.1, self.age * 0.001)
        fitness = max(0.0, fitness - age_penalty)
        
        self.fitness = fitness
        self.performance_history.append(fitness)
        
        return fitness

class StrategyAutoEvolution:
    """
    Strategy Auto-Evolution Module
    
    Uses evolutionary algorithms to breed new strategies by cross-mutating parameters
    of existing ones. Like being the ultimate Pokemon breeder but for AI strategies!
    
    Features:
    - Genetic algorithm with crossover and mutation
    - Multi-objective fitness evaluation
    - Population diversity maintenance
    - Elite preservation and selection pressure
    - Meta-AI selector for dynamic strategy picking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Evolution parameters
        self.population_size = self.config.get('population_size', 50)
        self.elite_size = self.config.get('elite_size', 10)
        self.mutation_rate = self.config.get('mutation_rate', 0.15)
        self.crossover_rate = self.config.get('crossover_rate', 0.7)
        self.selection_pressure = self.config.get('selection_pressure', 2.0)
        
        # Population management
        self.population: List[EvolutionaryStrategy] = []
        self.generation = 0
        self.evolution_history = []
        self.best_strategies = []
        
        # Meta-AI selector
        self.meta_ai_enabled = self.config.get('meta_ai_enabled', True)
        self.epoch_length = self.config.get('epoch_length', 1000)
        self.current_epoch_performance = defaultdict(list)
        self.meta_selector_weights = defaultdict(float)
        
        # Diversity maintenance
        self.diversity_threshold = self.config.get('diversity_threshold', 0.1)
        self.diversity_pressure = self.config.get('diversity_pressure', 0.2)
        
        # Performance tracking
        self.performance_database = {}
        self.evaluation_rounds = self.config.get('evaluation_rounds', 100)
        
        # Initialize with base strategies
        self._initialize_population()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for evolution parameters."""
        return {
            'population_size': 50,
            'elite_size': 10,
            'mutation_rate': 0.15,
            'crossover_rate': 0.7,
            'selection_pressure': 2.0,
            'meta_ai_enabled': True,
            'epoch_length': 1000,
            'diversity_threshold': 0.1,
            'diversity_pressure': 0.2,
            'evaluation_rounds': 100,
            'max_generations': 1000,
            'convergence_threshold': 0.001,
            'elite_preservation': True,
            'adaptive_mutation': True
        }
    
    def _initialize_population(self):
        """Initialize the population with base strategies and random variants."""
        # Base strategy templates
        base_strategies = {
            'takeshi': StrategyGenes(
                risk_tolerance=0.8, aggression_factor=0.9, learning_rate=0.02,
                exploration_rate=0.3, confidence_threshold=0.6
            ),
            'lelouch': StrategyGenes(
                risk_tolerance=0.6, aggression_factor=0.5, learning_rate=0.05,
                adaptation_speed=0.08, analytical_weight=0.7
            ),
            'kazuya': StrategyGenes(
                risk_tolerance=0.2, aggression_factor=0.3, patience_level=0.8,
                stop_loss_threshold=0.03, take_profit_threshold=0.15
            ),
            'senku': StrategyGenes(
                learning_rate=0.01, analytical_weight=0.9, pattern_recognition_depth=1000,
                confidence_threshold=0.8, meta_learning_rate=0.005
            ),
            'okabe': StrategyGenes(
                psychological_weight=0.4, bluff_probability=0.15, deception_threshold=0.6,
                strategy_switching_threshold=0.75, ensemble_weight=0.6
            )
        }
        
        # Add base strategies to population
        for name, genes in base_strategies.items():
            strategy = EvolutionaryStrategy(
                genes=genes,
                generation=0,
                strategy_id=f"base_{name}"
            )
            self.population.append(strategy)
        
        # Fill remaining population with random variants
        while len(self.population) < self.population_size:
            # Pick a random base strategy to mutate
            base_strategy = random.choice(list(base_strategies.values()))
            mutated_genes = base_strategy.mutate(mutation_rate=0.5, mutation_strength=0.3)
            
            strategy = EvolutionaryStrategy(
                genes=mutated_genes,
                generation=0,
                strategy_id=f"random_{len(self.population)}"
            )
            self.population.append(strategy)
    
    def evolve_generation(self, performance_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evolve one generation of strategies.
        
        Like running a Pokemon breeding center but for AI strategies!
        """
        # Evaluate fitness for all strategies
        self._evaluate_population_fitness(performance_data)
        
        # Record generation statistics
        generation_stats = self._calculate_generation_stats()
        self.evolution_history.append(generation_stats)
        
        # Check for convergence
        if self._check_convergence():
            return {
                'converged': True,
                'generation': self.generation,
                'best_fitness': generation_stats['best_fitness'],
                'population_diversity': generation_stats['diversity']
            }
        
        # Selection and reproduction
        new_population = self._create_next_generation()
        
        # Replace population
        self.population = new_population
        self.generation += 1
        
        # Update meta-AI selector
        if self.meta_ai_enabled:
            self._update_meta_selector(performance_data)
        
        return {
            'converged': False,
            'generation': self.generation,
            'best_fitness': generation_stats['best_fitness'],
            'population_diversity': generation_stats['diversity'],
            'new_strategies': len([s for s in new_population if s.generation == self.generation])
        }
    
    def _evaluate_population_fitness(self, performance_data: Dict[str, Dict[str, Any]]):
        """Evaluate fitness for all strategies in the population."""
        for strategy in self.population:
            if strategy.strategy_id in performance_data:
                strategy.evaluate_fitness(performance_data[strategy.strategy_id])
            else:
                # Simulate performance for new strategies
                simulated_performance = self._simulate_strategy_performance(strategy)
                strategy.evaluate_fitness(simulated_performance)
            
            strategy.age += 1
    
    def _simulate_strategy_performance(self, strategy: EvolutionaryStrategy) -> Dict[str, Any]:
        """Simulate performance for a strategy based on its genes."""
        genes = strategy.genes
        
        # Base performance influenced by genetic parameters
        base_win_rate = 0.5 + (genes.analytical_weight - 0.5) * 0.3
        base_return = (genes.risk_tolerance - 0.5) * 0.1
        
        # Add noise and genetic influence
        win_rate = np.clip(
            base_win_rate + np.random.normal(0, 0.1) + genes.learning_rate * 0.5,
            0.0, 1.0
        )
        
        avg_return = base_return + np.random.normal(0, 0.05)
        
        # Calculate other metrics
        volatility = genes.risk_tolerance * 0.3 + np.random.normal(0, 0.05)
        sharpe_ratio = avg_return / max(volatility, 0.01) if volatility > 0 else 0
        max_drawdown = genes.risk_tolerance * 0.4 + np.random.normal(0, 0.1)
        consistency = 1.0 - abs(genes.exploration_rate - 0.1) * 2
        
        return {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'consistency': np.clip(consistency, 0.0, 1.0),
            'volatility': max(0.01, volatility)
        }
    
    def _calculate_generation_stats(self) -> Dict[str, Any]:
        """Calculate statistics for the current generation."""
        fitnesses = [s.fitness for s in self.population]
        
        # Diversity calculation
        diversity = self._calculate_population_diversity()
        
        # Age distribution
        ages = [s.age for s in self.population]
        
        return {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'worst_fitness': min(fitnesses),
            'fitness_std': np.std(fitnesses),
            'diversity': diversity,
            'avg_age': np.mean(ages),
            'max_age': max(ages),
            'elite_count': self.elite_size
        }
    
    def _calculate_population_diversity(self) -> float:
        """Calculate genetic diversity of the population."""
        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self.population[i].genes.distance(self.population[j].genes)
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _check_convergence(self) -> bool:
        """Check if the population has converged."""
        if len(self.evolution_history) < 10:
            return False
        
        # Check fitness improvement over last 10 generations
        recent_best = [gen['best_fitness'] for gen in self.evolution_history[-10:]]
        improvement = max(recent_best) - min(recent_best)
        
        # Check diversity
        current_diversity = self.evolution_history[-1]['diversity']
        
        return (improvement < self.config['convergence_threshold'] and 
                current_diversity < self.diversity_threshold)
    
    def _create_next_generation(self) -> List[EvolutionaryStrategy]:
        """Create the next generation through selection, crossover, and mutation."""
        new_population = []
        
        # Elite preservation
        if self.config.get('elite_preservation', True):
            elite = self._select_elite()
            new_population.extend(elite)
        
        # Generate offspring to fill remaining population
        while len(new_population) < self.population_size:
            # Tournament selection for parents
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1_genes, child2_genes = parent1.genes.crossover(parent2.genes)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child1_genes = child1_genes.mutate(
                        mutation_rate=self._adaptive_mutation_rate(),
                        mutation_strength=0.2
                    )
                if random.random() < self.mutation_rate:
                    child2_genes = child2_genes.mutate(
                        mutation_rate=self._adaptive_mutation_rate(),
                        mutation_strength=0.2
                    )
                
                # Create offspring
                child1 = EvolutionaryStrategy(
                    genes=child1_genes,
                    generation=self.generation + 1,
                    parent_ids=[parent1.strategy_id, parent2.strategy_id]
                )
                child2 = EvolutionaryStrategy(
                    genes=child2_genes,
                    generation=self.generation + 1,
                    parent_ids=[parent1.strategy_id, parent2.strategy_id]
                )
                
                new_population.extend([child1, child2])
            else:
                # Direct copy with possible mutation
                child_genes = copy.deepcopy(parent1.genes)
                if random.random() < self.mutation_rate:
                    child_genes = child_genes.mutate()
                
                child = EvolutionaryStrategy(
                    genes=child_genes,
                    generation=self.generation + 1,
                    parent_ids=[parent1.strategy_id]
                )
                new_population.append(child)
        
        # Trim to exact population size
        return new_population[:self.population_size]
    
    def _select_elite(self) -> List[EvolutionaryStrategy]:
        """Select elite strategies for preservation."""
        sorted_population = sorted(self.population, key=lambda s: s.fitness, reverse=True)
        elite = sorted_population[:self.elite_size]
        
        # Create copies for next generation
        elite_copies = []
        for strategy in elite:
            elite_copy = EvolutionaryStrategy(
                genes=copy.deepcopy(strategy.genes),
                fitness=strategy.fitness,
                age=strategy.age,
                generation=self.generation + 1,
                parent_ids=[strategy.strategy_id],
                strategy_id=f"elite_{strategy.strategy_id}_{self.generation + 1}",
                performance_history=strategy.performance_history.copy()
            )
            elite_copies.append(elite_copy)
        
        return elite_copies
    
    def _tournament_selection(self, tournament_size: int = 3) -> EvolutionaryStrategy:
        """Select a strategy using tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        
        # Apply selection pressure
        tournament_with_pressure = []
        for strategy in tournament:
            adjusted_fitness = strategy.fitness ** self.selection_pressure
            tournament_with_pressure.append((strategy, adjusted_fitness))
        
        # Select best from tournament
        return max(tournament_with_pressure, key=lambda x: x[1])[0]
    
    def _adaptive_mutation_rate(self) -> float:
        """Calculate adaptive mutation rate based on population diversity."""
        if not self.config.get('adaptive_mutation', True):
            return self.mutation_rate
        
        current_diversity = self._calculate_population_diversity()
        
        # Increase mutation rate when diversity is low
        if current_diversity < self.diversity_threshold:
            return min(0.5, self.mutation_rate * 2.0)
        else:
            return self.mutation_rate
    
    def _update_meta_selector(self, performance_data: Dict[str, Dict[str, Any]]):
        """Update the meta-AI selector weights based on performance."""
        # Track performance for current epoch
        for strategy_id, perf in performance_data.items():
            self.current_epoch_performance[strategy_id].append(perf.get('avg_return', 0.0))
        
        # Update weights every epoch
        if sum(len(perfs) for perfs in self.current_epoch_performance.values()) >= self.epoch_length:
            self._rebalance_meta_selector()
            self.current_epoch_performance.clear()
    
    def _rebalance_meta_selector(self):
        """Rebalance meta-selector weights based on epoch performance."""
        epoch_performance = {}
        
        for strategy_id, returns in self.current_epoch_performance.items():
            if returns:
                epoch_performance[strategy_id] = {
                    'avg_return': np.mean(returns),
                    'sharpe_ratio': np.mean(returns) / max(np.std(returns), 0.01),
                    'consistency': 1.0 - np.std(returns)
                }
        
        # Calculate new weights using softmax
        if epoch_performance:
            scores = []
            strategy_ids = []
            
            for strategy_id, perf in epoch_performance.items():
                score = (
                    0.4 * perf['avg_return'] +
                    0.4 * perf['sharpe_ratio'] +
                    0.2 * perf['consistency']
                )
                scores.append(score)
                strategy_ids.append(strategy_id)
            
            # Softmax normalization
            exp_scores = np.exp(np.array(scores) - np.max(scores))
            weights = exp_scores / np.sum(exp_scores)
            
            # Update meta-selector weights
            for strategy_id, weight in zip(strategy_ids, weights):
                self.meta_selector_weights[strategy_id] = weight
    
    def get_best_strategies(self, top_k: int = 5) -> List[EvolutionaryStrategy]:
        """Get the top-k best strategies from current population."""
        sorted_population = sorted(self.population, key=lambda s: s.fitness, reverse=True)
        return sorted_population[:top_k]
    
    def get_meta_ai_recommendation(self) -> Optional[str]:
        """Get meta-AI recommendation for strategy selection."""
        if not self.meta_ai_enabled or not self.meta_selector_weights:
            return None
        
        # Select strategy based on weights
        strategies = list(self.meta_selector_weights.keys())
        weights = list(self.meta_selector_weights.values())
        
        if strategies and weights:
            return np.random.choice(strategies, p=weights)
        
        return None
    
    def export_strategy(self, strategy: EvolutionaryStrategy, filepath: str):
        """Export a strategy to a JSON file."""
        strategy_data = {
            'strategy_id': strategy.strategy_id,
            'generation': strategy.generation,
            'fitness': strategy.fitness,
            'age': strategy.age,
            'parent_ids': strategy.parent_ids,
            'genes': asdict(strategy.genes),
            'performance_history': strategy.performance_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(strategy_data, f, indent=2)
    
    def import_strategy(self, filepath: str) -> EvolutionaryStrategy:
        """Import a strategy from a JSON file."""
        with open(filepath, 'r') as f:
            strategy_data = json.load(f)
        
        genes = StrategyGenes(**strategy_data['genes'])
        strategy = EvolutionaryStrategy(
            genes=genes,
            fitness=strategy_data['fitness'],
            age=strategy_data['age'],
            generation=strategy_data['generation'],
            parent_ids=strategy_data['parent_ids'],
            strategy_id=strategy_data['strategy_id'],
            performance_history=strategy_data['performance_history']
        )
        
        return strategy
    
    def visualize_evolution(self, save_path: str = None) -> str:
        """Create visualization of evolution progress."""
        if not self.evolution_history:
            return "No evolution history to visualize"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        generations = [gen['generation'] for gen in self.evolution_history]
        
        # Fitness evolution
        best_fitness = [gen['best_fitness'] for gen in self.evolution_history]
        avg_fitness = [gen['avg_fitness'] for gen in self.evolution_history]
        
        ax1.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        ax1.plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Diversity evolution
        diversity = [gen['diversity'] for gen in self.evolution_history]
        ax2.plot(generations, diversity, 'g-', linewidth=2)
        ax2.axhline(y=self.diversity_threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Population Diversity')
        ax2.set_title('Genetic Diversity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Age distribution
        avg_age = [gen['avg_age'] for gen in self.evolution_history]
        max_age = [gen['max_age'] for gen in self.evolution_history]
        
        ax3.plot(generations, avg_age, 'purple', label='Average Age', linewidth=2)
        ax3.plot(generations, max_age, 'orange', label='Maximum Age', linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Age')
        ax3.set_title('Population Age Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Fitness distribution (current generation)
        current_fitnesses = [s.fitness for s in self.population]
        ax4.hist(current_fitnesses, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(x=np.mean(current_fitnesses), color='red', linestyle='--', linewidth=2, label='Mean')
        ax4.set_xlabel('Fitness')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Current Fitness Distribution (Gen {self.generation})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            save_path = f"/home/ubuntu/fusion-project/python-backend/visualizations/evolution_progress_gen_{self.generation}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics."""
        if not self.population:
            return {"status": "No population data"}
        
        current_fitnesses = [s.fitness for s in self.population]
        best_strategy = max(self.population, key=lambda s: s.fitness)
        
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": max(current_fitnesses),
            "average_fitness": np.mean(current_fitnesses),
            "fitness_std": np.std(current_fitnesses),
            "population_diversity": self._calculate_population_diversity(),
            "best_strategy_id": best_strategy.strategy_id,
            "best_strategy_age": best_strategy.age,
            "elite_count": self.elite_size,
            "meta_ai_enabled": self.meta_ai_enabled,
            "meta_selector_strategies": len(self.meta_selector_weights),
            "convergence_status": "Converged" if self._check_convergence() else "Evolving",
            "total_generations": len(self.evolution_history),
            "mutation_rate": self._adaptive_mutation_rate(),
            "crossover_rate": self.crossover_rate
        }

# Example usage and testing
if __name__ == "__main__":
    # Create evolution module
    evolution = StrategyAutoEvolution()
    
    print("🧬 Strategy Auto-Evolution Module Test")
    print(f"Initial population size: {len(evolution.population)}")
    
    # Simulate evolution for a few generations
    for generation in range(5):
        # Simulate performance data
        performance_data = {}
        for strategy in evolution.population:
            performance_data[strategy.strategy_id] = {
                'win_rate': random.uniform(0.4, 0.8),
                'avg_return': random.uniform(-0.05, 0.1),
                'sharpe_ratio': random.uniform(0.0, 2.0),
                'max_drawdown': random.uniform(0.05, 0.3),
                'consistency': random.uniform(0.3, 0.9)
            }
        
        # Evolve generation
        result = evolution.evolve_generation(performance_data)
        
        print(f"\nGeneration {generation + 1}:")
        print(f"  Best fitness: {result['best_fitness']:.4f}")
        print(f"  Population diversity: {result['population_diversity']:.4f}")
        print(f"  Converged: {result['converged']}")
        
        if result['converged']:
            print("  🎯 Evolution converged!")
            break
    
    # Get best strategies
    best_strategies = evolution.get_best_strategies(3)
    print(f"\n🏆 Top 3 Strategies:")
    for i, strategy in enumerate(best_strategies, 1):
        print(f"  {i}. {strategy.strategy_id} (Fitness: {strategy.fitness:.4f}, Age: {strategy.age})")
    
    # Meta-AI recommendation
    recommendation = evolution.get_meta_ai_recommendation()
    if recommendation:
        print(f"\n🤖 Meta-AI recommends: {recommendation}")
    
    # Create visualization
    viz_path = evolution.visualize_evolution()
    print(f"\n📊 Evolution visualization saved to: {viz_path}")
    
    # Get comprehensive stats
    stats = evolution.get_evolution_stats()
    print(f"\n📈 Evolution Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 Evolution module test completed! The strongest strategies have emerged!")

