import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
import random
import sympy as sp
from sympy import symbols, simplify, latex
import warnings
warnings.filterwarnings('ignore')

class SymbolicRegressionEngine:
    """Lightweight symbolic regression using genetic programming"""
    
    def __init__(self, population_size=50, generations=20, tournament_size=3):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.x = symbols('x')
        
        # Define primitive functions
        self.functions = {
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
            'div': self._safe_div,
            'sin': lambda a: sp.sin(a),
            'cos': lambda a: sp.cos(a),
            'exp': lambda a: sp.exp(a),
            'log': self._safe_log,
            'sqrt': self._safe_sqrt,
            'square': lambda a: a**2,
            'cube': lambda a: a**3,
            'pow': lambda a, b: a**abs(b),
        }
        
        # Define terminals (will include x and constants)
        self.constants = [-2, -1, -0.5, 0, 0.5, 1, 2, 3.14159, 2.718]
    
    def _safe_div(self, a, b):
        """Safe division to avoid division by zero"""
        return sp.Piecewise((a/b, sp.Ne(b, 0)), (1, True))
    
    def _safe_log(self, a):
        """Safe logarithm for positive values only"""
        return sp.Piecewise((sp.log(sp.Abs(a) + 1e-10), True), (0, True))
    
    def _safe_sqrt(self, a):
        """Safe square root for non-negative values"""
        return sp.sqrt(sp.Abs(a))
    
    def find_equations(self, x_data, y_data, max_equations=5):
        """Find symbolic equations that fit the data"""
        
        if len(x_data) < 3:
            return []
        
        # Store data
        self.x_data = x_data
        self.y_data = y_data
        
        # Initialize population
        population = self._initialize_population()
        
        best_individuals = []
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(individual) for individual in population]
            
            # Store best individuals
            for i, individual in enumerate(population):
                if fitness_scores[i] > 0.1:  # Minimum fitness threshold
                    best_individuals.append({
                        'expression': individual,
                        'fitness': fitness_scores[i],
                        'generation': generation
                    })
            
            # Selection and breeding
            new_population = []
            for _ in range(self.population_size):
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover and mutation
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Process and return best equations
        return self._process_best_equations(best_individuals, max_equations)
    
    def _initialize_population(self):
        """Create initial random population of expressions"""
        population = []
        
        for _ in range(self.population_size):
            # Create random expression tree
            depth = random.randint(1, 4)
            expr = self._generate_random_expression(depth)
            population.append(expr)
        
        return population
    
    def _generate_random_expression(self, max_depth):
        """Generate a random symbolic expression"""
        if max_depth == 0 or random.random() < 0.3:
            # Terminal node
            if random.random() < 0.5:
                return self.x
            else:
                return random.choice(self.constants)
        else:
            # Function node
            func_name = random.choice(list(self.functions.keys()))
            func = self.functions[func_name]
            
            if func_name in ['sin', 'cos', 'exp', 'log', 'sqrt', 'square', 'cube']:
                # Unary function
                arg = self._generate_random_expression(max_depth - 1)
                return func(arg)
            else:
                # Binary function
                arg1 = self._generate_random_expression(max_depth - 1)
                arg2 = self._generate_random_expression(max_depth - 1)
                return func(arg1, arg2)
    
    def _evaluate_fitness(self, expression):
        """Evaluate fitness of an expression"""
        try:
            # Convert to numerical function
            expr_func = sp.lambdify(self.x, expression, 'numpy')
            
            # Calculate predictions
            y_pred = expr_func(self.x_data)
            
            # Handle any invalid results
            if np.any(~np.isfinite(y_pred)):
                return 0
            
            # Calculate R² score
            r2 = r2_score(self.y_data, y_pred)
            
            # Penalize overly complex expressions
            complexity_penalty = max(0, len(str(expression)) - 50) * 0.01
            
            return max(0, r2 - complexity_penalty)
            
        except Exception as e:
            return 0
    
    def _tournament_selection(self, population, fitness_scores):
        """Select individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), 
                                         min(self.tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        best_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[best_index]
    
    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        try:
            # Simple crossover: randomly choose parts from each parent
            if random.random() < 0.5:
                return parent1
            else:
                return parent2
        except:
            return parent1
    
    def _mutate(self, expression):
        """Mutate an expression"""
        if random.random() < 0.1:  # 10% mutation rate
            try:
                # Simple mutation: replace with new random expression
                return self._generate_random_expression(random.randint(1, 3))
            except:
                return expression
        return expression
    
    def _process_best_equations(self, best_individuals, max_equations):
        """Process and return the best unique equations"""
        if not best_individuals:
            return []
        
        # Sort by fitness
        best_individuals.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Remove duplicates and process
        unique_equations = []
        seen_expressions = set()
        
        for individual in best_individuals:
            try:
                expr = individual['expression']
                simplified = simplify(expr)
                expr_str = str(simplified)
                
                # Check for uniqueness
                if expr_str not in seen_expressions and len(unique_equations) < max_equations:
                    seen_expressions.add(expr_str)
                    
                    # Calculate additional metrics
                    complexity = len(expr_str)
                    
                    unique_equations.append({
                        'equation': expr_str,
                        'expression': simplified,
                        'fitness': individual['fitness'],
                        'complexity': complexity,
                        'generation_found': individual['generation']
                    })
            
            except Exception as e:
                continue
        
        return unique_equations
    
    def predict(self, equation_info, x_values):
        """Generate predictions using a symbolic equation"""
        try:
            expression = equation_info['expression']
            expr_func = sp.lambdify(self.x, expression, 'numpy')
            return expr_func(x_values)
        except Exception as e:
            return np.full_like(x_values, np.nan)
    
    def get_latex_equation(self, equation_info):
        """Convert equation to LaTeX format"""
        try:
            expression = equation_info['expression']
            return latex(expression)
        except:
            return equation_info['equation']

class SimpleSymbolicRegressor:
    """Simplified symbolic regression for basic patterns"""
    
    def __init__(self):
        self.x = symbols('x')
        self.fitted_expression = None
        self.fitness_score = 0
    
    def fit(self, X, y):
        """Fit simple symbolic patterns to data"""
        x_data = X.flatten() if len(X.shape) > 1 else X
        
        # Try common patterns
        patterns = [
            ('linear', lambda x, a, b: a + b * x),
            ('quadratic', lambda x, a, b, c: a + b * x + c * x**2),
            ('exponential', lambda x, a, b: a * np.exp(b * x)),
            ('power', lambda x, a, b: a * x**b),
            ('inverse', lambda x, a: a / x),
            ('logarithmic', lambda x, a, b: a * np.log(x) + b),
        ]
        
        best_pattern = None
        best_score = -1
        
        for pattern_name, pattern_func in patterns:
            try:
                from scipy.optimize import curve_fit
                
                # Skip patterns that don't work with the data
                if pattern_name in ['inverse', 'logarithmic', 'power'] and np.any(x_data <= 0):
                    continue
                
                # Fit pattern
                if pattern_name == 'linear':
                    popt, _ = curve_fit(pattern_func, x_data, y, p0=[np.mean(y), 1])
                elif pattern_name == 'quadratic':
                    popt, _ = curve_fit(pattern_func, x_data, y, p0=[np.mean(y), 1, 0])
                elif pattern_name == 'exponential':
                    popt, _ = curve_fit(pattern_func, x_data, y, p0=[np.mean(y), 0.1])
                elif pattern_name == 'power':
                    popt, _ = curve_fit(pattern_func, x_data, y, p0=[1, 1])
                elif pattern_name == 'inverse':
                    popt, _ = curve_fit(pattern_func, x_data, y, p0=[np.mean(y) * np.mean(x_data)])
                elif pattern_name == 'logarithmic':
                    popt, _ = curve_fit(pattern_func, x_data, y, p0=[1, np.mean(y)])
                
                # Calculate R² score
                y_pred = pattern_func(x_data, *popt)
                r2 = r2_score(y, y_pred)
                
                if r2 > best_score:
                    best_score = r2
                    best_pattern = (pattern_name, popt)
                    
            except Exception as e:
                continue
        
        self.fitness_score = best_score
        self.best_pattern = best_pattern
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        if self.best_pattern is None:
            return np.zeros(len(X))
        
        pattern_name, params = self.best_pattern
        x_data = X.flatten() if len(X.shape) > 1 else X
        
        try:
            if pattern_name == 'linear':
                return params[0] + params[1] * x_data
            elif pattern_name == 'quadratic':
                return params[0] + params[1] * x_data + params[2] * x_data**2
            elif pattern_name == 'exponential':
                return params[0] * np.exp(params[1] * x_data)
            elif pattern_name == 'power':
                return params[0] * x_data**params[1]
            elif pattern_name == 'inverse':
                return params[0] / x_data
            elif pattern_name == 'logarithmic':
                return params[0] * np.log(x_data) + params[1]
        except:
            return np.zeros(len(x_data))
