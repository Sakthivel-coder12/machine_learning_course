import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class PhysicsEquationMatcher:
    """Pattern matching against known physics equations"""
    
    def __init__(self):
        self.equation_templates = self._initialize_equations()
    
    def _initialize_equations(self):
        """Initialize library of known physics equations"""
        
        equations = {
            # Kinematics
            'linear_motion': {
                'name': 'Linear Motion (v = v₀ + at)',
                'equation': 'y = a + b*x',
                'func': lambda x, a, b: a + b * x,
                'domain': 'kinematics',
                'description': 'Constant acceleration'
            },
            'quadratic_motion': {
                'name': 'Quadratic Motion (s = ut + ½at²)',
                'equation': 'y = a*x + b*x²',
                'func': lambda x, a, b: a * x + b * x**2,
                'domain': 'kinematics',
                'description': 'Position with constant acceleration'
            },
            'free_fall': {
                'name': 'Free Fall (d = ½gt²)',
                'equation': 'y = a*x²',
                'func': lambda x, a: a * x**2,
                'domain': 'kinematics',
                'description': 'Free falling object'
            },
            
            # Electromagnetism
            'ohms_law': {
                'name': "Ohm's Law (V = IR)",
                'equation': 'y = a*x',
                'func': lambda x, a: a * x,
                'domain': 'electromagnetism',
                'description': 'Linear voltage-current relationship'
            },
            'power_law': {
                'name': 'Power Law (P = I²R)',
                'equation': 'y = a*x²',
                'func': lambda x, a: a * x**2,
                'domain': 'electromagnetism',
                'description': 'Quadratic power relationship'
            },
            'capacitor_discharge': {
                'name': 'Capacitor Discharge (V = V₀e^(-t/RC))',
                'equation': 'y = a*exp(-b*x)',
                'func': lambda x, a, b: a * np.exp(-b * x),
                'domain': 'electromagnetism',
                'description': 'Exponential decay'
            },
            
            # Thermodynamics
            'boyles_law': {
                'name': "Boyle's Law (PV = constant)",
                'equation': 'y = a/x',
                'func': lambda x, a: a / x,
                'domain': 'thermodynamics',
                'description': 'Inverse pressure-volume relationship'
            },
            'charles_law': {
                'name': "Charles's Law (V ∝ T)",
                'equation': 'y = a*x',
                'func': lambda x, a: a * x,
                'domain': 'thermodynamics',
                'description': 'Linear volume-temperature relationship'
            },
            'ideal_gas_isothermal': {
                'name': 'Ideal Gas (Isothermal)',
                'equation': 'y = a/x',
                'func': lambda x, a: a / x,
                'domain': 'thermodynamics',
                'description': 'Inverse relationship at constant temperature'
            },
            
            # Wave Physics and Oscillations
            'simple_harmonic': {
                'name': 'Simple Harmonic Motion',
                'equation': 'y = a*sin(b*x + c)',
                'func': lambda x, a, b, c: a * np.sin(b * x + c),
                'domain': 'waves',
                'description': 'Sinusoidal oscillation'
            },
            'pendulum_period': {
                'name': 'Pendulum Period (T = 2π√(L/g))',
                'equation': 'y = a*sqrt(x)',
                'func': lambda x, a: a * np.sqrt(x),
                'domain': 'mechanics',
                'description': 'Square root relationship'
            },
            'wave_speed': {
                'name': 'Wave Speed (v = fλ)',
                'equation': 'y = a*x',
                'func': lambda x, a: a * x,
                'domain': 'waves',
                'description': 'Linear frequency-wavelength relationship'
            },
            
            # Energy and Work
            'kinetic_energy': {
                'name': 'Kinetic Energy (KE = ½mv²)',
                'equation': 'y = a*x²',
                'func': lambda x, a: a * x**2,
                'domain': 'mechanics',
                'description': 'Quadratic velocity relationship'
            },
            'potential_energy': {
                'name': 'Gravitational PE (PE = mgh)',
                'equation': 'y = a*x',
                'func': lambda x, a: a * x,
                'domain': 'mechanics',
                'description': 'Linear height relationship'
            },
            'spring_force': {
                'name': "Hooke's Law (F = kx)",
                'equation': 'y = a*x',
                'func': lambda x, a: a * x,
                'domain': 'mechanics',
                'description': 'Linear spring force'
            },
            
            # Optics
            'snells_law': {
                'name': "Snell's Law (n₁sin θ₁ = n₂sin θ₂)",
                'equation': 'y = a*sin(x)',
                'func': lambda x, a: a * np.sin(x),
                'domain': 'optics',
                'description': 'Sine relationship for refraction'
            },
            'thin_lens': {
                'name': 'Thin Lens (1/f = 1/u + 1/v)',
                'equation': '1/y = a + b/x',
                'func': lambda x, a, b: 1 / (a + b / x),
                'domain': 'optics',
                'description': 'Reciprocal lens equation'
            },
            
            # Exponential and Logarithmic
            'exponential_growth': {
                'name': 'Exponential Growth (N = N₀e^(rt))',
                'equation': 'y = a*exp(b*x)',
                'func': lambda x, a, b: a * np.exp(b * x),
                'domain': 'general',
                'description': 'Exponential growth'
            },
            'exponential_decay': {
                'name': 'Exponential Decay (N = N₀e^(-λt))',
                'equation': 'y = a*exp(-b*x)',
                'func': lambda x, a, b: a * np.exp(-b * x),
                'domain': 'general',
                'description': 'Exponential decay'
            },
            'logarithmic': {
                'name': 'Logarithmic Relationship',
                'equation': 'y = a*log(x) + b',
                'func': lambda x, a, b: a * np.log(x) + b,
                'domain': 'general',
                'description': 'Logarithmic relationship'
            }
        }
        
        return equations
    
    def find_best_matches(self, x, y, max_matches=5):
        """Find best matching equations for the given data"""
        
        matches = []
        
        # Ensure data is valid
        if len(x) < 3 or len(y) < 3:
            return matches
        
        # Remove any infinite or NaN values
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) < 3:
            return matches
        
        for eq_name, eq_info in self.equation_templates.items():
            try:
                # Skip equations that require positive x values if we have negative/zero
                if eq_name in ['boyles_law', 'ideal_gas_isothermal', 'logarithmic', 'pendulum_period']:
                    if np.any(x_clean <= 0):
                        continue
                
                # Skip sine functions if x values are too large (likely not angles)
                if eq_name in ['snells_law', 'simple_harmonic']:
                    if np.max(np.abs(x_clean)) > 2 * np.pi:
                        continue
                
                # Try to fit the equation
                r2_score, fitted_params, confidence = self._fit_equation(x_clean, y_clean, eq_info)
                
                if r2_score > 0.1:  # Minimum threshold for consideration
                    matches.append({
                        'name': eq_info['name'],
                        'equation': eq_info['equation'],
                        'domain': eq_info['domain'],
                        'description': eq_info['description'],
                        'r2_score': r2_score,
                        'confidence': confidence,
                        'fitted_params': fitted_params,
                        'equation_key': eq_name
                    })
            
            except Exception as e:
                # Skip equations that fail to fit
                continue
        
        # Sort by R² score and return top matches
        matches.sort(key=lambda x: x['r2_score'], reverse=True)
        return matches[:max_matches]
    
    def _fit_equation(self, x, y, eq_info):
        """Fit an equation to the data and return R² score"""
        
        try:
            # Get the function
            func = eq_info['func']
            
            # Determine initial parameters based on equation type
            if eq_info['equation'] == 'y = a*x':  # Linear
                p0 = [1.0]
            elif eq_info['equation'] == 'y = a + b*x':  # Linear with intercept
                p0 = [np.mean(y), 1.0]
            elif eq_info['equation'] == 'y = a*x²':  # Quadratic through origin
                p0 = [1.0]
            elif eq_info['equation'] == 'y = a*x + b*x²':  # Quadratic
                p0 = [1.0, 1.0]
            elif eq_info['equation'] == 'y = a/x':  # Inverse
                p0 = [np.mean(y) * np.mean(x)]
            elif eq_info['equation'] == 'y = a*exp(b*x)':  # Exponential growth
                p0 = [np.mean(y), 0.1]
            elif eq_info['equation'] == 'y = a*exp(-b*x)':  # Exponential decay
                p0 = [np.max(y), 0.1]
            elif eq_info['equation'] == 'y = a*sqrt(x)':  # Square root
                p0 = [1.0]
            elif eq_info['equation'] == 'y = a*log(x) + b':  # Logarithmic
                p0 = [1.0, np.mean(y)]
            elif eq_info['equation'] == 'y = a*sin(b*x + c)':  # Sinusoidal
                p0 = [np.std(y), 1.0, 0.0]
            elif eq_info['equation'] == 'y = a*sin(x)':  # Simple sine
                p0 = [np.std(y)]
            elif eq_info['equation'] == '1/y = a + b/x':  # Reciprocal
                p0 = [1.0, 1.0]
            else:
                p0 = [1.0] * (func.__code__.co_argcount - 1)
            
            # Fit the curve
            popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=1000)
            
            # Calculate R² score
            y_pred = func(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculate confidence based on parameter errors and R²
            param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else np.ones_like(popt)
            relative_errors = np.abs(param_errors / (popt + 1e-10))
            avg_relative_error = np.mean(relative_errors)
            
            # Confidence combines R² score and parameter certainty
            confidence = r2_score * (1 - min(avg_relative_error, 0.5))
            
            return max(r2_score, 0), popt, max(confidence, 0)
            
        except Exception as e:
            return 0, None, 0
    
    def predict(self, match_info, x_values):
        """Generate predictions using a fitted equation"""
        
        try:
            eq_key = match_info['equation_key']
            eq_info = self.equation_templates[eq_key]
            func = eq_info['func']
            params = match_info['fitted_params']
            
            return func(x_values, *params)
            
        except Exception as e:
            return np.full_like(x_values, np.nan)
    
    def get_equation_info(self):
        """Return information about all available equations"""
        
        info = []
        for eq_name, eq_data in self.equation_templates.items():
            info.append({
                'name': eq_data['name'],
                'equation': eq_data['equation'],
                'domain': eq_data['domain'],
                'description': eq_data['description']
            })
        
        return sorted(info, key=lambda x: x['domain'])
