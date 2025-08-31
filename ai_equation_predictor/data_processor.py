import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handle data cleaning, preprocessing, and validation"""
    
    def __init__(self):
        self.processed_data = None
        self.data_stats = None
    
    def clean_data(self, df, x_column, y_column):
        """Clean and prepare data for analysis"""
        
        try:
            # Extract relevant columns
            data = df[[x_column, y_column]].copy()
            
            # Convert to numeric, coercing errors to NaN
            data[x_column] = pd.to_numeric(data[x_column], errors='coerce')
            data[y_column] = pd.to_numeric(data[y_column], errors='coerce')
            
            # Remove rows with NaN values
            data = data.dropna()
            
            # Remove infinite values
            data = data[np.isfinite(data[x_column]) & np.isfinite(data[y_column])]
            
            # Remove outliers using IQR method
            data = self._remove_outliers(data, x_column, y_column)
            
            # Sort by x-values for better visualization
            data = data.sort_values(x_column)
            
            # Store statistics
            self.data_stats = self._calculate_stats(data, x_column, y_column)
            
            self.processed_data = data
            return data
            
        except Exception as e:
            print(f"Error in data cleaning: {str(e)}")
            return None
    
    def _remove_outliers(self, data, x_column, y_column, iqr_multiplier=2.0):
        """Remove outliers using Interquartile Range method"""
        
        try:
            # Calculate IQR for both columns
            for column in [x_column, y_column]:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                # Filter outliers
                data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
            
            return data
            
        except Exception as e:
            # If outlier removal fails, return original data
            return data
    
    def _calculate_stats(self, data, x_column, y_column):
        """Calculate descriptive statistics for the data"""
        
        stats_dict = {
            'data_points': len(data),
            'x_stats': {
                'mean': data[x_column].mean(),
                'std': data[x_column].std(),
                'min': data[x_column].min(),
                'max': data[x_column].max(),
                'range': data[x_column].max() - data[x_column].min()
            },
            'y_stats': {
                'mean': data[y_column].mean(),
                'std': data[y_column].std(),
                'min': data[y_column].min(),
                'max': data[y_column].max(),
                'range': data[y_column].max() - data[y_column].min()
            }
        }
        
        # Calculate correlation
        try:
            correlation, p_value = stats.pearsonr(data[x_column], data[y_column])
            stats_dict['correlation'] = {
                'pearson_r': correlation,
                'p_value': p_value,
                'strength': self._interpret_correlation(correlation)
            }
        except:
            stats_dict['correlation'] = {
                'pearson_r': 0,
                'p_value': 1,
                'strength': 'No correlation'
            }
        
        return stats_dict
    
    def _interpret_correlation(self, r):
        """Interpret correlation strength"""
        abs_r = abs(r)
        if abs_r >= 0.9:
            return "Very strong"
        elif abs_r >= 0.7:
            return "Strong"
        elif abs_r >= 0.5:
            return "Moderate"
        elif abs_r >= 0.3:
            return "Weak"
        else:
            return "Very weak"
    
    def detect_data_patterns(self, data, x_column, y_column):
        """Detect basic patterns in the data"""
        
        if data is None or len(data) < 3:
            return {}
        
        x = data[x_column].values
        y = data[y_column].values
        
        patterns = {}
        
        # Check for linearity
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            patterns['linear'] = {
                'r_squared': r_value**2,
                'slope': slope,
                'intercept': intercept,
                'p_value': p_value
            }
        except:
            patterns['linear'] = {'r_squared': 0}
        
        # Check for exponential pattern (log-linear)
        try:
            if np.all(y > 0):
                log_y = np.log(y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_y)
                patterns['exponential'] = {
                    'r_squared': r_value**2,
                    'growth_rate': slope,
                    'p_value': p_value
                }
        except:
            patterns['exponential'] = {'r_squared': 0}
        
        # Check for power law pattern (log-log)
        try:
            if np.all(x > 0) and np.all(y > 0):
                log_x = np.log(x)
                log_y = np.log(y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
                patterns['power_law'] = {
                    'r_squared': r_value**2,
                    'exponent': slope,
                    'p_value': p_value
                }
        except:
            patterns['power_law'] = {'r_squared': 0}
        
        # Check for inverse relationship
        try:
            if np.all(x > 0):
                inv_x = 1 / x
                slope, intercept, r_value, p_value, std_err = stats.linregress(inv_x, y)
                patterns['inverse'] = {
                    'r_squared': r_value**2,
                    'constant': slope,
                    'p_value': p_value
                }
        except:
            patterns['inverse'] = {'r_squared': 0}
        
        # Check for quadratic pattern
        try:
            # Fit quadratic polynomial
            coeffs = np.polyfit(x, y, 2)
            y_pred = np.polyval(coeffs, x)
            r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            patterns['quadratic'] = {
                'r_squared': r_squared,
                'coefficients': coeffs.tolist()
            }
        except:
            patterns['quadratic'] = {'r_squared': 0}
        
        return patterns
    
    def validate_data_quality(self, data, x_column, y_column):
        """Assess data quality and provide recommendations"""
        
        if data is None:
            return {
                'quality_score': 0,
                'issues': ['No valid data available'],
                'recommendations': ['Check data format and ensure numeric values']
            }
        
        issues = []
        recommendations = []
        quality_score = 100
        
        # Check data size
        if len(data) < 10:
            issues.append(f"Limited data points ({len(data)})")
            recommendations.append("Collect more data points for better analysis")
            quality_score -= 20
        
        # Check for constant values
        if data[x_column].std() == 0:
            issues.append("X-values are constant")
            recommendations.append("Ensure X-variable has variation")
            quality_score -= 50
        
        if data[y_column].std() == 0:
            issues.append("Y-values are constant")
            recommendations.append("Check if Y-variable measurement is working correctly")
            quality_score -= 50
        
        # Check data range
        x_range = data[x_column].max() - data[x_column].min()
        y_range = data[y_column].max() - data[y_column].min()
        
        if x_range < 1e-10:
            issues.append("Very small X-range")
            recommendations.append("Increase the range of X-variable measurements")
            quality_score -= 15
        
        if y_range < 1e-10:
            issues.append("Very small Y-range")
            recommendations.append("Check measurement precision for Y-variable")
            quality_score -= 15
        
        # Check for non-monotonic but highly correlated data
        if self.data_stats and 'correlation' in self.data_stats:
            correlation = abs(self.data_stats['correlation']['pearson_r'])
            if correlation < 0.3:
                issues.append("Weak correlation between variables")
                recommendations.append("Verify that variables are related or check for measurement errors")
                quality_score -= 10
        
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'recommendations': recommendations,
            'data_stats': self.data_stats
        }
    
    def get_data_summary(self):
        """Get summary of processed data"""
        
        if self.processed_data is None or self.data_stats is None:
            return "No data processed yet."
        
        stats = self.data_stats
        summary = f"""
        Data Summary:
        - Data points: {stats['data_points']}
        - X variable range: {stats['x_stats']['min']:.3f} to {stats['x_stats']['max']:.3f}
        - Y variable range: {stats['y_stats']['min']:.3f} to {stats['y_stats']['max']:.3f}
        - Correlation: {stats['correlation']['pearson_r']:.3f} ({stats['correlation']['strength']})
        """
        
        return summary.strip()
