import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import io
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class ImageDataExtractor:
    """Extract data points from graph images using computer vision"""
    
    def __init__(self):
        self.image = None
        self.processed_image = None
        self.data_points = []
        
    def process_image(self, uploaded_image):
        """Process uploaded image and extract data points"""
        try:
            # Convert PIL image to OpenCV format
            image = Image.open(uploaded_image)
            img_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            self.image = img_array
            return True
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return False
    
    def extract_graph_data(self, color_threshold=50, min_points=5):
        """Extract data points from graph image"""
        if self.image is None:
            return None
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to find dark pixels (data points)
            _, binary = cv2.threshold(gray, color_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract points from contours
            points = []
            for contour in contours:
                # Get contour area
                area = cv2.contourArea(contour)
                if area > 5:  # Filter small noise
                    # Get center of contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        points.append((cx, cy))
            
            if len(points) >= min_points:
                # Sort points by x-coordinate
                points.sort(key=lambda p: p[0])
                
                # Convert image coordinates to data coordinates
                self.data_points = self._normalize_coordinates(points)
                return self.data_points
            else:
                return None
                
        except Exception as e:
            st.error(f"Error extracting data: {str(e)}")
            return None
    
    def _normalize_coordinates(self, points):
        """Normalize image coordinates to data coordinates"""
        if not points:
            return []
        
        # Get image dimensions
        height, width = self.image.shape[:2]
        
        # Normalize coordinates (assuming standard graph orientation)
        normalized_points = []
        for x, y in points:
            # Normalize x from 0 to 1
            norm_x = x / width
            # Normalize y from 0 to 1 (flip y-axis for typical graph orientation)
            norm_y = (height - y) / height
            normalized_points.append((norm_x, norm_y))
        
        return normalized_points
    
    def scale_data(self, x_min, x_max, y_min, y_max):
        """Scale normalized coordinates to actual data range"""
        if not self.data_points:
            return None
        
        scaled_points = []
        for norm_x, norm_y in self.data_points:
            # Scale to actual data range
            actual_x = x_min + norm_x * (x_max - x_min)
            actual_y = y_min + norm_y * (y_max - y_min)
            scaled_points.append((actual_x, actual_y))
        
        return scaled_points
    
    def create_dataframe(self, scaled_points, x_label="x", y_label="y"):
        """Create pandas DataFrame from extracted points"""
        if not scaled_points:
            return None
        
        x_values = [point[0] for point in scaled_points]
        y_values = [point[1] for point in scaled_points]
        
        df = pd.DataFrame({
            x_label: x_values,
            y_label: y_values
        })
        
        return df
    
    def preview_extraction(self):
        """Create preview of extracted points on original image"""
        if self.image is None or not self.data_points:
            return None
        
        # Create copy of original image
        preview_img = self.image.copy()
        
        # Convert normalized points back to image coordinates
        height, width = self.image.shape[:2]
        
        for norm_x, norm_y in self.data_points:
            # Convert back to image coordinates
            x = int(norm_x * width)
            y = int((1 - norm_y) * height)  # Flip y-axis back
            
            # Draw red circle at detected point
            cv2.circle(preview_img, (x, y), 5, (0, 0, 255), -1)
        
        # Convert BGR to RGB for display
        preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
        return preview_rgb

class AdvancedDataProcessor:
    """Advanced data processing with multiple input formats"""
    
    def __init__(self):
        self.supported_formats = {
            'csv': self._read_csv,
            'xlsx': self._read_excel,
            'xls': self._read_excel,
            'json': self._read_json,
            'txt': self._read_text
        }
    
    def process_file(self, uploaded_file):
        """Process different file formats"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in self.supported_formats:
                return self.supported_formats[file_extension](uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None
    
    def _read_csv(self, file):
        """Read CSV file"""
        return pd.read_csv(file)
    
    def _read_excel(self, file):
        """Read Excel file"""
        try:
            # Try to read all sheets and let user choose
            excel_file = pd.ExcelFile(file)
            
            if len(excel_file.sheet_names) == 1:
                return pd.read_excel(file)
            else:
                # For multiple sheets, default to first sheet
                # In a real app, you'd want to let user choose
                return pd.read_excel(file, sheet_name=0)
                
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return None
    
    def _read_json(self, file):
        """Read JSON file"""
        try:
            import json
            content = json.load(file)
            
            # Try to convert to DataFrame
            if isinstance(content, list):
                return pd.DataFrame(content)
            elif isinstance(content, dict):
                return pd.DataFrame([content])
            else:
                st.error("JSON format not suitable for data analysis")
                return None
                
        except Exception as e:
            st.error(f"Error reading JSON file: {str(e)}")
            return None
    
    def _read_text(self, file):
        """Read text file (tab or space separated)"""
        try:
            # Try different separators
            content = file.read().decode('utf-8')
            lines = content.strip().split('\n')
            
            # Detect separator
            first_line = lines[0]
            if '\t' in first_line:
                separator = '\t'
            elif ',' in first_line:
                separator = ','
            else:
                separator = None  # Space separated
            
            # Read as CSV with detected separator
            from io import StringIO
            if separator:
                return pd.read_csv(StringIO(content), sep=separator)
            else:
                return pd.read_csv(StringIO(content), delim_whitespace=True)
                
        except Exception as e:
            st.error(f"Error reading text file: {str(e)}")
            return None

def create_manual_entry_interface():
    """Create interface for manual data entry"""
    st.subheader("📝 Manual Data Entry")
    
    # Choose data entry method
    entry_method = st.radio(
        "Choose entry method:",
        ["Individual Points", "Bulk Entry", "Function Generator"]
    )
    
    if entry_method == "Individual Points":
        return _individual_point_entry()
    elif entry_method == "Bulk Entry":
        return _bulk_entry()
    else:
        return _function_generator()

def _individual_point_entry():
    """Interface for entering individual data points"""
    if 'manual_points' not in st.session_state:
        st.session_state.manual_points = []
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        x_val = st.number_input("X value:", key="manual_x")
    with col2:
        y_val = st.number_input("Y value:", key="manual_y")
    with col3:
        if st.button("Add Point"):
            st.session_state.manual_points.append((x_val, y_val))
            st.success(f"Added point ({x_val}, {y_val})")
    
    # Display current points
    if st.session_state.manual_points:
        st.write("Current points:")
        points_df = pd.DataFrame(st.session_state.manual_points, columns=['X', 'Y'])
        st.dataframe(points_df)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear All Points"):
                st.session_state.manual_points = []
                st.rerun()
        
        with col2:
            if st.button("Use These Points"):
                return points_df
    
    return None

def _bulk_entry():
    """Interface for bulk data entry"""
    st.write("Enter data in CSV format (x,y pairs):")
    bulk_text = st.text_area(
        "Data (one point per line):",
        placeholder="1.0,2.5\n2.0,5.1\n3.0,7.8\n...",
        height=200
    )
    
    if st.button("Parse Data") and bulk_text:
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(bulk_text), header=None, names=['X', 'Y'])
            st.success(f"Parsed {len(df)} data points")
            st.dataframe(df)
            return df
        except Exception as e:
            st.error(f"Error parsing data: {str(e)}")
    
    return None

def _function_generator():
    """Generate data from mathematical functions"""
    st.write("Generate synthetic data for testing:")
    
    col1, col2 = st.columns(2)
    with col1:
        func_type = st.selectbox(
            "Function type:",
            ["Linear", "Quadratic", "Exponential", "Sinusoidal", "Custom"]
        )
        
        x_min = st.number_input("X min:", value=0.0)
        x_max = st.number_input("X max:", value=10.0)
        num_points = st.slider("Number of points:", 10, 100, 50)
    
    with col2:
        if func_type == "Linear":
            slope = st.number_input("Slope:", value=1.0)
            intercept = st.number_input("Intercept:", value=0.0)
            noise = st.slider("Noise level:", 0.0, 1.0, 0.1)
        elif func_type == "Quadratic":
            a = st.number_input("a (x² coefficient):", value=1.0)
            b = st.number_input("b (x coefficient):", value=0.0)
            c = st.number_input("c (constant):", value=0.0)
            noise = st.slider("Noise level:", 0.0, 1.0, 0.1)
        elif func_type == "Exponential":
            base = st.number_input("Base multiplier:", value=1.0)
            rate = st.number_input("Growth rate:", value=0.5)
            noise = st.slider("Noise level:", 0.0, 1.0, 0.1)
        elif func_type == "Sinusoidal":
            amplitude = st.number_input("Amplitude:", value=1.0)
            frequency = st.number_input("Frequency:", value=1.0)
            phase = st.number_input("Phase:", value=0.0)
            noise = st.slider("Noise level:", 0.0, 1.0, 0.1)
        else:  # Custom
            custom_func = st.text_input(
                "Custom function (use 'x' as variable):",
                placeholder="x**2 + 2*x + 1"
            )
            noise = st.slider("Noise level:", 0.0, 1.0, 0.1)
    
    if st.button("Generate Data"):
        x = np.linspace(x_min, x_max, num_points)
        
        try:
            if func_type == "Linear":
                y = slope * x + intercept
            elif func_type == "Quadratic":
                y = a * x**2 + b * x + c
            elif func_type == "Exponential":
                y = base * np.exp(rate * x)
            elif func_type == "Sinusoidal":
                y = amplitude * np.sin(frequency * x + phase)
            else:  # Custom
                # Safe evaluation of custom function
                allowed_names = {
                    "x": x,
                    "np": np,
                    "sin": np.sin,
                    "cos": np.cos,
                    "exp": np.exp,
                    "log": np.log,
                    "sqrt": np.sqrt,
                    "pi": np.pi,
                    "e": np.e
                }
                y = eval(custom_func, {"__builtins__": {}}, allowed_names)
            
            # Add noise
            if noise > 0:
                y += np.random.normal(0, noise * np.std(y), len(y))
            
            df = pd.DataFrame({'X': x, 'Y': y})
            st.success(f"Generated {len(df)} data points")
            
            # Show preview plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, 'bo-', markersize=4)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Generated {func_type} Data')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            return df
            
        except Exception as e:
            st.error(f"Error generating data: {str(e)}")
    
    return None