import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile
from physics_equations import PhysicsEquationMatcher
from symbolic_regression import SymbolicRegressionEngine
from data_processor import DataProcessor

# Set page config
st.set_page_config(
    page_title="AI Equation Mapper",
    page_icon="🧪",
    layout="wide"
)

def main():
    st.title("🧪 AI-Powered Equation Mapper from Experimental Data")
    st.markdown("""
    Upload your experimental data and let AI predict the most likely physical equations that explain your results.
    This tool uses symbolic regression and pattern matching against known physics laws.
    """)
    
    # Initialize components
    if 'equation_matcher' not in st.session_state:
        st.session_state.equation_matcher = PhysicsEquationMatcher()
    if 'symbolic_engine' not in st.session_state:
        st.session_state.symbolic_engine = SymbolicRegressionEngine()
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Choose Section", 
                          ["Upload Data", "Sample Datasets", "Analysis Results", "Dataset Sources"])
    
    if tab == "Upload Data":
        upload_data_section()
    elif tab == "Sample Datasets":
        sample_datasets_section()
    elif tab == "Analysis Results":
        analysis_results_section()
    elif tab == "Dataset Sources":
        dataset_sources_section()

def upload_data_section():
    st.header("📁 Upload Your Experimental Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing your experimental data",
        type=['csv'],
        help="Upload a CSV file with numerical columns representing your experimental measurements."
    )
    
    if uploaded_file is not None:
        try:
            # Read and process data
            df = pd.read_csv(uploaded_file)
            st.session_state.current_data = df
            
            st.success(f"✅ Data uploaded successfully! Shape: {df.shape}")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Data info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
            
            with col2:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                st.metric("Numeric Columns", len(numeric_cols))
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Column selection for analysis
            st.subheader("🎯 Select Variables for Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox(
                    "Independent Variable (X-axis)", 
                    numeric_cols,
                    help="Select the variable you controlled or measured independently"
                )
            
            with col2:
                y_column = st.selectbox(
                    "Dependent Variable (Y-axis)", 
                    [col for col in numeric_cols if col != x_column],
                    help="Select the variable that responds to changes in the independent variable"
                )
            
            if st.button("🔬 Analyze Data", type="primary"):
                analyze_data(df, x_column, y_column)
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure your CSV file has proper formatting with numeric data.")

def sample_datasets_section():
    st.header("📊 Sample Physics Datasets")
    st.markdown("Try the analysis with these pre-loaded physics datasets:")
    
    datasets = {
        "Free Fall Motion": "sample_datasets/kinematics_free_fall.csv",
        "Ohm's Law": "sample_datasets/ohms_law.csv", 
        "Ideal Gas Law": "sample_datasets/ideal_gas.csv",
        "Simple Pendulum": "sample_datasets/pendulum.csv"
    }
    
    selected_dataset = st.selectbox("Choose a sample dataset:", list(datasets.keys()))
    
    if st.button("Load Sample Dataset"):
        try:
            df = pd.read_csv(datasets[selected_dataset])
            st.session_state.current_data = df
            st.success(f"✅ Loaded {selected_dataset} dataset!")
            
            # Display dataset info
            st.subheader("Dataset Information")
            x_col, y_col = "", ""  # Initialize variables
            if selected_dataset == "Free Fall Motion":
                st.info("**Physics**: Distance vs Time for free falling object (d = ½gt²)")
                x_col, y_col = "time_s", "distance_m"
            elif selected_dataset == "Ohm's Law":
                st.info("**Physics**: Current vs Voltage relationship (V = IR)")
                x_col, y_col = "current_A", "voltage_V"
            elif selected_dataset == "Ideal Gas Law":
                st.info("**Physics**: Pressure vs Volume at constant temperature (PV = constant)")
                x_col, y_col = "volume_L", "pressure_atm"
            elif selected_dataset == "Simple Pendulum":
                st.info("**Physics**: Period vs Length of pendulum (T = 2π√(L/g))")
                x_col, y_col = "length_m", "period_s"
            
            st.dataframe(df)
            
            if st.button("🔬 Analyze This Dataset", type="primary"):
                analyze_data(df, x_col, y_col)
                
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")

def analyze_data(df, x_column, y_column):
    """Perform comprehensive analysis on the selected data"""
    
    with st.spinner("🤖 AI is analyzing your data..."):
        try:
            # Clean and prepare data
            clean_data = st.session_state.data_processor.clean_data(df, x_column, y_column)
            
            if clean_data is None or len(clean_data) < 3:
                st.error("❌ Insufficient valid data points for analysis.")
                return
            
            X = clean_data[x_column].values.reshape(-1, 1)
            y = clean_data[y_column].values
            
            # Store analysis results
            st.session_state.analysis_results = {
                'data': clean_data,
                'x_column': x_column,
                'y_column': y_column,
                'X': X,
                'y': y
            }
            
            # Pattern matching against known equations
            st.subheader("🔍 Pattern Matching Results")
            equation_matches = st.session_state.equation_matcher.find_best_matches(X.flatten(), y)
            
            if equation_matches:
                for i, match in enumerate(equation_matches[:3]):  # Top 3 matches
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{match['name']}**: {match['equation']}")
                    with col2:
                        st.metric("R² Score", f"{match['r2_score']:.4f}")
                    with col3:
                        st.metric("Confidence", f"{match['confidence']:.1%}")
            
            # Symbolic regression
            st.subheader("🧠 AI-Discovered Equations")
            symbolic_results = st.session_state.symbolic_engine.find_equations(X.flatten(), y)
            
            if symbolic_results:
                for i, result in enumerate(symbolic_results[:3]):  # Top 3 results
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**Equation {i+1}**: {result['equation']}")
                    with col2:
                        st.metric("Fitness", f"{result['fitness']:.4f}")
                    with col3:
                        st.metric("Complexity", result['complexity'])
            
            # Store complete results
            st.session_state.analysis_results.update({
                'equation_matches': equation_matches,
                'symbolic_results': symbolic_results
            })
            
            st.success("✅ Analysis complete! Check the 'Analysis Results' tab for detailed visualizations.")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

def analysis_results_section():
    st.header("📈 Analysis Results")
    
    if 'analysis_results' not in st.session_state:
        st.info("📝 No analysis results available. Please upload and analyze data first.")
        return
    
    results = st.session_state.analysis_results
    data = results['data']
    x_column = results['x_column']
    y_column = results['y_column']
    X = results['X']
    y = results['y']
    
    # Create interactive plots
    st.subheader("📊 Data Visualization and Model Fits")
    
    # Main scatter plot with model fits
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Original Data with Best Fits', 'Residuals Analysis', 
                       'Equation Comparison', 'Model Statistics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "table"}]]
    )
    
    # Original data
    fig.add_trace(
        go.Scatter(x=X.flatten(), y=y, mode='markers', name='Experimental Data',
                  marker=dict(color='blue', size=8)), row=1, col=1
    )
    
    # Add best fit lines
    x_smooth = np.linspace(X.min(), X.max(), 100)
    
    # Pattern matching fits
    if 'equation_matches' in results and results['equation_matches']:
        best_match = results['equation_matches'][0]
        y_pred = st.session_state.equation_matcher.predict(best_match, x_smooth)
        fig.add_trace(
            go.Scatter(x=x_smooth, y=y_pred, mode='lines', 
                      name=f"Best Match: {best_match['name']}", 
                      line=dict(color='red', width=3)), row=1, col=1
        )
        
        # Residuals
        y_pred_orig = st.session_state.equation_matcher.predict(best_match, X.flatten())
        residuals = y - y_pred_orig
        fig.add_trace(
            go.Scatter(x=X.flatten(), y=residuals, mode='markers', 
                      name='Residuals', marker=dict(color='red')), row=1, col=2
        )
    
    # Symbolic regression fits
    if 'symbolic_results' in results and results['symbolic_results']:
        best_symbolic = results['symbolic_results'][0]
        try:
            y_symbolic = st.session_state.symbolic_engine.predict(best_symbolic, x_smooth)
            fig.add_trace(
                go.Scatter(x=x_smooth, y=y_symbolic, mode='lines', 
                          name=f"AI Discovery", 
                          line=dict(color='green', width=2, dash='dash')), row=1, col=1
            )
        except:
            pass
    
    # Update layout
    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text=x_column, row=1, col=1)
    fig.update_yaxes(title_text=y_column, row=1, col=1)
    fig.update_xaxes(title_text=x_column, row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Pattern Matching Results")
        if 'equation_matches' in results and results['equation_matches']:
            df_matches = pd.DataFrame(results['equation_matches'])
            st.dataframe(df_matches[['name', 'equation', 'r2_score', 'confidence']])
        else:
            st.info("No pattern matches found.")
    
    with col2:
        st.subheader("🧠 Symbolic Regression Results")
        if 'symbolic_results' in results and results['symbolic_results']:
            df_symbolic = pd.DataFrame(results['symbolic_results'])
            st.dataframe(df_symbolic[['equation', 'fitness', 'complexity']])
        else:
            st.info("No symbolic equations discovered.")
    
    # Export functionality
    st.subheader("💾 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Results as CSV"):
            export_results_csv(results)
    
    with col2:
        if st.button("📊 Export Plots as HTML"):
            export_plots_html(fig)
    
    with col3:
        if st.button("📋 Export Summary Report"):
            export_summary_report(results)

def export_results_csv(results):
    """Export analysis results as CSV"""
    try:
        # Create results summary
        summary_data = []
        
        if 'equation_matches' in results:
            for match in results['equation_matches']:
                summary_data.append({
                    'Type': 'Pattern Match',
                    'Name': match['name'],
                    'Equation': match['equation'],
                    'Score': match['r2_score'],
                    'Confidence': match['confidence']
                })
        
        if 'symbolic_results' in results:
            for symbolic in results['symbolic_results']:
                summary_data.append({
                    'Type': 'Symbolic Regression',
                    'Name': f"AI Discovery",
                    'Equation': symbolic['equation'],
                    'Score': symbolic['fitness'],
                    'Confidence': 1.0 - symbolic['complexity'] / 100
                })
        
        df_export = pd.DataFrame(summary_data)
        csv = df_export.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="equation_analysis_results.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def export_plots_html(fig):
    """Export plots as HTML"""
    try:
        html_str = fig.to_html()
        st.download_button(
            label="📥 Download HTML Plot",
            data=html_str,
            file_name="analysis_plots.html",
            mime="text/html"
        )
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def export_summary_report(results):
    """Export a comprehensive summary report"""
    try:
        report = f"""
# AI Equation Mapper - Analysis Report

## Data Summary
- Variables: {results['x_column']} vs {results['y_column']}
- Data Points: {len(results['data'])}

## Pattern Matching Results
"""
        
        if 'equation_matches' in results:
            for i, match in enumerate(results['equation_matches'][:3]):
                report += f"""
### {i+1}. {match['name']}
- Equation: {match['equation']}
- R² Score: {match['r2_score']:.4f}
- Confidence: {match['confidence']:.1%}
"""
        
        report += "\n## AI-Discovered Equations\n"
        
        if 'symbolic_results' in results:
            for i, symbolic in enumerate(results['symbolic_results'][:3]):
                report += f"""
### {i+1}. Discovered Equation
- Expression: {symbolic['equation']}
- Fitness: {symbolic['fitness']:.4f}
- Complexity: {symbolic['complexity']}
"""
        
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="analysis_report.md",
            mime="text/markdown"
        )
        
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def dataset_sources_section():
    st.header("📚 Dataset Sources and References")
    
    st.markdown("""
    ## Available Physics Datasets
    
    This application includes sample datasets representing common physics phenomena:
    
    ### 🚀 Kinematics - Free Fall
    - **Description**: Distance vs time measurements for free-falling objects
    - **Expected Equation**: d = ½gt² (where g ≈ 9.81 m/s²)
    - **Variables**: time (s), distance (m)
    
    ### ⚡ Electromagnetism - Ohm's Law
    - **Description**: Current vs voltage measurements in resistive circuits
    - **Expected Equation**: V = IR (Ohm's Law)
    - **Variables**: current (A), voltage (V)
    
    ### 🌡️ Thermodynamics - Ideal Gas
    - **Description**: Pressure vs volume at constant temperature
    - **Expected Equation**: PV = constant (Boyle's Law)
    - **Variables**: volume (L), pressure (atm)
    
    ### 🎯 Mechanics - Simple Pendulum
    - **Description**: Period vs length measurements for simple pendulum
    - **Expected Equation**: T = 2π√(L/g)
    - **Variables**: length (m), period (s)
    
    ## External Dataset Sources
    
    For real experimental data, consider these sources:
    
    ### 🌌 NASA Open Data Portal
    - **URL**: https://data.nasa.gov/
    - **Content**: Atmospheric data, planetary motion, space weather
    - **Format**: CSV, JSON, various formats
    
    ### 🔬 CERN Open Data Portal
    - **URL**: http://opendata.cern.ch/
    - **Content**: Particle physics experimental data
    - **Format**: ROOT files, CSV exports available
    
    ### 📊 Physics Dataset Repositories
    - **UC Irvine ML Repository**: https://archive.ics.uci.edu/ml/
    - **Kaggle Physics Datasets**: https://www.kaggle.com/datasets?search=physics
    - **GitHub Physics Data**: Search for "physics-datasets" repositories
    
    ### 🎓 Educational Physics Data
    - **PhET Simulations**: https://phet.colorado.edu/ (simulation data exports)
    - **NIST Physical Constants**: https://physics.nist.gov/cuu/Constants/
    - **HyperPhysics Database**: http://hyperphysics.phy-astr.gsu.edu/
    
    ## Data Format Requirements
    
    For best results, your CSV files should:
    - Have clear column headers
    - Contain only numeric data in analysis columns
    - Include at least 10-20 data points
    - Have minimal missing values
    - Represent a clear relationship between variables
    
    ## Supported Physics Domains
    
    The AI equation mapper is optimized for:
    - **Classical Mechanics**: kinematics, dynamics, oscillations
    - **Thermodynamics**: gas laws, heat transfer, phase transitions
    - **Electromagnetism**: Ohm's law, capacitance, inductance
    - **Optics**: Snell's law, thin lens equations
    - **Wave Physics**: frequency, wavelength, amplitude relationships
    """)

if __name__ == "__main__":
    main()
