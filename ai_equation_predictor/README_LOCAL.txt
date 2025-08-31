AI Equation Mapper - Local Setup Instructions
===============================================

This is an AI-powered application that analyzes experimental data to predict physical equations.

SETUP INSTRUCTIONS:
==================

1. Install Python 3.8+ on your machine

2. Install required packages:
   pip install -r dependencies.txt
   
   Or install manually:
   pip install streamlit pandas numpy plotly scipy scikit-learn sympy opencv-python pillow pytesseract openpyxl matplotlib

3. Run the application:
   streamlit run app.py

4. Open your web browser and go to:
   http://localhost:8501

FEATURES:
=========
- Upload CSV, Excel, JSON, and text files
- Extract data from graph images using computer vision
- Manual data entry with individual points, bulk entry, and function generator
- AI-powered equation discovery using symbolic regression
- Pattern matching against known physics equations
- Interactive data visualization and analysis

SAMPLE DATASETS:
===============
The sample_datasets/ folder contains physics datasets for testing:
- Free Fall Motion (kinematics)
- Ohm's Law (electromagnetism)
- Ideal Gas Law (thermodynamics)
- Simple Pendulum (mechanics)

BACKUP:
=======
The backup_v1/ folder contains the original version for reference.

TROUBLESHOOTING:
===============
- If you get import errors, make sure all dependencies are installed
- For image processing issues, install tesseract-ocr on your system
- On macOS: brew install tesseract
- On Ubuntu: sudo apt-get install tesseract-ocr
- On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

For any issues, check the console output for error messages.