@echo off
ECHO "======================================="
ECHO "Step 1: Building the ML model..."
ECHO "======================================="
python build_model.py

ECHO "======================================="
ECHO "Step 2: Launching the Streamlit App..."
ECHO "======================================="
streamlit run app.py

pause
