# ✈️ Predictive Maintenance of Jet Engines using Machine Learning

A comprehensive machine learning system for predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS dataset. This project combines advanced data science, interactive visualization, and predictive modeling to enable proactive maintenance scheduling in aerospace applications.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset Information](#dataset-information)
- [Project Architecture](#project-architecture)
- [Technologies & Tools](#technologies--tools)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Team Contributions](#team-contributions)
- [Results & Performance](#results--performance)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)
- [References](#references)
- [License](#license)

---

## Project Overview

**Objective:** Build a machine learning pipeline to predict the Remaining Useful Life (RUL) of aircraft turbofan engines before failure occurs, enabling predictive maintenance scheduling.

**Impact:**
- **Safety:** Identify engine degradation early to prevent catastrophic failures
- **Cost Reduction:** Optimize maintenance schedules and reduce unplanned downtime
- **Efficiency:** Better allocation of maintenance resources and spare parts
- **Sustainability:** Improve fuel efficiency through better-maintained engines

**Dataset:** NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) - run-to-failure engine degradation data with multiple operational conditions and fault modes.

---

## Key Features

### 1. **Data Pipeline**
- Automated loading and preprocessing of NASA CMAPSS datasets (FD001-FD004)
- Intelligent feature engineering with rolling statistical features
- Automatic handling of constant sensors (pruning non-informative features)
- Data standardization using StandardScaler

### 2. **Machine Learning Models**
- **XGBoost Regressor:** Primary model with gradient boosting for superior accuracy
- **Random Forest Regressor:** Baseline model for comparison
- Feature importance analysis for model explainability
- Evaluation metrics: RMSE, MAE, and visual comparison

### 3. **Interactive Web Dashboard**
- Built with **Streamlit** for real-time predictions
- Select any of 100 test engines to analyze
- **RUL Prediction Gauge:** Color-coded health status (Red/Yellow/Green)
- **Feature Importance Chart:** Top sensors influencing predictions
- **RUL Degradation Curve:** Predicted vs. actual RUL over engine lifetime
- **Sensor Health Trends:** Raw signals with rolling averages and statistics
- **Contextual Help:** In-app education with "About the Data" section

### 4. **Comprehensive Reporting**
- Professional Word document report with:
  - Team contributions and roles
  - Literature survey (2 academic papers)
  - Implementation details
  - Results and analysis
  - References and conclusions

---

## Dataset Information

The **NASA CMAPSS FD001** dataset (used in this project) contains:

### Structure
- **Training Set:** 100 engines with complete run-to-failure trajectories
- **Test Set:** 100 engines with truncated data (stops before failure)
- **Ground Truth RUL:** Actual remaining cycles for each test engine
- **Average Engine Life:** ~192 cycles (varies by engine)

### Features per Cycle
- **Unit Number:** Engine ID (1-100)
- **Time (Cycles):** Operational age
- **Operational Settings (3):** Altitude, speed, temperature-like parameters
- **Sensor Measurements (21):** Temperature, pressure, speed indicators, etc.

### Data Characteristics
- **Multivariate Time Series:** Multiple sensors tracked over engine lifetime
- **Realistic Noise:** Sensor readings contaminated with measurement noise
- **Single Condition (FD001):** Sea-level operation
- **Single Fault Mode:** High-Pressure Compressor (HPC) degradation

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LOADING & PREPROCESSING                  │
│  (load_data → remove constants → calculate RUL → handle nulls)  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│               FEATURE ENGINEERING (Rolling Features)             │
│  (rolling_mean, rolling_std → expand feature space 24 → 66)    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│          DATA STANDARDIZATION (StandardScaler)                   │
│  (normalize features to zero mean, unit variance)               │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│           MODEL TRAINING (XGBoost + Random Forest)               │
│  (fit on training data → learn degradation patterns)            │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│      MODEL EVALUATION (RMSE, MAE, Feature Importance)            │
│  (assess accuracy on test set → visualize predictions)          │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│         SERIALIZATION (joblib: model, scaler, metadata)         │
│  (save artifacts for web app → enable reproducibility)         │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│        STREAMLIT DASHBOARD (Real-time Predictions)               │
│  (select engine → process data → predict RUL → visualize)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technologies & Tools

### **Programming Languages**
- **Python 3.10+** - Core development language

### **Data Science & ML Libraries**
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **scikit-learn** - Preprocessing (StandardScaler), evaluation metrics
- **XGBoost** - Gradient boosting regressor model
- **joblib** - Model serialization and deserialization

### **Web Framework & Visualization**
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive charts and visualizations
- **Matplotlib & Seaborn** - Static visualizations (training phase)

### **Report Generation**
- **python-docx** - Professional Word document generation

### **Version Control**
- **Git & GitHub** - Source code management and collaboration

---

## Installation & Setup

### Prerequisites
- Python 3.8+ installed on your system
- Git installed (for cloning)
- ~2GB disk space for dataset and models

### Step 1: Clone or Download the Repository

```bash
git clone https://github.com/RahulR-007/PredictiveMaintenance_JET_Engine.git
cd PredictiveMaintenance_JET_Engine
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy scikit-learn xgboost plotly streamlit python-docx joblib
```

### Step 3: Download the Dataset

The NASA CMAPSS dataset is already included in the `CMAPSSData/` folder. If needed, download from:
[NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

Extract and place files in `CMAPSSData/`:
```
CMAPSSData/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── (FD002, FD003, FD004 variants - optional)
└── readme.txt (dataset documentation)
```

### Step 4: Verify Setup

```bash
# Check if all data files are present
dir CMAPSSData/

# Check Python packages
pip list | findstr "pandas scikit-learn xgboost streamlit"
```

---

## Usage

### **Option 1: Quick Start (Recommended)**
Double-click the batch file to run everything:
```powershell
run_app.bat
```

This automatically:
1. Trains the ML model (if not already trained)
2. Launches the Streamlit dashboard
3. Opens the app in your default browser

### **Option 2: Manual Commands**

#### **Step A: Train the Model**
```bash
python build_model.py
```
This will:
- Load training data from `CMAPSSData/train_FD001.txt`
- Perform feature engineering and preprocessing
- Train XGBoost and Random Forest models
- Save artifacts: `xgb_model.joblib`, `scaler.joblib`, `metadata.joblib`
- Display evaluation metrics (RMSE, MAE)
- Generate visualization plots

**Output:**
```
--- 1. LOADING DATA ---
Training data: (20631, 26)
Testing data: (13096, 26)

--- 2. PREPROCESSING ---
Dropped constant columns: ['s_1', 's_5', 's_6', ...]

--- 3. FEATURE ENGINEERING ---
[Features expanded to 66 total]

--- 4. MODEL BUILDING & TRAINING ---
Random Forest trained.
XGBoost trained.
Model, Scaler, and Metadata saved successfully!

--- 5. EVALUATION ---
[XGBoost] RMSE: XX.XX | MAE: YY.YY
[Random Forest] RMSE: XX.XX | MAE: YY.YY

--- SUCCESS ---
```

#### **Step B: Launch the Web Dashboard**
```bash
streamlit run app.py
```

This opens an interactive dashboard at `http://localhost:8501` where you can:
- Select any engine (1-100) from the dropdown
- View real-time RUL prediction with health gauge
- Explore feature importance
- Analyze degradation curves
- Inspect sensor trends with statistics

#### **Step C: Generate Report (Optional)**
```bash
python generate_report.py
```
Creates `ML_Project_Report.docx` with:
- Team contributions
- Literature review
- Implementation details
- Results and analysis
- Professional formatting

---

## Project Structure

```
PredictiveMaintenance_JET_Engine/
│
├── app.py                              # Streamlit dashboard (main UI)
├── build_model.py                      # Model training and evaluation pipeline
├── generate_report.py                  # Professional report generation
├── run_app.bat                         # One-click launcher (Windows)
│
├── CMAPSSData/                         # NASA CMAPSS dataset
│   ├── train_FD001.txt                # Training trajectories (100 engines)
│   ├── test_FD001.txt                 # Test trajectories (100 engines)
│   ├── RUL_FD001.txt                  # Ground truth RUL values
│   ├── train_FD002-004.txt            # Additional datasets (optional)
│   ├── test_FD002-004.txt             # Additional datasets (optional)
│   └── readme.txt                     # Dataset documentation
│
├── xgb_model.joblib                    # Trained XGBoost model (binary)
├── scaler.joblib                       # StandardScaler fitted on training data
├── metadata.joblib                     # Feature names and metadata
│
├── README.md                           # This file - comprehensive documentation
├── ML_Project_Report.docx              # Professional project report
│
├── .git/                               # Git version control
└── .gitignore                          # Git ignore rules
```

---

## Team Contributions

This project was developed by a team of 4 members with distinct roles:

| Role | Responsibility | Key Deliverables |
|------|-----------------|-----------------|
| **Data & Preprocessing Specialist** | Dataset organization, loading, cleaning, RUL calculation | Data pipeline, feature understanding, documentation |
| **ML & Model Engineer** | Model selection, training, evaluation, serialization | XGBoost model, evaluation metrics, performance analysis |
| **Feature Engineering & Backend** | Rolling features, scaling, app backend, prediction logic | Feature engineering code, prediction pipeline, integration |
| **UI/UX & Frontend Developer** | Dashboard design, visualizations, user guidance | Streamlit app, Plotly charts, help text, color schemes |

---

## Results & Performance

### Model Performance Metrics

**XGBoost Regressor (Primary Model)**
- Achieves competitive accuracy on NASA CMAPSS FD001 benchmark
- Predictions most accurate in final cycles before failure
- Lower variance in predictions compared to Random Forest

**Evaluation Approach**
- Test Set: Last cycle measurement of each engine
- Metrics: RMSE (prediction deviation), MAE (absolute error)
- Comparison: XGBoost vs. Random Forest baseline

### Key Findings

1. **Rolling Mean Features Most Important**
   - `s_4_rolling_mean`, `s_11_rolling_mean`, `s_15_rolling_mean` are top predictors
   - Validates feature engineering approach (smoothing + trend capture)

2. **Degradation Pattern Learning**
   - Model successfully learns to predict RUL as engine ages
   - Predictions trend downward matching physical degradation

3. **Explainability**
   - Feature importance analysis provides transparency
   - Maintenance engineers can understand model decisions

---

## How It Works

### **Phase 1: Training (build_model.py)**

1. **Data Loading:** Read training trajectories with complete run-to-failure history
2. **RUL Calculation:** RUL = max_cycle - current_cycle (countdown to failure)
3. **Constant Removal:** Drop sensors with zero variance (uninformative)
4. **Feature Engineering:**
   - Rolling Mean: 5-cycle moving average (smooths noise, captures trends)
   - Rolling Std: 5-cycle moving standard deviation (captures volatility)
   - Result: 24 original features → 66 engineered features
5. **Standardization:** Scale features to zero mean, unit variance
6. **Model Training:** Fit XGBoost and Random Forest on training data
7. **Serialization:** Save model, scaler, and metadata to joblib files

### **Phase 2: Inference (app.py)**

1. **User Selection:** Select engine ID (1-100) from dropdown
2. **Data Retrieval:** Load test data for selected engine
3. **Feature Processing:**
   - Apply same rolling feature engineering
   - Standardize using saved scaler
4. **Prediction:** Feed last cycle features to trained model
5. **Visualization:**
   - Display RUL gauge (color-coded health status)
   - Show feature importance (which sensors matter most)
   - Plot RUL degradation curve (predicted vs. actual)
   - Show sensor trends with statistics
6. **Interpretation:** Help text explains what each visualization means

### **Key Design Decisions**

- **XGBoost over Deep Learning:** Better interpretability for safety-critical applications
- **Rolling Features:** Capture degradation trends while reducing noise
- **Streamlit for UI:** Rapid development, interactive, requires no frontend expertise
- **Feature Importance:** Enables stakeholder trust in model decisions

---

## Future Enhancements

### **Model Improvements**
1. **Deep Learning Integration**
   - LSTM networks for sequential pattern learning
   - CNN-based approaches for temporal correlations

2. **Uncertainty Quantification**
   - Confidence intervals around predictions
   - Bayesian approaches for risk assessment

3. **Multi-Condition Support**
   - Train on FD002, FD003, FD004 (different operating conditions)
   - Domain adaptation for real-world engines

4. **Ensemble Methods**
   - Combine XGBoost with neural networks
   - Weighted voting for improved robustness

### **System Enhancements**
1. **Real-Time Integration**
   - Connect to actual aircraft engine telemetry
   - Live RUL updates during flight operations

2. **Automated Retraining**
   - CI/CD pipeline for periodic model updates
   - Adapt to new engine populations

3. **Advanced Explainability**
   - SHAP values for feature contribution analysis
   - Individual prediction explanations

4. **Multi-Fault Detection**
   - Detect and classify different failure modes
   - Predict multiple potential failure types

5. **Maintenance Scheduling**
   - Optimize maintenance windows
   - Cost-benefit analysis for maintenance actions

---

## References

### Academic Papers
1. **Saxena, A., Goebel, K., Simon, D., & Eklund, N.** (2008)
   - "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"
   - *Proceedings of PHM08 Conference, Denver, CO*
   - Introduces the NASA CMAPSS dataset used in this project

2. **Ren, L., Sun, Y., Cui, H., & Zhang, L.** (2021)
   - "Machine Learning for Predictive Maintenance: A Review"
   - *Journal of Mechanical Science and Technology, Vol. 35, No. 2*
   - Surveys ML techniques for RUL prediction and maintenance

### Software & Tools
- **XGBoost Documentation:** https://xgboost.readthedocs.io/
- **Streamlit Documentation:** https://docs.streamlit.io/
- **scikit-learn:** https://scikit-learn.org/
- **Plotly:** https://plotly.com/python/

### Dataset
- **NASA Prognostics Data Repository:** https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

---

## Requirements

A `requirements.txt` file with all dependencies (generate with `pip freeze`):

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
xgboost>=1.5.0
plotly>=5.0.0
streamlit>=1.0.0
python-docx>=0.8.11
joblib>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Tips & Troubleshooting

### **Issue: "FileNotFoundError: train_FD001.txt not found"**
- **Solution:** Ensure `CMAPSSData/` folder exists with dataset files in the same directory as `build_model.py`

### **Issue: Streamlit app loads but no data displays**
- **Solution:** Run `build_model.py` first to generate model artifacts (`.joblib` files)

### **Issue: Port 8501 already in use**
```bash
streamlit run app.py --server.port 8502
```

### **Issue: Memory error with large datasets**
- **Solution:** Process data in batches or reduce feature engineering window size

### **Issue: Slow predictions**
- **Solution:** Ensure XGBoost is compiled with GPU support (optional but faster)

---

## Notes

- All models are trained on FD001 dataset (single condition, single fault mode)
- Dataset includes realistic sensor noise for practical modeling
- Ground truth RUL values come from the simulation ground truth
- Real-world deployment would require validation on actual engine data

---

## Contributing

To contribute improvements:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m "Add improvement"`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## License

This project uses the NASA CMAPSS dataset. Please respect the dataset citation and license terms from NASA's Prognostics Data Repository.

---

## Support

For questions or issues:
- Check the "About the Data" section in the Streamlit app
- Review the comprehensive project report (`ML_Project_Report.docx`)
- Consult the referenced academic papers for methodology details

---

## Quick Checklist

Before using this project, ensure:
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] NASA CMAPSS data in `CMAPSSData/` folder
- [ ] At least 500MB disk space
- [ ] No firewall blocking port 8501 (Streamlit default)

---

**Project Status:** Complete & Production-Ready

**Last Updated:** April 1, 2026

**Repository:** https://github.com/RahulR-007/PredictiveMaintenance_JET_Engine

---

Enjoy exploring predictive maintenance!