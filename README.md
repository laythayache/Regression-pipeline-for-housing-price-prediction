# Housing Price Prediction Pipeline

A complete machine learning pipeline for predicting house prices using synthetic data. This project demonstrates end-to-end ML workflow from data generation to deployment, including an interactive GUI for price predictions.

## Project Overview

This pipeline implements:
- Synthetic housing data generation with realistic features and quality issues
- Automated data preprocessing and feature engineering
- Model training with multiple algorithms and hyperparameter tuning
- Comprehensive model evaluation with visualizations
- Production-ready deployment package
- Interactive GUI for non-technical users

### Performance Summary
- Model Accuracy: R² = 0.576 (explains 57.6% of price variance)
- Prediction Error: RMSE = $88,859 (median error: 5.4%)
- Data Quality: 1620 → 1197 clean records (26% improvement)
- Feature Engineering: 9 → 45 → 15 optimized features

## Project Structure

```
housing-price-prediction/
├── data.py                     # Synthetic data generation
├── README.md                   # This file - project overview
├── data/                       # Generated datasets
├── src/                        # Core ML pipeline modules
│   ├── preprocessing.py        # Data cleaning
│   ├── feature_engineering.py # Feature creation and selection
│   ├── model_selection_training.py # Model training
│   ├── evaluation.py          # Model evaluation
│   └── deployment_monitoring.py # Deployment setup
├── app/                        # Applications
│   ├── main.py                # Complete ML pipeline
│   ├── gui.py                 # Interactive price prediction
│   └── README.md              # Application instructions
├── models/                     # Trained models
├── plots/                      # Evaluation charts
├── reports/                    # Performance reports
└── deployment/                 # Production package
```

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Option 1: Interactive GUI (For Users)
```bash
# Generate data and train model
python data.py
cd app
python main.py

# Launch price prediction interface
python gui.py
```
Use the GUI to input house characteristics and get instant price predictions.

### Option 2: Complete Pipeline (For Data Scientists)
```bash
# Generate synthetic dataset
python data.py

# Run full ML pipeline
cd app
python main.py
```
This runs the complete pipeline with model training, evaluation, and deployment setup.

### Option 3: Data Generation Only
```bash
python data.py
```
Creates synthetic housing dataset for experimentation.

## Key Features

### Interactive GUI Application
- User-friendly interface for price predictions
- Input validation with helpful error messages
- Automatic model loading and feature engineering
- Professional output with confidence intervals

### Complete ML Pipeline
- End-to-end automation from raw data to deployed model
- 5-step process: preprocessing, feature engineering, training, evaluation, deployment
- Comprehensive logging and error handling
- Professional reports and visualizations

### Data Generation
- 1500+ synthetic housing records with 9 features
- Intentional quality issues: 15% missing values, 8% duplicates
- Realistic price relationships and feature correlations

### Model Training
- Tests 10+ algorithms including Ridge, Random Forest, Gradient Boosting
- Cross-validation and hyperparameter tuning
- Automated model selection based on performance

## Expected Results

### Model Performance
- Algorithm: Ridge Regression (tuned)
- R² Score: 0.576 (explains 57.6% of price variance)
- RMSE: $88,859 (typical prediction error)
- 67% of predictions within 10% of actual price
- 90% of predictions within 25% of actual price

### Generated Outputs
- Trained model files in `models/` directory
- 7 evaluation plots in `plots/` directory
- Performance reports in `reports/` directory
- Deployment package in `deployment/` directory

## Use Cases

### Real Estate Professionals
Use the GUI for quick property valuations and client consultations.

### Data Scientists
Study the complete pipeline for learning ML best practices and workflow automation.

### Students and Researchers
Educational resource for understanding end-to-end machine learning projects.

## Documentation

- `README.md` - This file, project overview
- `app/README.md` - Detailed application instructions
- `reports/evaluation_report.txt` - Model performance analysis
- `logs/pipeline.log` - Execution details and debugging

## Common Issues

1. **Import errors**: Run commands from correct directories
2. **No model found**: Run `python main.py` before using GUI
3. **Missing data**: Run `python data.py` to generate dataset
4. **GUI won't open**: Check tkinter installation

This project provides a practical example of a complete machine learning workflow with both technical depth for learning and user-friendly tools for practical application.
