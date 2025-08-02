# Housing Price Prediction Application

This directory contains two main applications for the housing price prediction system.

## Files

### main.py
Complete ML pipeline that handles the entire workflow from data preprocessing to model deployment. Runs automatically with minimal user input and generates comprehensive reports.

### gui.py
Interactive graphical interface for getting house price predictions. Designed for end users who want instant price estimates without technical knowledge.

## Quick Start

### Complete Pipeline (Data Scientists)
```bash
cd app
python main.py
```
This runs the full machine learning pipeline including data processing, feature engineering, model training, evaluation, and deployment setup. Takes about 10-15 seconds to complete.

### Price Prediction GUI (Everyone)
```bash
cd app
python gui.py
```
Opens an interactive window where you can input house characteristics and get instant price predictions. Requires a trained model from running main.py first.

## Requirements

### For main.py
- All source modules in ../src/ directory
- Housing data in ../data/housing_data.csv (or run ../data.py first)

### For gui.py  
- Trained model file in ../models/ directory
- tkinter (included with most Python installations)

## GUI Usage

The interface has 8 input fields:
- Living Area (square feet)
- Number of Floors (1.0, 1.5, 2.0, etc.)
- Condition (1-5 scale: Poor to Excellent)
- Grade (1-13 scale: Construction quality)
- Bedrooms
- Bathrooms
- Age (years)
- Lot Size (square feet)

Example: A 2000 sq ft house with 2 floors, condition 4, grade 8, 3 bedrooms, 2.5 bathrooms, 15 years old, on an 8000 sq ft lot might be predicted at $485,000.

## Output Files

After running main.py, these directories contain results:
- ../data/ - Processed and engineered datasets
- ../models/ - Trained model files
- ../plots/ - Evaluation charts and visualizations
- ../reports/ - Performance analysis and metrics
- ../deployment/ - Production-ready model package

## Performance Expectations

- Model accuracy: R² = 0.576 (explains 57.6% of price variance)
- Typical error: $88,859 RMSE (median error 5.4%)
- 67% of predictions within 10% of actual price
- 90% of predictions within 25% of actual price

## Troubleshooting

**Import errors**: Make sure you're running from the app/ directory

**No trained model found**: Run `python main.py` before using the GUI

**Data file not found**: Run `python ../data.py` to generate the dataset

**GUI won't open**: Check if tkinter is installed with `python -c "import tkinter"`

**Pipeline runs slowly**: This is normal for the first run as it trains multiple models

Check ../logs/pipeline.log for detailed error information and execution details.
